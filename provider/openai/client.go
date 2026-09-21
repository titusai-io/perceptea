// Package openai speaks to any OpenAI-compatible /chat/completions endpoint —
// OpenAI itself, OpenRouter, DeepInfra, Z.ai, a local vLLM — and implements the
// [classifier.Scorer] and [classifier.Generator] interfaces on top of it.
//
// The package depends only on the standard library: the request and response
// documents are hand-rolled JSON structs rather than a vendored SDK.
//
// A [Client] is safe for concurrent use and expects to be used that way: one
// evaluation fans out into many simultaneous Score calls against the same
// client. The only mutable state is the negotiated structured-output level,
// which is held in an atomic and only ever moves in the degrading direction.
package openai

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"time"

	"github.com/titusai-io/perceptea/classifier"
)

// A *Client is both halves of the classifier's backend contract.
var (
	_ classifier.Scorer    = (*Client)(nil)
	_ classifier.Generator = (*Client)(nil)
)

// ErrNoAPIKey is returned by [New] when Config.APIKey is blank.
var ErrNoAPIKey = errors.New("openai: no API key configured")

const (
	// defaultBaseURL is OpenAI's own endpoint, used when Config.BaseURL is
	// blank.
	defaultBaseURL = "https://api.openai.com/v1"
	// defaultMaxRetries is the number of retries — not attempts — used when
	// Config.MaxRetries is zero.
	defaultMaxRetries = 2
	// defaultTimeout bounds one whole HTTP attempt, including reading the
	// response body.
	defaultTimeout = 120 * time.Second
)

// Config configures a [Client]. Only APIKey is required.
type Config struct {
	// APIKey is the bearer token for the endpoint. Required.
	APIKey string
	// BaseURL is the API root, without the /chat/completions suffix. It
	// defaults to https://api.openai.com/v1, and trailing slashes are
	// trimmed so that both forms of a pasted URL work.
	BaseURL string
	// Model is used for any request that does not name a model of its own.
	Model string
	// HTTPClient overrides the transport. The default client has sane
	// timeouts and enough idle connections per host for a fanned-out
	// evaluation.
	HTTPClient *http.Client
	// MaxRetries is the number of retries after the first attempt. Zero
	// selects the default of 2 (3 attempts in total); a negative value
	// disables retrying entirely.
	MaxRetries int
	// Referer sets the optional HTTP-Referer header OpenRouter uses for
	// attribution.
	Referer string
	// Title sets the optional X-Title header OpenRouter uses for
	// attribution.
	Title string
	// Logger receives debug records for every attempt. It defaults to
	// slog.Default(). The API key is never written to it.
	Logger *slog.Logger
}

// Client calls an OpenAI-compatible chat completions endpoint. Create one with
// [New]; the zero value is not usable. It is safe for concurrent use.
type Client struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
	maxRetries int
	referer    string
	title      string
	log        *slog.Logger

	// level is the structured-output level currently believed to work,
	// shared by every in-flight call so that a provider that rejects
	// response_format is discovered once rather than once per call. It only
	// ever moves towards the plainer end; see downgrade.
	level atomic.Int32

	// sleep waits out a retry backoff. It is a field so that tests can run
	// the retry logic without spending the wall clock.
	sleep func(context.Context, time.Duration) error
}

// New validates cfg and returns a ready client. It returns [ErrNoAPIKey] when
// no key is configured.
func New(cfg Config) (*Client, error) {
	key := strings.TrimSpace(cfg.APIKey)
	if key == "" {
		return nil, ErrNoAPIKey
	}

	base := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if base == "" {
		base = defaultBaseURL
	}
	parsed, err := url.Parse(base)
	if err != nil {
		return nil, fmt.Errorf("openai: invalid base URL %q: %w", base, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("openai: invalid base URL %q: want an http or https URL", base)
	}
	if parsed.Host == "" {
		return nil, fmt.Errorf("openai: invalid base URL %q: no host", base)
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = newHTTPClient()
	}

	retries := cfg.MaxRetries
	switch {
	case retries < 0:
		retries = 0
	case retries == 0:
		retries = defaultMaxRetries
	}

	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &Client{
		apiKey:     key,
		baseURL:    base,
		model:      strings.TrimSpace(cfg.Model),
		httpClient: httpClient,
		maxRetries: retries,
		referer:    cfg.Referer,
		title:      cfg.Title,
		log:        logger,
		sleep:      sleepContext,
	}, nil
}

// newHTTPClient builds the default transport: bounded handshakes, a generous
// overall timeout because a cold model can be slow, and a raised idle
// connection pool because an evaluation fans out.
func newHTTPClient() *http.Client {
	return &http.Client{
		Timeout: defaultTimeout,
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   10 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          100,
			MaxIdleConnsPerHost:   64,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
		},
	}
}

// resolveModel picks the model for one call: the request's own, else the
// client's fallback.
func (c *Client) resolveModel(requested string) (string, error) {
	model := strings.TrimSpace(requested)
	if model == "" {
		model = c.model
	}
	if model == "" {
		return "", errors.New("openai: no model: set Config.Model or name one on the request")
	}
	return model, nil
}

// redact removes the API key from text that came back from the provider. A few
// gateways echo the key into their error messages; nothing this package
// produces should carry it onwards.
func (c *Client) redact(s string) string {
	if c.apiKey == "" || !strings.Contains(s, c.apiKey) {
		return s
	}
	return strings.ReplaceAll(s, c.apiKey, "[REDACTED]")
}

// sleepContext is the production sleep: wait out d, or give up early when the
// context is done.
func sleepContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

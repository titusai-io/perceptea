// Package inference speaks the OpenAI-compatible /chat/completions protocol
// and implements the [classifier.Scorer] and [classifier.Generator] interfaces
// on top of it.
//
// That protocol is a wire format rather than a product: DeepInfra, OpenRouter,
// Z.ai and local servers such as Ollama and vLLM all expose it, and any of
// them can be the endpoint. Which one is a matter of configuration — the
// caller always names it, and this package has no opinion about the answer.
//
// The package depends only on the standard library: the request and response
// documents are hand-rolled JSON structs rather than a vendored SDK.
//
// # Two scorers
//
// There are two ways to get a probability out of a model, and Config.Scorer
// picks between them for the life of a client.
//
// The chat scorer — "chat", and the default — asks the model to answer with
// {"p": 0.87} and parses the number out of the reply. It asks nothing of a
// provider beyond chat completions, so it works everywhere, and it inherits
// the model's writing habits: a written probability clusters on 0.8, 0.9 and
// 0.95, because those are the numbers models write, and the shape of a set of
// scores is partly the shape of that habit.
//
// The logprob scorer — "logprob" — asks the same question as a one-word Yes/No
// decision, with logprobs and a one-token cap, and never reads the word that
// comes back. It reads the distribution the word was to be sampled from, and
// computes P = exp(l_yes) / (exp(l_yes) + exp(l_no)) over the two branches.
// That is continuous, it is the model's estimate rather than its description
// of one, and it decodes one token where the other decodes ten, which on a
// reply this short is most of the latency. It is the better measurement
// wherever the provider supports it, and the calibration work — thresholds,
// reliability curves — is what tells you whether it is better for a given
// model.
//
// What it requires is an endpoint that accepts logprobs and top_logprobs on a
// chat completion and returns the top-k distribution for the generated token.
// Endpoints differ, models on one endpoint differ, and a gateway may strip the
// fields on the way through. When the logprobs do not come back the call fails
// with [ErrNoLogprobs], which names the setting and the model: there is no
// fall back to the chat scorer, because a service that silently changes
// estimator keeps producing numbers while quietly changing what they mean.
// See [ErrNoLogprobs] and [ErrNoDecisionToken].
//
// Both scorers send the same two-message prompt layout — a shared system
// prefix and a per-candidate user suffix — so both get the same prefix
// caching from an endpoint that offers it.
//
// # Concurrency
//
// A [Client] is safe for concurrent use and expects to be used that way: one
// evaluation fans out into many simultaneous Score calls against the same
// client. The only mutable state is the negotiated structured-output level,
// which is held in an atomic and only ever moves in the degrading direction.
// The logprob scorer sends no response_format at all and so never touches it.
package inference

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
var ErrNoAPIKey = errors.New("inference: no API key configured")

// ErrNoBaseURL is returned by [New] when Config.BaseURL is blank. There is no
// default endpoint: nothing here picks a service on the operator's behalf.
var ErrNoBaseURL = errors.New("inference: no base URL configured")

const (
	// defaultMaxRetries is the number of retries — not attempts — used when
	// Config.MaxRetries is zero.
	defaultMaxRetries = 2
	// defaultTimeout bounds one whole HTTP attempt, including reading the
	// response body.
	defaultTimeout = 120 * time.Second
)

// Config configures a [Client]. APIKey and BaseURL are required.
type Config struct {
	// APIKey is the bearer token for the endpoint. Required.
	APIKey string
	// BaseURL is the API root, without the /chat/completions suffix.
	// Required: there is no default endpoint. It must be an absolute http
	// or https URL, and trailing slashes are trimmed so that both forms of
	// a pasted URL work.
	BaseURL string
	// Model is used for any request that does not name a model of its own.
	Model string
	// Scorer selects how [Client.Score] obtains a probability. Empty — the
	// default — is the chat scorer, so a caller that has never heard of this
	// field gets exactly the behaviour it had before the field existed.
	//
	// The two values are "chat" and "logprob", and they name two different
	// measurements of the same thing:
	//
	//   - "chat" asks the model to write a probability as JSON and parses the
	//     number out of the reply. It works against any endpoint that serves
	//     chat completions, which is why it is the default. What it returns is
	//     quantised by the model's writing habits: models type 0.8, 0.9 and
	//     0.95 and hardly ever 0.87, so the distribution owes as much to how a
	//     model phrases a number as to what it believes.
	//
	//   - "logprob" asks for a one-word Yes/No decision with logprobs, and
	//     computes the probability from the two branches' log probabilities
	//     rather than from any word the model wrote. It is continuous where
	//     the written number clusters on round values, it is the model's
	//     estimate rather than its description of one, and it decodes a single
	//     token, which on a reply this short is most of the latency.
	//
	// The logprob scorer needs more of the provider than the chat scorer
	// does: the endpoint must accept logprobs and top_logprobs on a chat
	// completion and return the top-k distribution for the generated token.
	// Not every endpoint, and not every model on an endpoint that does,
	// will. When one will not, the call fails with [ErrNoLogprobs], naming
	// the setting and the model; it does not fall back to the chat scorer and
	// it does not return a neutral 0.5. Both of those would keep answering
	// with numbers that no longer mean what the configuration says they mean.
	// A model that returns logprobs but will not answer Yes or No fails the
	// same way, with [ErrNoDecisionToken].
	//
	// Both scorers send the same two-message prompt layout, so both get the
	// same prefix caching; only the instruction and what is read back differ.
	// The two calibrate differently, though — same question, different
	// estimator — so a threshold tuned against one is not a threshold for the
	// other.
	Scorer string
	// ReasoningEffort is sent as reasoning_effort on every call. Empty — the
	// default — sends no reasoning field at all, which is what a provider
	// that has never heard of one expects. The usual values are "none",
	// "low", "medium" and "high"; the value is passed through rather than
	// checked here, because which of them an endpoint honours is the
	// endpoint's business and an operator's typo belongs to whoever reads
	// the configuration.
	ReasoningEffort string
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

// Client calls one OpenAI-compatible chat completions endpoint. Create one with
// [New]; the zero value is not usable. It is safe for concurrent use.
type Client struct {
	apiKey          string
	baseURL         string
	model           string
	scorer          string
	reasoningEffort string
	httpClient      *http.Client
	maxRetries      int
	referer         string
	title           string
	log             *slog.Logger

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
// no key is configured, [ErrNoBaseURL] when no endpoint is, and an error
// naming both accepted values when Config.Scorer is a word this package does
// not know.
//
// Which scorer [Client.Score] runs is decided here, once, rather than on each
// call: the choice is configuration, and a client that could change estimator
// between two candidates of one question would be returning probabilities
// from two different scales in one answer.
func New(cfg Config) (*Client, error) {
	key := strings.TrimSpace(cfg.APIKey)
	if key == "" {
		return nil, ErrNoAPIKey
	}

	base := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if base == "" {
		return nil, ErrNoBaseURL
	}
	parsed, err := url.Parse(base)
	if err != nil {
		return nil, fmt.Errorf("inference: invalid base URL %q: %w", base, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("inference: invalid base URL %q: want an http or https URL", base)
	}
	if parsed.Host == "" {
		return nil, fmt.Errorf("inference: invalid base URL %q: no host", base)
	}

	// An unknown scorer is rejected rather than quietly read as the default.
	// A typo that fell through to "chat" would answer every request with
	// numbers from the estimator the operator had just decided not to use,
	// and nothing anywhere would say so.
	scorer := strings.TrimSpace(cfg.Scorer)
	switch scorer {
	case "":
		scorer = scorerChat
	case scorerChat, scorerLogprob:
	default:
		return nil, fmt.Errorf("inference: %q is not a scorer: expected %q or %q (%s)",
			cfg.Scorer, scorerChat, scorerLogprob, envScorer)
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
		apiKey:          key,
		baseURL:         base,
		model:           strings.TrimSpace(cfg.Model),
		scorer:          scorer,
		reasoningEffort: strings.TrimSpace(cfg.ReasoningEffort),
		httpClient:      httpClient,
		maxRetries:      retries,
		referer:         cfg.Referer,
		title:           cfg.Title,
		log:             logger,
		sleep:           sleepContext,
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
		return "", errors.New("inference: no model: set Config.Model or name one on the request")
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

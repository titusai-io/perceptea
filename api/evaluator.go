package api

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
	"github.com/titusai-io/perceptea/provider/inference"
)

const (
	// defaultMaxRetries is how often the provider client retries a call that
	// failed in a way worth retrying. One evaluation makes many calls, so a
	// generous number multiplies badly; two is enough for a blip.
	defaultMaxRetries = 2
	// attribution identifies this service to providers that record it.
	attributionURL   = "https://github.com/titusai-io/perceptea"
	attributionTitle = "Perceptea"
	// maxCachedClients bounds the client cache. Every distinct base URL,
	// model and key a caller sends would otherwise be one more entry, for
	// ever, in a service that lets callers supply their own; the cache is a
	// performance aid, not a registry.
	maxCachedClients = 64
)

// defaultNewEvaluator is the factory used when Options.NewEvaluator is nil.
func defaultNewEvaluator(logger *slog.Logger) func(Settings) (Evaluator, error) {
	return newEvaluatorFactory(logger).newEvaluator
}

// evaluatorFactory builds one request's evaluator.
//
// It keeps one *http.Client for the life of the server, because evaluations
// fan out into many small calls to the same host and a fresh client per
// request would throw away every pooled connection. It keeps the provider
// clients too: an *inference.Client holds the structured-output level it has
// negotiated, so that a provider which rejects response_format is discovered
// once. Rebuilt per request, that discovery was repaid in full on every
// request — a choice with 8 options cost 23 upstream calls instead of 8.
type evaluatorFactory struct {
	logger *slog.Logger
	client *http.Client
	cache  *clientCache
	// newClient builds one provider client. It is a field so that a test can
	// count how often the factory actually reaches for a new one.
	newClient func(inference.Config) (*inference.Client, error)
}

func newEvaluatorFactory(logger *slog.Logger) *evaluatorFactory {
	return &evaluatorFactory{
		logger: logger,
		client: &http.Client{
			// No client-level timeout: every request already carries the
			// server's deadline in its context, and one timeout is easier to
			// reason about than two.
			Transport: pooledTransport(),
		},
		cache:     newClientCache(maxCachedClients),
		newClient: inference.New,
	}
}

// newEvaluator answers one request's worth of settings.
func (f *evaluatorFactory) newEvaluator(st Settings) (Evaluator, error) {
	c, err := f.cache.get(clientKey(st), func() (*inference.Client, error) {
		return f.newClient(inference.Config{
			APIKey:          st.APIKey,
			BaseURL:         st.BaseURL,
			Model:           st.Model,
			ReasoningEffort: st.ReasoningEffort,
			HTTPClient:      f.client,
			MaxRetries:      defaultMaxRetries,
			Referer:         attributionURL,
			Title:           attributionTitle,
			Logger:          f.logger,
		})
	})
	if err != nil {
		return nil, err
	}
	n := st.MaxConcurrency
	if n < 1 {
		n = config.DefaultMaxConcurrency
	}
	// The classifier wrapper is cheap and holds no state worth keeping, so
	// only the client below it is cached; MaxConcurrency is therefore not
	// part of the cache key.
	return classifier.New(c, classifier.WithMaxConcurrency(n)), nil
}

// clientKey identifies a provider client by everything it is built from.
//
// The key is hashed rather than assembled: a map keyed on plaintext
// credentials is a credential store nobody asked for, and would put every key
// the service has ever been handed into a heap dump. Each part is
// length-prefixed so that two different splits cannot produce one key.
//
// The reasoning effort is in the key although today it cannot vary: it comes
// from the server's configuration, so every request in a process resolves the
// same one and the key gains no entries by including it. It is here because
// the client is *built* from it — it goes into every request body the client
// sends — and the rule this key follows is that everything the client is made
// of is in it. MaxConcurrency, which the client is not made of, stays out.
// Were a per-request effort ever allowed, leaving it out would quietly serve
// the first caller's client to the second.
func clientKey(st Settings) string {
	sum := sha256.New()
	for _, part := range []string{st.BaseURL, st.Model, st.APIKey, st.ReasoningEffort} {
		fmt.Fprintf(sum, "%d:%s", len(part), part)
	}
	return hex.EncodeToString(sum.Sum(nil))
}

// clientCache is a bounded, concurrency-safe cache of provider clients, least
// recently used first out. Evicting a client only costs it its negotiated
// structured-output level, which the next call rediscovers.
type clientCache struct {
	mu    sync.Mutex
	max   int
	order *list.List // front is most recently used; values are *cacheEntry
	items map[string]*list.Element
}

type cacheEntry struct {
	key    string
	client *inference.Client
}

func newClientCache(max int) *clientCache {
	if max < 1 {
		max = 1
	}
	return &clientCache{
		max:   max,
		order: list.New(),
		items: make(map[string]*list.Element, max),
	}
}

// get returns the client cached under key, building and storing one with
// build if there is none. A build failure is returned as-is and cached not at
// all: a missing key must fail every time, not once.
func (c *clientCache) get(key string, build func() (*inference.Client, error)) (*inference.Client, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if el, ok := c.items[key]; ok {
		c.order.MoveToFront(el)
		return el.Value.(*cacheEntry).client, nil
	}

	// Building is local work — parsing and validation, no I/O — so holding
	// the lock across it costs nothing and saves two callers racing to build
	// the same client and losing one of them.
	client, err := build()
	if err != nil {
		return nil, err
	}
	c.items[key] = c.order.PushFront(&cacheEntry{key: key, client: client})
	for c.order.Len() > c.max {
		oldest := c.order.Back()
		c.order.Remove(oldest)
		delete(c.items, oldest.Value.(*cacheEntry).key)
	}
	return client, nil
}

// len reports how many clients are cached.
func (c *clientCache) len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.order.Len()
}

// pooledTransport is the default transport with room for a fan-out: the
// standard two idle connections per host would serialise a burst of scoring
// calls behind new TLS handshakes.
func pooledTransport() http.RoundTripper {
	base, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		return http.DefaultTransport
	}
	t := base.Clone()
	t.MaxIdleConns = 256
	t.MaxIdleConnsPerHost = 64
	t.IdleConnTimeout = 90 * time.Second
	return t
}

package api

import (
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/titusai-io/perceptea/provider/inference"
)

// countingFactory is the real factory with a counter over the one step that
// reaches for a new provider client.
func countingFactory(t *testing.T) (*evaluatorFactory, *[]inference.Config) {
	t.Helper()
	f := newEvaluatorFactory(slog.New(slog.DiscardHandler))
	var built []inference.Config
	inner := f.newClient
	f.newClient = func(cfg inference.Config) (*inference.Client, error) {
		built = append(built, cfg)
		return inner(cfg)
	}
	return f, &built
}

// TestDefaultEvaluatorFactory checks the wiring of the real factory without
// letting it make a call: building a client is offline, and no Evaluate
// happens here.
func TestDefaultEvaluatorFactory(t *testing.T) {
	f, built := countingFactory(t)

	if _, err := f.newEvaluator(Settings{BaseURL: "https://inference.example/v1"}); !errors.Is(err, inference.ErrNoAPIKey) {
		t.Errorf("building an evaluator without a key = %v, want ErrNoAPIKey", err)
	}

	ev, err := f.newEvaluator(Settings{
		APIKey:         testKey,
		BaseURL:        "https://inference.example/v1",
		Model:          "probe-1",
		MaxConcurrency: 4,
	})
	if err != nil {
		t.Fatalf("building an evaluator: %v", err)
	}
	if ev == nil {
		t.Fatal("the factory returned no evaluator")
	}

	if len(*built) != 2 {
		t.Fatalf("built %d clients, want 2", len(*built))
	}
	got := (*built)[1]
	if got.APIKey != testKey {
		t.Errorf("Config.APIKey = %q, want the request's key", got.APIKey)
	}
	if got.BaseURL != "https://inference.example/v1" {
		t.Errorf("Config.BaseURL = %q", got.BaseURL)
	}
	if got.Model != "probe-1" {
		t.Errorf("Config.Model = %q", got.Model)
	}
	// Unset is the default and has to reach the client as unset: the client
	// sends no reasoning field at all for an empty one.
	if got.ReasoningEffort != "" {
		t.Errorf("Config.ReasoningEffort = %q, want it unset", got.ReasoningEffort)
	}
	// One evaluation is many calls, so the retry count multiplies: it is a
	// setting, not a default to drift.
	if got.MaxRetries != defaultMaxRetries {
		t.Errorf("Config.MaxRetries = %d, want %d", got.MaxRetries, defaultMaxRetries)
	}
	// OpenRouter bills and ranks by these two; losing them is invisible until
	// somebody looks at a dashboard.
	if got.Referer != attributionURL {
		t.Errorf("Config.Referer = %q, want %q", got.Referer, attributionURL)
	}
	if got.Title != attributionTitle {
		t.Errorf("Config.Title = %q, want %q", got.Title, attributionTitle)
	}
	if got.Logger == nil {
		t.Error("Config.Logger is nil: the provider would log to the default logger")
	}
	// A failed build is not cached: a request with no key must fail every
	// time, not once.
	if n := f.cache.len(); n != 1 {
		t.Errorf("cached %d clients, want only the one that built", n)
	}
}

// The effort is a setting the provider client is built from, so it has to
// reach it — and has to be part of the cache key, or a client built for one
// effort would be handed to a request that resolved another. It cannot vary
// per request today, which is exactly why the key is the cheap place to keep
// the invariant: including it costs no cache entries while nothing varies,
// and costs a silent wrong answer if anything ever does.
func TestTheReasoningEffortReachesTheClientAndTheCacheKey(t *testing.T) {
	f, built := countingFactory(t)
	base := Settings{APIKey: testKey, BaseURL: "https://inference.example/v1", Model: "probe-1"}

	withEffort := base
	withEffort.ReasoningEffort = "none"
	if _, err := f.newEvaluator(withEffort); err != nil {
		t.Fatalf("building an evaluator: %v", err)
	}
	if len(*built) != 1 {
		t.Fatalf("built %d clients, want 1", len(*built))
	}
	if got := (*built)[0].ReasoningEffort; got != "none" {
		t.Errorf("Config.ReasoningEffort = %q, want %q", got, "none")
	}

	// The same settings reuse it...
	if _, err := f.newEvaluator(withEffort); err != nil {
		t.Fatalf("building an evaluator: %v", err)
	}
	if len(*built) != 1 {
		t.Errorf("built %d clients for two identical requests, want 1", len(*built))
	}

	// ...and a different effort does not.
	for _, effort := range []string{"", "high"} {
		st := base
		st.ReasoningEffort = effort
		before := len(*built)
		if _, err := f.newEvaluator(st); err != nil {
			t.Fatalf("building an evaluator: %v", err)
		}
		if len(*built) != before+1 {
			t.Errorf("effort %q reused a client built for a different one", effort)
		}
	}
}

// The single shared *http.Client is the reason this factory is a closure and
// not a function: a client per request throws away every pooled connection,
// and an evaluation is a fan-out to one host.
func TestDefaultEvaluatorFactorySharesOneHTTPClient(t *testing.T) {
	f, built := countingFactory(t)

	for _, key := range []string{testKey, "sk-someone-else-0123456789"} {
		if _, err := f.newEvaluator(Settings{APIKey: key, BaseURL: "https://inference.example/v1", Model: "m"}); err != nil {
			t.Fatalf("building an evaluator: %v", err)
		}
	}

	if len(*built) != 2 {
		t.Fatalf("built %d clients, want 2", len(*built))
	}
	first, second := (*built)[0].HTTPClient, (*built)[1].HTTPClient
	if first == nil {
		t.Fatal("Config.HTTPClient is nil: the provider would build its own, unpooled")
	}
	if first != second {
		t.Error("each request was handed a different *http.Client; the connection pool is thrown away")
	}
	if first != f.client {
		t.Error("the client handed to the provider is not the factory's own")
	}
	if first.Timeout != 0 {
		t.Errorf("HTTPClient.Timeout = %s, want the request context to be the only deadline", first.Timeout)
	}
}

// A provider client holds the structured-output level it has negotiated. Built
// afresh per request, that negotiation is repaid on every request for ever: a
// choice with 8 options cost 23 upstream calls instead of 8.
func TestTheProviderClientIsReusedAcrossRequests(t *testing.T) {
	f, built := countingFactory(t)
	base := Settings{APIKey: testKey, BaseURL: "https://inference.example/v1", Model: "probe-1", MaxConcurrency: 8}

	for range 3 {
		if _, err := f.newEvaluator(base); err != nil {
			t.Fatalf("building an evaluator: %v", err)
		}
	}
	if len(*built) != 1 {
		t.Errorf("built %d clients for three identical requests, want 1", len(*built))
	}

	// The concurrency limit belongs to the wrapper, not the client, so it
	// must not split the cache.
	other := base
	other.MaxConcurrency = 2
	if _, err := f.newEvaluator(other); err != nil {
		t.Fatalf("building an evaluator: %v", err)
	}
	if len(*built) != 1 {
		t.Errorf("built %d clients, want the concurrency limit not to be part of the key", len(*built))
	}

	// Anything the client is actually made of does split it.
	for _, tweak := range []func(*Settings){
		func(s *Settings) { s.APIKey = "sk-someone-else-0123456789" },
		func(s *Settings) { s.BaseURL = "https://other.example/v1" },
		func(s *Settings) { s.Model = "probe-2" },
	} {
		st := base
		tweak(&st)
		before := len(*built)
		if _, err := f.newEvaluator(st); err != nil {
			t.Fatalf("building an evaluator: %v", err)
		}
		if len(*built) != before+1 {
			t.Errorf("settings %+v reused a client built for something else", st)
		}
	}
}

// The key is caller-supplied, so the cache is caller-grown. It must not be
// the memory leak that credentials-in-the-body would otherwise buy.
func TestTheClientCacheIsBounded(t *testing.T) {
	f, built := countingFactory(t)

	for i := range maxCachedClients * 3 {
		st := Settings{APIKey: fmt.Sprintf("sk-caller-%010d", i), BaseURL: "https://inference.example/v1", Model: "m"}
		if _, err := f.newEvaluator(st); err != nil {
			t.Fatalf("building an evaluator: %v", err)
		}
	}

	if got := f.cache.len(); got != maxCachedClients {
		t.Errorf("cache holds %d clients, want it bounded at %d", got, maxCachedClients)
	}
	if len(*built) != maxCachedClients*3 {
		t.Errorf("built %d clients, want one per distinct key", len(*built))
	}
}

// Eviction is least-recently-used, so a busy server's own client does not get
// pushed out by a burst of one-off callers.
func TestTheClientCacheEvictsTheLeastRecentlyUsed(t *testing.T) {
	f, built := countingFactory(t)
	mine := Settings{APIKey: testKey, BaseURL: "https://inference.example/v1", Model: "m"}

	if _, err := f.newEvaluator(mine); err != nil {
		t.Fatalf("building an evaluator: %v", err)
	}
	for i := range maxCachedClients - 1 {
		st := Settings{APIKey: fmt.Sprintf("sk-caller-%010d", i), BaseURL: "https://inference.example/v1", Model: "m"}
		if _, err := f.newEvaluator(st); err != nil {
			t.Fatalf("building an evaluator: %v", err)
		}
		// Touching the server's own client keeps it the most recent.
		if _, err := f.newEvaluator(mine); err != nil {
			t.Fatalf("building an evaluator: %v", err)
		}
	}
	built2 := len(*built)

	// One more caller evicts something, and it must not be the one in use.
	st := Settings{APIKey: "sk-caller-the-last", BaseURL: "https://inference.example/v1", Model: "m"}
	if _, err := f.newEvaluator(st); err != nil {
		t.Fatalf("building an evaluator: %v", err)
	}
	if _, err := f.newEvaluator(mine); err != nil {
		t.Fatalf("building an evaluator: %v", err)
	}
	if got := len(*built) - built2; got != 1 {
		t.Errorf("%d clients built, want only the new caller's: the server's own was evicted", got)
	}
}

// The cache key must not be the credential itself: a map keyed on plaintext
// keys is a credential store, and a heap dump is a breach.
func TestTheCacheKeyDoesNotCarryTheCredential(t *testing.T) {
	const secret = "sk-plaintext-0123456789"
	key := clientKey(Settings{APIKey: secret, BaseURL: "https://inference.example/v1", Model: "m"})

	if len(key) != 64 {
		t.Errorf("key = %q, want a hex sha-256 digest", key)
	}
	for _, part := range []string{secret, "https://inference.example/v1"} {
		if strings.Contains(key, part) {
			t.Errorf("the cache key contains %q verbatim: %q", part, key)
		}
	}
	// Distinct settings must not collide across the field boundaries.
	a := clientKey(Settings{BaseURL: "https://host/a", Model: "b", APIKey: "c"})
	b := clientKey(Settings{BaseURL: "https://host/ab", Model: "", APIKey: "c"})
	if a == b {
		t.Error("two different settings produced one cache key")
	}
}

// An evaluation fans out, and so does a burst of requests; the cache sits in
// front of both.
func TestTheClientCacheIsSafeForConcurrentUse(t *testing.T) {
	f := newEvaluatorFactory(slog.New(slog.DiscardHandler))
	var mu sync.Mutex
	var builds int
	f.newClient = func(cfg inference.Config) (*inference.Client, error) {
		mu.Lock()
		builds++
		mu.Unlock()
		return inference.New(cfg)
	}

	var wg sync.WaitGroup
	for i := range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			st := Settings{
				APIKey:  fmt.Sprintf("sk-caller-%010d", i%4),
				BaseURL: "https://inference.example/v1",
				Model:   "m",
			}
			if _, err := f.newEvaluator(st); err != nil {
				t.Errorf("building an evaluator: %v", err)
			}
		}()
	}
	wg.Wait()

	if got := f.cache.len(); got != 4 {
		t.Errorf("cache holds %d clients, want 4", got)
	}
	if builds != 4 {
		t.Errorf("built %d clients for 4 distinct settings, want 4", builds)
	}
}

func TestDefaultNewEvaluatorUsesTheFactory(t *testing.T) {
	factory := defaultNewEvaluator(slog.New(slog.DiscardHandler))
	ev, err := factory(Settings{APIKey: testKey, BaseURL: "https://inference.example/v1", Model: "m"})
	if err != nil {
		t.Fatalf("building an evaluator: %v", err)
	}
	if ev == nil {
		t.Fatal("the factory returned no evaluator")
	}
}

func TestPooledTransportRaisesTheHostLimit(t *testing.T) {
	tr, ok := pooledTransport().(*http.Transport)
	if !ok {
		t.Skip("the default transport is not an *http.Transport")
	}
	if tr.MaxIdleConnsPerHost < 8 {
		t.Errorf("MaxIdleConnsPerHost = %d, too low for a fan-out", tr.MaxIdleConnsPerHost)
	}
	if tr.IdleConnTimeout != 90*time.Second {
		t.Errorf("IdleConnTimeout = %s", tr.IdleConnTimeout)
	}
}

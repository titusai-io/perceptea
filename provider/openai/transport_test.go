package openai

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/titusai-io/perceptea/classifier"
)

func TestNewRejectsABlankKey(t *testing.T) {
	for _, key := range []string{"", "   ", "\t\n"} {
		client, err := New(Config{APIKey: key})
		if !errors.Is(err, ErrNoAPIKey) {
			t.Errorf("New(%q) error = %v, want ErrNoAPIKey", key, err)
		}
		if client != nil {
			t.Errorf("New(%q) returned a client alongside the error", key)
		}
	}
}

func TestNewAppliesDefaults(t *testing.T) {
	client, err := New(Config{APIKey: testKey})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if client.baseURL != "https://api.openai.com/v1" {
		t.Errorf("baseURL = %q, want the OpenAI default", client.baseURL)
	}
	if client.maxRetries != 2 {
		t.Errorf("maxRetries = %d, want 2", client.maxRetries)
	}
	if client.httpClient == nil {
		t.Error("httpClient is nil")
	}
	if client.log == nil {
		t.Error("logger is nil")
	}
	if client.sleep == nil {
		t.Error("sleep is nil")
	}
	if client.outputLevel() != levelJSONSchema {
		t.Errorf("level = %v, want json_schema", client.outputLevel())
	}
}

func TestNewTrimsTrailingSlashesFromTheBaseURL(t *testing.T) {
	for _, in := range []string{
		"https://openrouter.ai/api/v1",
		"https://openrouter.ai/api/v1/",
		"  https://openrouter.ai/api/v1///  ",
	} {
		client, err := New(Config{APIKey: testKey, BaseURL: in})
		if err != nil {
			t.Fatalf("New(%q): %v", in, err)
		}
		if client.baseURL != "https://openrouter.ai/api/v1" {
			t.Errorf("New(%q).baseURL = %q", in, client.baseURL)
		}
	}
}

func TestNewRejectsAnUnusableBaseURL(t *testing.T) {
	for _, in := range []string{"api.openai.com/v1", "ftp://example.com", "://nope"} {
		if _, err := New(Config{APIKey: testKey, BaseURL: in}); err == nil {
			t.Errorf("New(%q) succeeded, want an error", in)
		}
	}
}

func TestNewTreatsANegativeMaxRetriesAsNone(t *testing.T) {
	client, err := New(Config{APIKey: testKey, MaxRetries: -1})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if client.maxRetries != 0 {
		t.Errorf("maxRetries = %d, want 0", client.maxRetries)
	}
}

func TestRetriesOnRateLimitAndHonoursRetryAfterSeconds(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
		if n == 0 {
			w.Header().Set("Retry-After", "7")
			writeJSON(w, http.StatusTooManyRequests, `{"error":{"message":"slow down","type":"rate_limit_error"}}`)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.3}`))
	})
	client, sleeps := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got.Probability != 0.3 {
		t.Errorf("probability = %v, want 0.3", got.Probability)
	}
	if api.count() != 2 {
		t.Fatalf("made %d attempts, want 2", api.count())
	}
	waits := sleeps.durations()
	if len(waits) != 1 {
		t.Fatalf("backed off %d times, want 1", len(waits))
	}
	if waits[0] != 7*time.Second {
		t.Errorf("waited %v, want the 7s the Retry-After header asked for", waits[0])
	}
}

func TestRetryAfterAcceptsAnHTTPDate(t *testing.T) {
	when := time.Now().Add(5 * time.Second).UTC()
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
		if n == 0 {
			w.Header().Set("Retry-After", when.Format(http.TimeFormat))
			writeJSON(w, http.StatusTooManyRequests, `{"error":{"message":"slow down"}}`)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.3}`))
	})
	client, sleeps := newTestClient(t, api, nil)

	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}
	waits := sleeps.durations()
	if len(waits) != 1 {
		t.Fatalf("backed off %d times, want 1", len(waits))
	}
	// The header has one-second resolution and the clock moves; anything in
	// this window can only have come from the date.
	if waits[0] < 3*time.Second || waits[0] > 6*time.Second {
		t.Errorf("waited %v, want roughly 5s from the Retry-After date", waits[0])
	}
}

func TestRetryAfterIsCapped(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
		if n == 0 {
			w.Header().Set("Retry-After", "86400")
			writeJSON(w, http.StatusTooManyRequests, `{"error":{"message":"tomorrow"}}`)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.3}`))
	})
	client, sleeps := newTestClient(t, api, nil)

	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}
	if waits := sleeps.durations(); len(waits) != 1 || waits[0] != maxBackoff {
		t.Errorf("waits = %v, want a single %v", waits, maxBackoff)
	}
}

func TestRetriesOnServerErrorWithExponentialBackoff(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
		if n < 2 {
			writeJSON(w, http.StatusInternalServerError, `{"error":{"message":"boom"}}`)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.3}`))
	})
	client, sleeps := newTestClient(t, api, nil)

	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}
	if api.count() != 3 {
		t.Fatalf("made %d attempts, want 3", api.count())
	}
	waits := sleeps.durations()
	if len(waits) != 2 {
		t.Fatalf("backed off %d times, want 2", len(waits))
	}
	// Equal jitter: each wait lies in [d/2, d] for d = 500ms << attempt.
	for i, w := range waits {
		full := baseBackoff << i
		if w < full/2 || w > full {
			t.Errorf("wait %d = %v, want within [%v, %v]", i, w, full/2, full)
		}
	}
	if waits[1] <= waits[0]/2 {
		t.Errorf("backoff did not grow: %v then %v", waits[0], waits[1])
	}
}

func TestMaxRetriesIsRespected(t *testing.T) {
	for _, tc := range []struct{ retries, attempts int }{
		{-1, 1},
		{1, 2},
		{4, 5},
	} {
		api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
			writeJSON(w, http.StatusServiceUnavailable, `{"error":{"message":"down"}}`)
		})
		client, sleeps := newTestClient(t, api, func(cfg *Config) { cfg.MaxRetries = tc.retries })

		_, err := client.Score(context.Background(), fixtureRequest)
		if err == nil {
			t.Fatalf("MaxRetries %d: want an error", tc.retries)
		}
		if api.count() != tc.attempts {
			t.Errorf("MaxRetries %d: made %d attempts, want %d", tc.retries, api.count(), tc.attempts)
		}
		if got := len(sleeps.durations()); got != tc.attempts-1 {
			t.Errorf("MaxRetries %d: backed off %d times, want %d", tc.retries, got, tc.attempts-1)
		}
		var apiErr *APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusServiceUnavailable {
			t.Errorf("MaxRetries %d: error = %v, want a 503 APIError", tc.retries, err)
		}
	}
}

// flakyTransport fails the first failures round trips at the transport layer,
// as a dial error would, and then delegates to the real one.
type flakyTransport struct {
	failures int32
	attempts atomic.Int32
	next     http.RoundTripper
}

func (f *flakyTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if f.attempts.Add(1) <= f.failures {
		return nil, errors.New("dial tcp 203.0.113.1:443: connect: connection refused")
	}
	return f.next.RoundTrip(r)
}

func TestRetriesOnATransportFailure(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.77}`))
	transport := &flakyTransport{failures: 2, next: http.DefaultTransport}
	client, sleeps := newTestClient(t, api, func(cfg *Config) {
		cfg.HTTPClient = &http.Client{Transport: transport}
	})

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got.Probability != 0.77 {
		t.Errorf("probability = %v, want 0.77", got.Probability)
	}
	if n := transport.attempts.Load(); n != 3 {
		t.Errorf("made %d attempts, want 3", n)
	}
	if api.count() != 1 {
		t.Errorf("the server saw %d requests, want 1", api.count())
	}
	if n := len(sleeps.durations()); n != 2 {
		t.Errorf("backed off %d times, want 2", n)
	}
}

func TestATransportFailureThatNeverClearsIsReturned(t *testing.T) {
	// A server that is closed before the first request: every dial fails.
	dead := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := dead.URL
	dead.Close()

	client, err := New(Config{
		APIKey:  testKey,
		BaseURL: url,
		Model:   "m",
		Logger:  slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	sleeps := &sleepLog{}
	client.sleep = sleeps.record

	if _, err := client.Score(context.Background(), fixtureRequest); err == nil {
		t.Fatal("want an error when nothing is listening")
	} else {
		var apiErr *APIError
		if errors.As(err, &apiErr) {
			t.Errorf("error = %v, want a transport error rather than an APIError", err)
		}
	}
	if n := len(sleeps.durations()); n != 2 {
		t.Errorf("backed off %d times, want 2", n)
	}
}

func TestDoesNotRetryABadRequest(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		writeJSON(w, http.StatusBadRequest, `{"error":{"message":"bad model","type":"invalid_request_error","code":"model_not_found"}}`)
	})
	client, sleeps := newTestClient(t, api, func(cfg *Config) { cfg.MaxRetries = 5 })

	_, err := client.Score(context.Background(), fixtureRequest)
	if err == nil {
		t.Fatal("want an error")
	}
	// One attempt per structured-output level, and no retry within a level.
	if api.count() != 3 {
		t.Fatalf("made %d requests, want 3 (one per level, none retried)", api.count())
	}
	if n := len(sleeps.durations()); n != 0 {
		t.Errorf("backed off %d times, want 0: a 400 is final", n)
	}
}

func TestAPIErrorCarriesTheStatusAndDetail(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		writeJSON(w, http.StatusUnauthorized,
			`{"error":{"message":"Incorrect API key provided","type":"invalid_request_error","code":"invalid_api_key"}}`)
	})
	client, _ := newTestClient(t, api, nil)

	_, err := client.Score(context.Background(), fixtureRequest)
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v (%T), want an *APIError", err, err)
	}
	if apiErr.StatusCode != http.StatusUnauthorized {
		t.Errorf("StatusCode = %d, want 401", apiErr.StatusCode)
	}
	if apiErr.Message != "Incorrect API key provided" {
		t.Errorf("Message = %q", apiErr.Message)
	}
	if apiErr.Type != "invalid_request_error" {
		t.Errorf("Type = %q", apiErr.Type)
	}
	if apiErr.Code != "invalid_api_key" {
		t.Errorf("Code = %q", apiErr.Code)
	}
	if !strings.Contains(apiErr.Body, "Incorrect API key provided") {
		t.Errorf("Body = %q, want the raw document", apiErr.Body)
	}
	for _, want := range []string{"401", "Incorrect API key provided", "invalid_api_key"} {
		if !strings.Contains(apiErr.Error(), want) {
			t.Errorf("Error() = %q, want it to mention %q", apiErr.Error(), want)
		}
	}
}

func TestAPIErrorFromANonJSONBody(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, "<html><body>502 Bad Gateway</body></html>")
	})
	client, _ := newTestClient(t, api, nil)

	_, err := client.Score(context.Background(), fixtureRequest)
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v, want an *APIError", err)
	}
	if !strings.Contains(apiErr.Message, "502 Bad Gateway") {
		t.Errorf("Message = %q, want the raw body", apiErr.Message)
	}
}

func TestAPIErrorFromABareStringBody(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		writeJSON(w, http.StatusBadRequest, `"upstream refused the request"`)
	})
	client, _ := newTestClient(t, api, nil)

	_, err := client.Score(context.Background(), fixtureRequest)
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v, want an *APIError", err)
	}
	if apiErr.Message != "upstream refused the request" {
		t.Errorf("Message = %q", apiErr.Message)
	}
}

func TestAPIErrorAcceptsANumericCode(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		writeJSON(w, http.StatusBadRequest, `{"error":{"message":"nope","code":402}}`)
	})
	client, _ := newTestClient(t, api, nil)

	_, err := client.Score(context.Background(), fixtureRequest)
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v, want an *APIError", err)
	}
	if apiErr.Code != "402" {
		t.Errorf("Code = %q, want 402", apiErr.Code)
	}
}

func TestAPIErrorBodyIsTruncated(t *testing.T) {
	huge := strings.Repeat("x", 100_000)
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, huge)
	})
	client, _ := newTestClient(t, api, nil)

	_, err := client.Score(context.Background(), fixtureRequest)
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v, want an *APIError", err)
	}
	if len(apiErr.Body) > maxErrorBody+64 {
		t.Errorf("len(Body) = %d, want it truncated near %d", len(apiErr.Body), maxErrorBody)
	}
	if len(apiErr.Error()) > 4096 {
		t.Errorf("len(Error()) = %d, want a bounded message", len(apiErr.Error()))
	}
}

func TestContextCancellationMidFlight(t *testing.T) {
	release := make(chan struct{})
	api := newFakeAPI(t, func(w http.ResponseWriter, r *http.Request, _ int) {
		<-release
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.5}`))
	})
	t.Cleanup(func() { close(release) })

	client, sleeps := newTestClient(t, api, nil)
	ctx, cancel := context.WithCancel(context.Background())

	started := make(chan struct{})
	go func() {
		<-started
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()
	close(started)

	_, err := client.Score(ctx, fixtureRequest)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if n := len(sleeps.durations()); n != 0 {
		t.Errorf("backed off %d times, want 0: a cancelled context is final", n)
	}
	if api.count() != 1 {
		t.Errorf("made %d attempts, want 1", api.count())
	}
}

// TestAlreadyCancelledContextMakesNoCall covers complete's check at the top of
// the attempt loop. The request never reaching the server does not prove the
// check ran: http.Client.Do refuses an already-cancelled request itself, and
// the call would fail either way. What distinguishes them is the error. The
// check returns ctx.Err() as it stands, the sentinel itself; the transport
// returns a *url.Error wrapping it, which the client then wraps again in
// "openai: chat completion". Both satisfy errors.Is, so this asserts on
// identity instead.
func TestAlreadyCancelledContextMakesNoCall(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
	client, _ := newTestClient(t, api, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := client.Score(ctx, fixtureRequest)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if err != context.Canceled {
		t.Errorf("error = %v (%T), want the bare sentinel: an already-cancelled call reached the transport", err, err)
	}
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		t.Errorf("error = %v, want no transport error: the request should never have been built", err)
	}
	if api.count() != 0 {
		t.Errorf("made %d calls, want none", api.count())
	}
}

func TestCancellationDuringBackoffStops(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		writeJSON(w, http.StatusTooManyRequests, `{"error":{"message":"later"}}`)
	})
	client, _ := newTestClient(t, api, func(cfg *Config) { cfg.MaxRetries = 5 })

	ctx, cancel := context.WithCancel(context.Background())
	client.sleep = func(context.Context, time.Duration) error {
		cancel()
		return context.Canceled
	}

	_, err := client.Score(ctx, fixtureRequest)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Error("the abandoned attempt's APIError should still be reachable")
	}
	if api.count() != 1 {
		t.Errorf("made %d attempts, want 1", api.count())
	}
}

func TestMalformedSuccessBodyIsNotRetried(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		writeJSON(w, http.StatusOK, `{"choices": [ truncated`)
	})
	client, sleeps := newTestClient(t, api, func(cfg *Config) { cfg.MaxRetries = 3 })

	if _, err := client.Score(context.Background(), fixtureRequest); err == nil {
		t.Fatal("want an error for an undecodable body")
	}
	if api.count() != 1 {
		t.Errorf("made %d attempts, want 1", api.count())
	}
	if n := len(sleeps.durations()); n != 0 {
		t.Errorf("backed off %d times, want 0", n)
	}
}

func TestTheAPIKeyNeverLeaks(t *testing.T) {
	// A provider that helpfully echoes the credential back at us, in the
	// body, the message and a header.
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		writeJSON(w, http.StatusUnauthorized,
			`{"error":{"message":"key `+testKey+` is not valid","type":"auth_`+testKey+`","code":"`+testKey+`"}}`)
	})

	sink := &logSink{}
	client, _ := newTestClient(t, api, func(cfg *Config) {
		cfg.Logger = slog.New(sink.handler())
	})

	_, err := client.Score(context.Background(), fixtureRequest)
	if err == nil {
		t.Fatal("want an error")
	}
	if strings.Contains(err.Error(), testKey) {
		t.Errorf("the API key appears in the error: %q", err.Error())
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v, want an *APIError", err)
	}
	for name, field := range map[string]string{
		"Body":    apiErr.Body,
		"Message": apiErr.Message,
		"Type":    apiErr.Type,
		"Code":    apiErr.Code,
	} {
		if strings.Contains(field, testKey) {
			t.Errorf("the API key appears in APIError.%s: %q", name, field)
		}
	}

	// And a successful call, whose debug records are the noisiest thing the
	// client writes.
	ok := alwaysJSON(t, scoreBody(`{"p":0.5}`))
	quiet, _ := newTestClient(t, ok, func(cfg *Config) { cfg.Logger = slog.New(sink.handler()) })
	if _, err := quiet.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}

	logged := sink.String()
	if logged == "" {
		t.Fatal("nothing was logged at debug level; the leak assertion would be vacuous")
	}
	if strings.Contains(logged, testKey) {
		t.Errorf("the API key appears in the log output:\n%s", logged)
	}
	if strings.Contains(logged, "Bearer") {
		t.Errorf("an Authorization header appears in the log output:\n%s", logged)
	}
}

func TestRetryableClassification(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{"request timeout", &APIError{StatusCode: 408}, true},
		{"rate limit", &APIError{StatusCode: 429}, true},
		{"server error", &APIError{StatusCode: 500}, true},
		{"gateway timeout", &APIError{StatusCode: 504}, true},
		{"bad request", &APIError{StatusCode: 400}, false},
		{"unauthorized", &APIError{StatusCode: 401}, false},
		{"not found", &APIError{StatusCode: 404}, false},
		{"unprocessable", &APIError{StatusCode: 422}, false},
		{"not implemented", &APIError{StatusCode: 501}, false},
		{"cancelled", context.Canceled, false},
		{"deadline", context.DeadlineExceeded, false},
		{"decode failure", &nonRetryable{errors.New("bad json")}, false},
		{"transport", errors.New("connection reset by peer"), true},
	} {
		if got := retryable(tc.err); got != tc.want {
			t.Errorf("retryable(%s) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestUnsupportedShapeClassification(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		// The only three statuses that say anything about the request's shape.
		{"bad request", &APIError{StatusCode: 400}, true},
		{"unprocessable", &APIError{StatusCode: 422}, true},
		{"not implemented", &APIError{StatusCode: 501}, true},
		// A wrong base URL or an unknown model, not a rejected field.
		{"not found", &APIError{StatusCode: 404}, false},
		{"method not allowed", &APIError{StatusCode: 405}, false},
		{"conflict", &APIError{StatusCode: 409}, false},
		{"payload too large", &APIError{StatusCode: 413}, false},
		{"unauthorized", &APIError{StatusCode: 401}, false},
		{"forbidden", &APIError{StatusCode: 403}, false},
		{"request timeout", &APIError{StatusCode: 408}, false},
		{"rate limit", &APIError{StatusCode: 429}, false},
		{"server error", &APIError{StatusCode: 500}, false},
		{"bad gateway", &APIError{StatusCode: 502}, false},
		{"transport", errors.New("connection reset by peer"), false},
	} {
		if got := unsupportedShape(tc.err); got != tc.want {
			t.Errorf("unsupportedShape(%s) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestClientSatisfiesTheClassifierContract(t *testing.T) {
	client, err := New(Config{APIKey: testKey})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var (
		scorer    classifier.Scorer    = client
		generator classifier.Generator = client
	)
	if scorer == nil || generator == nil {
		t.Fatal("unreachable")
	}
}

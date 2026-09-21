package api

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
	"github.com/titusai-io/perceptea/provider/inference"
)

const validBody = `{"state":"s","questions":{"q":{"type":"noul"}}}`

func TestEvaluateRejectsBadBodies(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		wantStatus int
		wantCode   string
	}{
		{"not JSON at all", `not json`, http.StatusBadRequest, codeInvalidJSON},
		{"truncated JSON", `{"state":"s",`, http.StatusBadRequest, codeInvalidJSON},
		{"empty body", ``, http.StatusBadRequest, codeInvalidJSON},
		{"a JSON array", `[]`, http.StatusBadRequest, codeInvalidJSON},
		{
			"a question with an unknown type",
			`{"state":"s","questions":{"q":{"type":"vibes"}}}`,
			http.StatusBadRequest, codeInvalidJSON,
		},
		{
			"a question with no type",
			`{"state":"s","questions":{"q":{"instructions":"?"}}}`,
			http.StatusBadRequest, codeInvalidJSON,
		},
		{
			"choice criteria of the wrong shape",
			`{"state":"s","questions":{"q":{"type":"choice","criteria":["a","b"]}}}`,
			http.StatusBadRequest, codeInvalidJSON,
		},
		{
			"score criteria of the wrong shape",
			`{"state":"s","questions":{"q":{"type":"score","criteria":{"a":"b"}}}}`,
			http.StatusBadRequest, codeInvalidJSON,
		},
		{
			"questions that are not an object",
			`{"state":"s","questions":"lots"}`,
			http.StatusBadRequest, codeInvalidJSON,
		},
		{"no state", `{"questions":{"q":{"type":"noul"}}}`, http.StatusBadRequest, codeInvalidRequest},
		{"a null state", `{"state":null,"questions":{"q":{"type":"noul"}}}`, http.StatusBadRequest, codeInvalidRequest},
		{"no questions", `{"state":"s"}`, http.StatusBadRequest, codeInvalidRequest},
		{"null questions", `{"state":"s","questions":null}`, http.StatusBadRequest, codeInvalidRequest},
		{"empty questions", `{"state":"s","questions":{}}`, http.StatusBadRequest, codeInvalidRequest},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newHarness(t, nil)
			h.expectError(h.post(tt.body), tt.wantStatus, tt.wantCode)
			if len(h.seenRequests) != 0 {
				t.Error("a rejected request still reached the evaluator")
			}
		})
	}
}

func TestEvaluateAcceptsAnyJSONState(t *testing.T) {
	for _, state := range []string{`"text"`, `{"order":{"id":7}}`, `[1,2,3]`, `0`, `false`} {
		t.Run(state, func(t *testing.T) {
			h := newHarness(t, nil)
			body := fmt.Sprintf(`{"state":%s,"questions":{"q":{"type":"noul"}}}`, state)
			if w := h.post(body); w.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
			}
			if got := string(h.request().State.Raw()); got != state {
				t.Errorf("state = %s, want %s", got, state)
			}
		})
	}
}

func TestEvaluateErrorTable(t *testing.T) {
	tests := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
		// wantMessage, when set, must appear in the message.
		wantMessage string
	}{
		{
			name:       "no questions",
			err:        classifier.ErrNoQuestions,
			wantStatus: http.StatusBadRequest,
			wantCode:   codeInvalidRequest,
		},
		{
			name:        "a validation error",
			err:         &classifier.ValidationError{Question: "urgency", Field: "criteria", Message: "needs at least 2 levels"},
			wantStatus:  http.StatusBadRequest,
			wantCode:    codeInvalidRequest,
			wantMessage: "needs at least 2 levels",
		},
		{
			name:       "a wrapped validation error",
			err:        fmt.Errorf("evaluating: %w", &classifier.ValidationError{Question: "q", Field: "type", Message: "unknown"}),
			wantStatus: http.StatusBadRequest,
			wantCode:   codeInvalidRequest,
		},
		{
			name:        "oneshot is unsupported",
			err:         fmt.Errorf("%w (%T)", classifier.ErrOneshotUnsupported, struct{}{}),
			wantStatus:  http.StatusBadRequest,
			wantCode:    codeUnsupportedMode,
			wantMessage: "oneshot",
		},
		{
			name:       "an unknown mode",
			err:        fmt.Errorf("%w %q", classifier.ErrUnknownMode, "sideways"),
			wantStatus: http.StatusBadRequest,
			wantCode:   codeUnsupportedMode,
		},
		{
			name:       "no API key",
			err:        inference.ErrNoAPIKey,
			wantStatus: http.StatusUnauthorized,
			wantCode:   codeMissingAPIKey,
		},
		{
			name:        "the upstream rate limited us",
			err:         &inference.APIError{StatusCode: http.StatusTooManyRequests, Type: "rate_limit", Message: "slow down"},
			wantStatus:  http.StatusTooManyRequests,
			wantCode:    codeUpstreamRateLimited,
			wantMessage: "slow down",
		},
		{
			name:        "the upstream failed",
			err:         &inference.APIError{StatusCode: http.StatusInternalServerError, Type: "server_error", Message: "boom"},
			wantStatus:  http.StatusBadGateway,
			wantCode:    codeUpstreamError,
			wantMessage: "boom",
		},
		{
			name:       "the upstream refused our credentials",
			err:        &inference.APIError{StatusCode: http.StatusUnauthorized, Type: "invalid_api_key", Message: "bad key"},
			wantStatus: http.StatusBadGateway,
			wantCode:   codeUpstreamError,
		},
		{
			name: "a transport failure",
			err: fmt.Errorf("inference: chat completion: %w", &url.Error{
				Op:  "Post",
				URL: "https://inference.example/v1/chat/completions",
				Err: errors.New("dial tcp: connection refused"),
			}),
			wantStatus:  http.StatusBadGateway,
			wantCode:    codeUpstreamError,
			wantMessage: "could not be reached",
		},
		{
			name:        "the deadline expired",
			err:         fmt.Errorf("scoring: %w", context.DeadlineExceeded),
			wantStatus:  http.StatusGatewayTimeout,
			wantCode:    codeTimeout,
			wantMessage: "deadline",
		},
		{
			name:       "no scorer is a bug, not a bad request",
			err:        classifier.ErrNoScorer,
			wantStatus: http.StatusInternalServerError,
			wantCode:   codeInternal,
		},
		{
			name:        "anything else",
			err:         errors.New("something came loose"),
			wantStatus:  http.StatusInternalServerError,
			wantCode:    codeInternal,
			wantMessage: "internal error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newHarness(t, nil)
			h.failWith(tt.err)

			message := h.expectError(h.post(validBody), tt.wantStatus, tt.wantCode)
			if tt.wantMessage != "" && !strings.Contains(message, tt.wantMessage) {
				t.Errorf("message = %q, want it to mention %q", message, tt.wantMessage)
			}
		})
	}
}

// An upstream that says "429, come back in 40 seconds" and a deadline that
// expires waiting for that to pass arrive as one error wrapping both. The
// rate limit is the half worth reporting: it is the only one that tells the
// caller to back off rather than retry now.
func TestARateLimitBeatsTheDeadlineItRanInto(t *testing.T) {
	h := newHarness(t, nil)
	h.failWith(fmt.Errorf("inference: retry abandoned: %w (last attempt: %w)",
		context.DeadlineExceeded,
		&inference.APIError{StatusCode: http.StatusTooManyRequests, Type: "rate_limit", Message: "slow down"}))

	message := h.expectError(h.post(validBody), http.StatusTooManyRequests, codeUpstreamRateLimited)
	if !strings.Contains(message, "slow down") {
		t.Errorf("message = %q, want the provider's own text", message)
	}
}

// The same wrapping with any other upstream status is still the upstream's
// failure, not ours.
func TestAnUpstreamFailureBeatsTheDeadlineItRanInto(t *testing.T) {
	h := newHarness(t, nil)
	h.failWith(fmt.Errorf("inference: retry abandoned: %w (last attempt: %w)",
		context.DeadlineExceeded,
		&inference.APIError{StatusCode: http.StatusBadGateway, Message: "upstream unavailable"}))

	h.expectError(h.post(validBody), http.StatusBadGateway, codeUpstreamError)
}

// brokenReader hands over a prefix and then fails, the way a connection does
// when it dies part-way through a body.
type brokenReader struct {
	prefix string
	err    error
}

func (b *brokenReader) Read(p []byte) (int, error) {
	if b.prefix != "" {
		n := copy(p, b.prefix)
		b.prefix = b.prefix[n:]
		return n, nil
	}
	return 0, b.err
}

// socketTimeout is what http.Server.ReadTimeout produces: a net.Error that
// reports a timeout, and whose text names the address this process is bound
// to.
type socketTimeout struct{}

func (socketTimeout) Error() string   { return "read tcp 10.0.3.17:8080->203.0.113.9:54321: i/o timeout" }
func (socketTimeout) Timeout() bool   { return true }
func (socketTimeout) Temporary() bool { return true }

func TestABodyReadThatTimesOutIs504AndNamesNoAddress(t *testing.T) {
	h := newHarness(t, nil)

	w := h.postReader(&brokenReader{prefix: `{"state":"s",`, err: socketTimeout{}})

	message := h.expectError(w, http.StatusGatewayTimeout, codeTimeout)
	for _, secret := range []string{"10.0.3.17", "8080", "read tcp"} {
		if strings.Contains(message, secret) {
			t.Errorf("the caller was told %q: %s", secret, message)
		}
	}
	line := findLog(t, h.logs, "request failed")
	if detail, _ := line["error"].(string); !strings.Contains(detail, "10.0.3.17") {
		t.Errorf("the socket error was not logged: %v", line)
	}
}

func TestABodyReadThatIsCutShortIs499WithNoBody(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
	}{
		{"the client hung up", io.ErrUnexpectedEOF},
		{"the connection was reset", &net.OpError{
			Op: "read", Net: "tcp",
			Source: &net.TCPAddr{IP: net.IPv4(10, 0, 3, 17), Port: 8080},
			Err:    syscall.ECONNRESET,
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, nil)

			w := h.postReader(&brokenReader{prefix: `{"state":"s",`, err: tc.err})

			if w.Code != StatusClientClosedRequest {
				t.Errorf("status = %d, want %d (body: %s)", w.Code, StatusClientClosedRequest, w.Body.String())
			}
			if body := w.Body.String(); body != "" {
				t.Errorf("body = %q, want nothing sent to a caller that has gone", body)
			}
			line := findLog(t, h.logs, "request")
			if detail, _ := line["error"].(string); !strings.Contains(detail, "reading the request body") {
				t.Errorf("the read failure was not logged: %v", line)
			}
		})
	}
}

// simultaneousReader hands over bytes and an error in one Read, which a
// socket is allowed to do. encoding/json scans what arrived before it acts on
// the error, so both a syntax error and a read failure are in play at once.
type simultaneousReader struct {
	data string
	err  error
	done bool
}

func (r *simultaneousReader) Read(p []byte) (int, error) {
	if r.done {
		return 0, r.err
	}
	r.done = true
	return copy(p, r.data), r.err
}

// When the document is broken *and* the connection is, the document wins: the
// bytes that arrived are enough to say what is wrong with them, and saying so
// gives away nothing about the socket.
func TestABrokenDocumentOnABrokenConnectionIs400(t *testing.T) {
	h := newHarness(t, nil)

	w := h.postReader(&simultaneousReader{
		data: `{"state"::}`,
		err:  &net.OpError{Op: "read", Net: "tcp", Err: syscall.ECONNRESET},
	})

	message := h.expectError(w, http.StatusBadRequest, codeInvalidJSON)
	if strings.Contains(message, "read tcp") || strings.Contains(message, "reset") {
		t.Errorf("the socket error reached the caller: %q", message)
	}
}

// A body that arrived whole and is simply not valid JSON is the caller's own
// document, and telling them what is wrong with it costs nothing.
func TestATruncatedDocumentIsStill400WithItsReason(t *testing.T) {
	h := newHarness(t, nil)

	message := h.expectError(h.post(`{"state":"s",`), http.StatusBadRequest, codeInvalidJSON)
	if !strings.Contains(message, "could not decode") {
		t.Errorf("message = %q, want the JSON problem", message)
	}
}

func TestEvaluateRejectsUnknownFields(t *testing.T) {
	// The case that matters: a caller who misspells api_key has it ignored
	// and the server's own credential spent on their behalf, with nothing on
	// the wire to say so.
	for _, field := range []string{"apikey", "api-key", "apiKey", "base-url", "mode_"} {
		t.Run(field, func(t *testing.T) {
			h := newHarness(t, nil)
			body := fmt.Sprintf(`{"state":"s","questions":{"q":{"type":"noul"}},%q:"sk-theirs-0123456789"}`, field)

			message := h.expectError(h.post(body), http.StatusBadRequest, codeInvalidJSON)
			if !strings.Contains(message, field) {
				t.Errorf("message = %q, want it to name the field that was not understood", message)
			}
			if len(h.seenSettings) != 0 {
				t.Error("the request reached the evaluator on the server's key")
			}
		})
	}
}

// encoding/json matches a field name case-insensitively, so a caller who
// types Api_Key is understood rather than ignored. Pinned because it decides
// whether the rejection above is the whole of the protection.
func TestAMiscasedFieldNameIsStillUnderstood(t *testing.T) {
	const theirKey = "sk-from-the-body-9876543210"
	h := newHarness(t, nil)

	body := fmt.Sprintf(`{"state":"s","questions":{"q":{"type":"noul"}},"Api_Key":%q}`, theirKey)
	if w := h.post(body); w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}
	if got := h.settings().APIKey; got != theirKey {
		t.Errorf("Settings.APIKey = %q, want the caller's own key, not the server's", got)
	}
}

func TestEvaluateRejectsTrailingContent(t *testing.T) {
	for _, body := range []string{
		validBody + `{"state":"again"}`,
		validBody + `garbage`,
		validBody + `]`,
	} {
		t.Run(body[len(validBody):], func(t *testing.T) {
			h := newHarness(t, nil)
			h.expectError(h.post(body), http.StatusBadRequest, codeInvalidJSON)
			if len(h.seenSettings) != 0 {
				t.Error("a body with two documents in it was evaluated anyway")
			}
		})
	}
}

// Trailing whitespace is not trailing content.
func TestEvaluateAcceptsTrailingWhitespace(t *testing.T) {
	h := newHarness(t, nil)
	if w := h.post(validBody + "\n\n  \t\n"); w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}
}

func TestEvaluateFactoryErrorsAreClassifiedToo(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.APIKey = "" })
	h.factoryErr = inference.ErrNoAPIKey

	message := h.expectError(h.post(validBody), http.StatusUnauthorized, codeMissingAPIKey)
	for _, want := range []string{config.EnvAPIKey, "api_key"} {
		if !strings.Contains(message, want) {
			t.Errorf("message = %q, want it to mention %q", message, want)
		}
	}
	// Exactly one environment variable, because exactly one is read. A
	// message that offered a second one would be sending the operator to set
	// a variable this service never looks at.
	if n := strings.Count(message, "_API_KEY"); n != 1 {
		t.Errorf("message = %q, want it to name one key variable, named %d", message, n)
	}
	if len(h.seenRequests) != 0 {
		t.Error("the server evaluated a request it had no key for")
	}
}

func TestMissingKeyMessageOmitsTheBodyRouteWhenDisallowed(t *testing.T) {
	h := newHarness(t, func(c *config.Config) {
		c.APIKey = ""
		c.AllowRequestCredentials = false
	})
	h.factoryErr = inference.ErrNoAPIKey

	message := h.expectError(h.post(validBody), http.StatusUnauthorized, codeMissingAPIKey)
	if strings.Contains(message, "api_key") {
		t.Errorf("message = %q, want no mention of a body key when the server does not accept one", message)
	}
}

func TestEvaluateInternalErrorsAreLoggedNotReturned(t *testing.T) {
	h := newHarness(t, nil)
	h.failWith(errors.New("the internal detail"))

	message := h.expectError(h.post(validBody), http.StatusInternalServerError, codeInternal)
	if strings.Contains(message, "the internal detail") {
		t.Errorf("the response body leaked the internal detail: %q", message)
	}
	line := findLog(t, h.logs, "request failed")
	if got, _ := line["error"].(string); !strings.Contains(got, "the internal detail") {
		t.Errorf("the 5xx log line does not carry the detail: %v", line)
	}
	if line["status"] != float64(http.StatusInternalServerError) {
		t.Errorf("status = %v, want 500", line["status"])
	}
}

func TestEvaluateTimesOut(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.RequestTimeout = 25 * time.Millisecond })
	h.evaluate = func(ctx context.Context, _ classifier.Request) (classifier.Response, error) {
		select {
		case <-ctx.Done():
			return classifier.Response{}, fmt.Errorf("scoring: %w", ctx.Err())
		case <-time.After(5 * time.Second):
			// Only reachable if the handler forgot to impose a deadline.
			return classifier.Response{}, errors.New("no deadline was imposed on the request")
		}
	}

	start := time.Now()
	message := h.expectError(h.post(validBody), http.StatusGatewayTimeout, codeTimeout)
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("the request took %s; the deadline was not enforced", elapsed)
	}
	if !strings.Contains(message, "25ms") {
		t.Errorf("message = %q, want it to name the deadline", message)
	}
}

func TestEvaluateRequestCarriesADeadline(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.RequestTimeout = 3 * time.Second })
	var deadline time.Time
	var ok bool
	h.evaluate = func(ctx context.Context, _ classifier.Request) (classifier.Response, error) {
		deadline, ok = ctx.Deadline()
		return classifier.Response{}, nil
	}

	if w := h.post(validBody); w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if !ok {
		t.Fatal("the evaluator was handed a context with no deadline")
	}
	if remaining := time.Until(deadline); remaining <= 0 || remaining > 3*time.Second {
		t.Errorf("deadline is %s away, want it within the configured 3s", remaining)
	}
}

// The evaluator here answers from the context it was handed, so the test
// fails if the handler passes one that does not carry the client's departure:
// a fake that returns context.Canceled unconditionally would pass against a
// handler that ignored cancellation entirely.
func TestEvaluateClientHangUpIs499WithNoBody(t *testing.T) {
	h := newHarness(t, nil)
	h.evaluate = func(ctx context.Context, _ classifier.Request) (classifier.Response, error) {
		select {
		case <-ctx.Done():
			return classifier.Response{}, fmt.Errorf("scoring: %w", ctx.Err())
		default:
			// Only reachable if the client's cancellation never arrived.
			return sampleResponse(), nil
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req := httptest.NewRequest(http.MethodPost, "/api/evaluate", strings.NewReader(validBody)).WithContext(ctx)
	w := h.send(req)

	if w.Code != StatusClientClosedRequest {
		t.Errorf("status = %d, want %d", w.Code, StatusClientClosedRequest)
	}
	if body := w.Body.String(); body != "" {
		t.Errorf("body = %q, want nothing sent to a caller that has gone", body)
	}
}

func TestEvaluateRejectsAnOversizedBody(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.MaxBodyBytes = 256 })

	padding := strings.Repeat("x", 512)
	body := fmt.Sprintf(`{"state":%q,"questions":{"q":{"type":"noul"}}}`, padding)
	message := h.expectError(h.post(body), http.StatusRequestEntityTooLarge, codePayloadTooLarge)
	if !strings.Contains(message, "256") {
		t.Errorf("message = %q, want it to name the limit", message)
	}
	if len(h.seenRequests) != 0 {
		t.Error("an oversized body still reached the evaluator")
	}

	// A body just under the limit still goes through.
	small := newHarness(t, func(c *config.Config) { c.MaxBodyBytes = 256 })
	if w := small.post(validBody); w.Code != http.StatusOK {
		t.Errorf("a small body was rejected: %d %s", w.Code, w.Body.String())
	}
}

func TestEvaluateHonoursRequestCredentialsOnlyWhenAllowed(t *testing.T) {
	const bodyKey = "sk-from-the-body-9876543210"
	body := fmt.Sprintf(`{"state":"s","questions":{"q":{"type":"noul"}},"api_key":%q,"inference_base_url":"https://body.example/v1"}`, bodyKey)

	t.Run("allowed", func(t *testing.T) {
		h := newHarness(t, func(c *config.Config) { c.AllowRequestCredentials = true })
		if w := h.post(body); w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
		}
		st := h.settings()
		if st.APIKey != bodyKey {
			t.Errorf("Settings.APIKey = %q, want the body's key", st.APIKey)
		}
		if st.BaseURL != "https://body.example/v1" {
			t.Errorf("Settings.BaseURL = %q, want the body's base URL", st.BaseURL)
		}
	})

	t.Run("not allowed", func(t *testing.T) {
		h := newHarness(t, func(c *config.Config) { c.AllowRequestCredentials = false })
		w := h.post(body)
		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want the request to be served, not refused (body: %s)", w.Code, w.Body.String())
		}
		st := h.settings()
		if st.APIKey != testKey {
			t.Errorf("Settings.APIKey = %q, want the configured key", st.APIKey)
		}
		if st.BaseURL != "https://inference.example/v1" {
			t.Errorf("Settings.BaseURL = %q, want the configured base URL", st.BaseURL)
		}
		if !strings.Contains(h.logs.String(), "ignoring request-supplied credentials") {
			t.Errorf("the ignored credentials were not logged at debug:\n%s", h.logs.String())
		}
		if strings.Contains(h.logs.String(), bodyKey) {
			t.Error("the ignored key was written to the log")
		}
	})

	t.Run("blank body credentials never win", func(t *testing.T) {
		h := newHarness(t, nil)
		blank := `{"state":"s","questions":{"q":{"type":"noul"}},"api_key":"   ","inference_base_url":""}`
		if w := h.post(blank); w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		if st := h.settings(); st.APIKey != testKey {
			t.Errorf("Settings.APIKey = %q, want the configured key", st.APIKey)
		}
	})
}

func TestScrub(t *testing.T) {
	if got := scrub("key sk-0123456789 leaked", "sk-0123456789"); strings.Contains(got, "sk-0123456789") {
		t.Errorf("scrub left the secret in %q", got)
	}
	if got := scrub("nothing to do", ""); got != "nothing to do" {
		t.Errorf("scrub(%q, \"\") = %q", "nothing to do", got)
	}
	// A short string is not a credential and must not be mangled.
	if got := scrub("a rate of 12 per second", "12"); got != "a rate of 12 per second" {
		t.Errorf("scrub mangled ordinary text: %q", got)
	}
}

// Both sides of the threshold, because both are load-bearing: a local model
// server conventionally takes a four-character key, and anything shorter
// would redact ordinary words out of every message.
func TestScrubThreshold(t *testing.T) {
	for _, secret := range []string{"ollama", "EMPTY", "x-ai", "1234"} {
		if got := scrub("rejected key "+secret+" at the gateway", secret); strings.Contains(got, secret) {
			t.Errorf("a %d character key survived: %q", len(secret), got)
		}
	}
	for _, short := range []string{"x", "of", "the"} {
		text := "one of the things"
		if got := scrub(text, short); got != text {
			t.Errorf("scrub(%q, %q) = %q, want ordinary text left alone", text, short, got)
		}
	}
}

// A transport error carries the URL it failed to reach, and net/http masks
// only the password in it: a key in the query rides along into the 5xx log
// line, which is where an operator's gateway credential would have ended up.
func TestABaseURLCredentialIsScrubbedFromTheLog(t *testing.T) {
	const gateway = "https://svc:sk-secret-0123456789@gw.example/v1?key=abc123def"

	t.Run("the server's own", func(t *testing.T) {
		h := newHarness(t, func(c *config.Config) { c.BaseURL = gateway })
		h.failWith(errors.New(`inference: chat completion: Post "https://svc:***@gw.example/v1?key=abc123def/chat/completions": dial tcp: connection refused`))

		h.post(validBody)

		logs := h.logs.String()
		for _, secret := range []string{"key=abc123def", "sk-secret-0123456789"} {
			if strings.Contains(logs, secret) {
				t.Errorf("the log contains %q:\n%s", secret, logs)
			}
		}
		if !strings.Contains(logs, "gw.example") {
			t.Errorf("the log lost the host too, which is what it was for:\n%s", logs)
		}
	})

	t.Run("the caller's", func(t *testing.T) {
		h := newHarness(t, nil)
		h.failWith(errors.New(`inference: chat completion: Post "https://theirs.example/v1?token=zzz-secret-999/chat/completions": dial tcp: connection refused`))

		h.post(`{"state":"s","questions":{"q":{"type":"noul"}},"inference_base_url":"https://theirs.example/v1?token=zzz-secret-999"}`)

		if logs := h.logs.String(); strings.Contains(logs, "token=zzz-secret-999") {
			t.Errorf("the log contains the caller's own gateway credential:\n%s", logs)
		}
	})
}

// The threshold is not only a scrub() property: a short server key must not
// reach a caller through an upstream message either.
func TestAShortServerKeyIsStillScrubbedFromAResponse(t *testing.T) {
	const shortKey = "ollama"
	h := newHarness(t, func(c *config.Config) { c.APIKey = shortKey })
	h.failWith(&inference.APIError{
		StatusCode: http.StatusUnauthorized,
		Message:    "Incorrect API key provided: " + shortKey,
	})

	w := h.post(validBody)

	if strings.Contains(w.Body.String(), shortKey) {
		t.Errorf("the response echoed a short server key: %s", w.Body.String())
	}
	if strings.Contains(h.logs.String(), shortKey) {
		t.Errorf("the log contains a short server key:\n%s", h.logs.String())
	}
}

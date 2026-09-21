package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// testKey is the only API key used anywhere in these tests. It is deliberately
// distinctive so that a leak assertion can search for it.
const testKey = "sk-test-DO-NOT-LEAK-6f2a1c"

// testBaseURL stands in for a configured endpoint wherever a test needs a
// valid one but makes no call. The .example TLD is reserved for documentation
// and cannot resolve, so a test that started calling it would fail rather than
// reach a stranger.
const testBaseURL = "https://inference.example/v1"

// capturedRequest is one request as the fake endpoint saw it.
type capturedRequest struct {
	method string
	path   string
	header http.Header
	raw    []byte
}

// body decodes the captured request into a generic map, so a test can assert
// on the absence of a field as easily as on its value.
func (r capturedRequest) body(t *testing.T) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(r.raw, &out); err != nil {
		t.Fatalf("captured body is not JSON: %v\nbody: %s", err, r.raw)
	}
	return out
}

// format reports the response_format type of the captured request, or "" when
// the request carried none.
func (r capturedRequest) format(t *testing.T) string {
	t.Helper()
	rf, ok := r.body(t)["response_format"].(map[string]any)
	if !ok {
		return ""
	}
	s, _ := rf["type"].(string)
	return s
}

// fakeAPI is an OpenAI-compatible endpoint that records what it was sent.
type fakeAPI struct {
	*httptest.Server

	mu       sync.Mutex
	captured []capturedRequest
}

// newFakeAPI starts a server whose handler is called with the zero-based index
// of the request, so a test can behave differently on each attempt.
func newFakeAPI(t *testing.T, handle func(w http.ResponseWriter, r *http.Request, n int)) *fakeAPI {
	t.Helper()
	api := &fakeAPI{}
	api.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		// Hand the body back so a handler can inspect it too.
		r.Body = io.NopCloser(bytes.NewReader(raw))
		api.mu.Lock()
		n := len(api.captured)
		api.captured = append(api.captured, capturedRequest{
			method: r.Method,
			path:   r.URL.Path,
			header: r.Header.Clone(),
			raw:    raw,
		})
		api.mu.Unlock()
		handle(w, r, n)
	}))
	t.Cleanup(api.Close)
	return api
}

// alwaysJSON starts a server that answers every request with the same JSON
// document and a 200.
func alwaysJSON(t *testing.T, document string) *fakeAPI {
	t.Helper()
	return newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		writeJSON(w, http.StatusOK, document)
	})
}

// count reports how many requests reached the server.
func (a *fakeAPI) count() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.captured)
}

// request returns the i-th captured request.
func (a *fakeAPI) request(t *testing.T, i int) capturedRequest {
	t.Helper()
	a.mu.Lock()
	defer a.mu.Unlock()
	if i >= len(a.captured) {
		t.Fatalf("want at least %d requests, got %d", i+1, len(a.captured))
	}
	return a.captured[i]
}

// writeJSON sends a canned response body.
func writeJSON(w http.ResponseWriter, status int, document string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = io.WriteString(w, document)
}

// scoreBody builds a minimal successful completion carrying the given content.
func scoreBody(content string) string {
	raw, _ := json.Marshal(content)
	return `{"choices":[{"message":{"role":"assistant","content":` + string(raw) + `}}]}`
}

// replyFixture is one completion as a provider would send it, assembled field
// by field so that a test can stage a reply with no content, a populated
// reasoning field, or a finish reason — the three things a truncated answer is
// made of. A blank field is omitted, except content, which is sent as null
// the way a thinking model's reply arrives.
type replyFixture struct {
	content          string
	reasoning        string
	reasoningContent string
	finishReason     string
	promptTokens     int
	completionTokens int
}

// body renders the fixture as a response document.
func (f replyFixture) body() string {
	message := map[string]any{"role": "assistant", "content": nil}
	if f.content != "" {
		message["content"] = f.content
	}
	if f.reasoning != "" {
		message["reasoning"] = f.reasoning
	}
	if f.reasoningContent != "" {
		message["reasoning_content"] = f.reasoningContent
	}
	choice := map[string]any{"message": message}
	if f.finishReason != "" {
		choice["finish_reason"] = f.finishReason
	}
	document := map[string]any{"choices": []any{choice}}
	if f.promptTokens != 0 || f.completionTokens != 0 {
		document["usage"] = map[string]any{
			"prompt_tokens":     f.promptTokens,
			"completion_tokens": f.completionTokens,
		}
	}
	raw, _ := json.Marshal(document)
	return string(raw)
}

// thinkingOutLoud is what a reasoning model has produced by the time a small
// output cap stops it: the beginning of its deliberation, and no answer.
const thinkingOutLoud = "Okay, let me think about this. The state says the sky is grey, which often precedes rain, but grey skies also"

// sleepLog records the backoffs a client asked for instead of serving them.
type sleepLog struct {
	mu    sync.Mutex
	calls []time.Duration
}

func (s *sleepLog) record(ctx context.Context, d time.Duration) error {
	s.mu.Lock()
	s.calls = append(s.calls, d)
	s.mu.Unlock()
	return ctx.Err()
}

func (s *sleepLog) durations() []time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]time.Duration(nil), s.calls...)
}

// logSink is a slog handler writing into a buffer a test can inspect.
type logSink struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *logSink) handler() slog.Handler {
	return slog.NewJSONHandler(l, &slog.HandlerOptions{Level: slog.LevelDebug})
}

func (l *logSink) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

func (l *logSink) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.String()
}

// newTestClient builds a client pointed at api, with logging discarded and
// sleeping recorded rather than performed. mutate may adjust the config first.
func newTestClient(t *testing.T, api *fakeAPI, mutate func(*Config)) (*Client, *sleepLog) {
	t.Helper()
	cfg := Config{
		APIKey:  testKey,
		BaseURL: api.URL,
		Model:   "config-model",
		Logger:  slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	if mutate != nil {
		mutate(&cfg)
	}
	client, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	sleeps := &sleepLog{}
	client.sleep = sleeps.record
	return client, sleeps
}

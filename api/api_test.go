package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
)

// testKey stands in for a provider key. It is long enough to be scrubbed and
// distinctive enough to grep for.
const testKey = "sk-server-0123456789abcdef"

// evaluateFunc adapts a function to the Evaluator interface. It deliberately
// implements nothing else, so it also stands in for an evaluator that cannot
// serve a batch.
type evaluateFunc func(context.Context, classifier.Request) (classifier.Response, error)

func (f evaluateFunc) Evaluate(ctx context.Context, req classifier.Request) (classifier.Response, error) {
	return f(ctx, req)
}

// batchFunc is what the fake does when a batch is asked for.
type batchFunc func(context.Context, classifier.BatchRequest) (classifier.BatchResponse, error)

// fakeEvaluator is what the harness puts on the other side of
// Options.NewEvaluator. It answers both endpoints, because the real evaluator
// does; every call is recorded on the harness that made it.
type fakeEvaluator struct{ h *harness }

func (f *fakeEvaluator) Evaluate(ctx context.Context, req classifier.Request) (classifier.Response, error) {
	f.h.mu.Lock()
	f.h.seenRequests = append(f.h.seenRequests, req)
	fn := f.h.evaluate
	f.h.mu.Unlock()
	if fn == nil {
		return classifier.Response{}, nil
	}
	return fn(ctx, req)
}

func (f *fakeEvaluator) EvaluateBatch(ctx context.Context, req classifier.BatchRequest) (classifier.BatchResponse, error) {
	f.h.mu.Lock()
	f.h.seenBatches = append(f.h.seenBatches, req)
	fn := f.h.evaluateBatch
	f.h.mu.Unlock()
	if fn == nil {
		return classifier.BatchResponse{}, nil
	}
	return fn(ctx, req)
}

// harness is a server wired to a fake evaluator and a logger that writes to a
// buffer. No test in this package may reach the network; the fake is the only
// thing on the other side of Options.NewEvaluator.
type harness struct {
	t      *testing.T
	server *Server
	logs   *bytes.Buffer

	mu sync.Mutex
	// evaluate is what the fake evaluator does. Nil answers with an empty
	// response.
	evaluate evaluateFunc
	// evaluateBatch is what it does with a batch. Nil answers with an empty
	// response.
	evaluateBatch batchFunc
	// factoryErr, when set, fails NewEvaluator instead of building one.
	factoryErr error
	// singleOnly builds an evaluator that cannot serve a batch, which is what
	// a server wired to some other implementation would have.
	singleOnly bool
	// seen records what the server asked for, in order.
	seenSettings []Settings
	seenRequests []classifier.Request
	seenBatches  []classifier.BatchRequest
}

// newHarness builds a server from the default test configuration, after tweak
// has had a chance to change it.
func newHarness(t *testing.T, tweak func(*config.Config)) *harness {
	t.Helper()

	cfg := config.Config{
		Addr:                    ":0",
		BaseURL:                 "https://inference.example/v1",
		APIKey:                  testKey,
		Model:                   "probe-1",
		Temperature:             0,
		MaxConcurrency:          8,
		RequestTimeout:          5 * time.Second,
		MaxBodyBytes:            1 << 20,
		AllowRequestCredentials: true,
		LogLevel:                slog.LevelDebug,
		LogFormat:               config.LogFormatJSON,
	}
	if tweak != nil {
		tweak(&cfg)
	}

	h := &harness{t: t, logs: &bytes.Buffer{}}
	logger := slog.New(slog.NewJSONHandler(h.logs, &slog.HandlerOptions{Level: slog.LevelDebug}))

	srv, err := NewServer(Options{
		Config: cfg,
		Logger: logger,
		NewEvaluator: func(st Settings) (Evaluator, error) {
			h.mu.Lock()
			h.seenSettings = append(h.seenSettings, st)
			failWith := h.factoryErr
			singleOnly := h.singleOnly
			h.mu.Unlock()
			if failWith != nil {
				return nil, failWith
			}
			fake := &fakeEvaluator{h: h}
			if singleOnly {
				// A method value behind a func type: it records exactly as the
				// struct does but implements Evaluator and nothing more.
				return evaluateFunc(fake.Evaluate), nil
			}
			return fake, nil
		},
	})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	h.server = srv
	return h
}

// answer sets the response the fake evaluator returns.
func (h *harness) answer(resp classifier.Response) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.evaluate = func(context.Context, classifier.Request) (classifier.Response, error) { return resp, nil }
}

// failWith sets the error the fake evaluator returns.
func (h *harness) failWith(err error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.evaluate = func(context.Context, classifier.Request) (classifier.Response, error) {
		return classifier.Response{}, err
	}
}

// settings returns the Settings the server built the last evaluator with.
func (h *harness) settings() Settings {
	h.t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.seenSettings) == 0 {
		h.t.Fatal("the server never built an evaluator")
	}
	return h.seenSettings[len(h.seenSettings)-1]
}

// request returns the classifier request the server last submitted.
func (h *harness) request() classifier.Request {
	h.t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.seenRequests) == 0 {
		h.t.Fatal("the server never called Evaluate")
	}
	return h.seenRequests[len(h.seenRequests)-1]
}

// post sends a body to /api/evaluate.
func (h *harness) post(body string) *httptest.ResponseRecorder {
	h.t.Helper()
	return h.send(httptest.NewRequest(http.MethodPost, "/api/evaluate", strings.NewReader(body)))
}

// get sends a GET to path.
func (h *harness) get(path string) *httptest.ResponseRecorder {
	h.t.Helper()
	return h.send(httptest.NewRequest(http.MethodGet, path, nil))
}

// postReader sends a body that is read rather than handed over whole, so that
// a connection failing mid-body can be staged.
func (h *harness) postReader(body io.Reader) *httptest.ResponseRecorder {
	h.t.Helper()
	return h.send(httptest.NewRequest(http.MethodPost, "/api/evaluate", body))
}

// send runs one request through the routed handler.
func (h *harness) send(r *http.Request) *httptest.ResponseRecorder {
	h.t.Helper()
	w := httptest.NewRecorder()
	h.server.Handler().ServeHTTP(w, r)
	return w
}

// expectError asserts the status and code of an error response and returns
// its message.
func (h *harness) expectError(w *httptest.ResponseRecorder, status int, code string) string {
	h.t.Helper()
	if w.Code != status {
		h.t.Errorf("status = %d, want %d (body: %s)", w.Code, status, w.Body.String())
	}
	var body errorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		h.t.Fatalf("decoding the error body %q: %v", w.Body.String(), err)
	}
	if body.Code != code {
		h.t.Errorf("code = %q, want %q", body.Code, code)
	}
	if body.Error == "" {
		h.t.Error("the error body carries no message")
	}
	return body.Error
}

// ptr is a pointer to a value, for the nullable usage counters.
func ptr[T any](v T) *T { return &v }

// sampleBody is the worked example from the README, with a neutral model id:
// no test in this package cares which model it names, and one that borrowed
// the real default would have to be edited every time the default moved.
const sampleBody = `{
  "state": "Charged twice again!! Second month in a row.",
  "mode": "parallel",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Charges, refunds, invoices",
        "technical": "Bugs or product issues",
        "other": "Doesn't fit"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent?",
      "criteria": ["Low", "Medium", "High", "Critical"]
    },
    "angry": {
      "type": "noul",
      "instructions": "Strong frustration or anger?"
    }
  },
  "model": "probe-1"
}`

// sampleResponse is the answer the README documents for it.
func sampleResponse() classifier.Response {
	probs := classifier.NewOrderedMap[float64]()
	probs.Set("billing", 0.71)
	probs.Set("technical", 0.12)
	probs.Set("other", 0.17)

	legend := classifier.NewOrderedMap[string]()
	scores := classifier.NewOrderedMap[float64]()
	for i, label := range []string{"Low", "Medium", "High", "Critical"} {
		key := string(rune('0' + i))
		legend.Set(key, label)
		scores.Set(key, []float64{0.05, 0.15, 0.45, 0.35}[i])
	}

	answers := classifier.NewOrderedMap[classifier.Answer]()
	answers.Set("department", classifier.Answer{
		Type:          classifier.TypeChoice,
		Choice:        "billing",
		Confidence:    0.82,
		Probabilities: *probs,
	})
	answers.Set("urgency", classifier.Answer{
		Type:          classifier.TypeScore,
		Score:         2.4,
		Confidence:    0.75,
		Legend:        *legend,
		Probabilities: *scores,
	})
	answers.Set("angry", classifier.Answer{Type: classifier.TypeNoul, Noul: 0.88})

	return classifier.Response{
		Model:   "probe-1",
		Answers: *answers,
		Usage:   classifier.Usage{InputTokens: ptr(1840), OutputTokens: ptr(96)},
		Meta:    &classifier.Meta{Mode: classifier.ModeParallel, LatencyMS: 620, ParallelCalls: 9},
	}
}

func TestEvaluateHappyPath(t *testing.T) {
	h := newHarness(t, nil)
	h.answer(sampleResponse())

	w := h.post(sampleBody)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/json; charset=utf-8" {
		t.Errorf("Content-Type = %q", ct)
	}

	const want = `{"model":"probe-1",` +
		`"answers":{` +
		`"department":{"type":"choice","choice":"billing","confidence":0.82,` +
		`"probabilities":{"billing":0.71,"technical":0.12,"other":0.17}},` +
		`"urgency":{"type":"score","score":2.4,"confidence":0.75,` +
		`"legend":{"0":"Low","1":"Medium","2":"High","3":"Critical"},` +
		`"probabilities":{"0":0.05,"1":0.15,"2":0.45,"3":0.35}},` +
		`"angry":{"type":"noul","noul":0.88}},` +
		`"usage":{"input_tokens":1840,"output_tokens":96},` +
		`"meta":{"mode":"parallel","latency_ms":620,"parallel_calls":9}}`
	if got := strings.TrimSpace(w.Body.String()); got != want {
		t.Errorf("response JSON:\n got %s\nwant %s", got, want)
	}
}

func TestEvaluatePassesTheRequestThrough(t *testing.T) {
	h := newHarness(t, nil)
	h.answer(sampleResponse())

	if w := h.post(sampleBody); w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}

	req := h.request()
	state, err := req.State.Text()
	if err != nil {
		t.Fatalf("State.Text(): %v", err)
	}
	if state != "Charged twice again!! Second month in a row." {
		t.Errorf("state = %q", state)
	}
	if got := req.Questions.Len(); got != 3 {
		t.Errorf("questions = %d, want 3", got)
	}
	if got := req.Questions.Keys(); !slicesEqual(got, []string{"department", "urgency", "angry"}) {
		t.Errorf("question order = %v, want the document order", got)
	}
	if req.Model != "probe-1" {
		t.Errorf("model = %q", req.Model)
	}
	if req.Mode != classifier.ModeParallel {
		t.Errorf("mode = %q", req.Mode)
	}
	if req.Temperature != 0 {
		t.Errorf("temperature = %v, want 0", req.Temperature)
	}
}

func TestEvaluateResolvesModeModelAndTemperature(t *testing.T) {
	tests := []struct {
		name            string
		body            string
		wantModel       string
		wantTemperature float64
		wantMode        classifier.Mode
	}{
		{
			name:            "defaults come from the configuration",
			body:            `{"state":"s","questions":{"q":{"type":"noul"}}}`,
			wantModel:       "configured-model",
			wantTemperature: 0.3,
			wantMode:        classifier.ModeParallel,
		},
		{
			name:            "the body wins",
			body:            `{"state":"s","questions":{"q":{"type":"noul"}},"model":"body-model","temperature":0.9,"mode":"oneshot"}`,
			wantModel:       "body-model",
			wantTemperature: 0.9,
			wantMode:        classifier.ModeOneshot,
		},
		{
			name:            "an explicit zero temperature is honoured",
			body:            `{"state":"s","questions":{"q":{"type":"noul"}},"temperature":0}`,
			wantModel:       "configured-model",
			wantTemperature: 0,
			wantMode:        classifier.ModeParallel,
		},
		{
			name:            "blank fields fall back",
			body:            `{"state":"s","questions":{"q":{"type":"noul"}},"model":"  ","mode":"  "}`,
			wantModel:       "configured-model",
			wantTemperature: 0.3,
			wantMode:        classifier.ModeParallel,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newHarness(t, func(c *config.Config) {
				c.Model = "configured-model"
				c.Temperature = 0.3
			})
			if w := h.post(tt.body); w.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
			}
			req := h.request()
			if req.Model != tt.wantModel {
				t.Errorf("model = %q, want %q", req.Model, tt.wantModel)
			}
			if req.Temperature != tt.wantTemperature {
				t.Errorf("temperature = %v, want %v", req.Temperature, tt.wantTemperature)
			}
			if req.Mode != tt.wantMode {
				t.Errorf("mode = %q, want %q", req.Mode, tt.wantMode)
			}
			if got := h.settings().Model; got != tt.wantModel {
				t.Errorf("Settings.Model = %q, want %q", got, tt.wantModel)
			}
		})
	}
}

func TestEvaluateSettingsCarryTheConfiguredLimits(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.MaxConcurrency = 3 })
	if w := h.post(`{"state":"s","questions":{"q":{"type":"noul"}}}`); w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	st := h.settings()
	if st.MaxConcurrency != 3 {
		t.Errorf("Settings.MaxConcurrency = %d, want 3", st.MaxConcurrency)
	}
	if st.APIKey != testKey {
		t.Errorf("Settings.APIKey = %q, want the configured key", st.APIKey)
	}
	if st.BaseURL != "https://inference.example/v1" {
		t.Errorf("Settings.BaseURL = %q, want the configured base URL", st.BaseURL)
	}
}

func TestEvaluateSettingsCarryTheConfiguredReasoningEffort(t *testing.T) {
	for _, effort := range []string{"", config.EffortNone, config.EffortHigh} {
		name := effort
		if name == "" {
			name = "unset"
		}
		t.Run(name, func(t *testing.T) {
			h := newHarness(t, func(c *config.Config) { c.ReasoningEffort = effort })
			if w := h.post(validBody); w.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
			}
			if got := h.settings().ReasoningEffort; got != effort {
				t.Errorf("Settings.ReasoningEffort = %q, want %q", got, effort)
			}
		})
	}
}

// The reasoning effort is a property of the model the operator chose, not of
// the question being asked, so it is server-side only. A body that tries to
// name one is rejected like any other unknown field rather than quietly
// ignored — silently dropping it would read as support.
func TestEvaluateRejectsAReasoningEffortInTheBody(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.ReasoningEffort = config.EffortNone })

	message := h.expectError(
		h.post(`{"state":"s","questions":{"q":{"type":"noul"}},"reasoning_effort":"high"}`),
		http.StatusBadRequest, codeInvalidJSON)
	if !strings.Contains(message, "reasoning_effort") {
		t.Errorf("message = %q, want it to name the field it rejected", message)
	}
	if len(h.seenSettings) != 0 {
		t.Error("a rejected request still reached the evaluator")
	}
}

func TestEvaluateRejectsAWrongMethod(t *testing.T) {
	h := newHarness(t, nil)
	for _, method := range []string{http.MethodGet, http.MethodPut, http.MethodDelete} {
		w := h.send(httptest.NewRequest(method, "/api/evaluate", nil))
		h.expectError(w, http.StatusMethodNotAllowed, "method_not_allowed")
		if got := w.Header().Get("Allow"); got != http.MethodPost {
			t.Errorf("%s /api/evaluate: Allow = %q, want %q", method, got, http.MethodPost)
		}
		if ct := w.Header().Get("Content-Type"); ct != "application/json; charset=utf-8" {
			t.Errorf("%s /api/evaluate: Content-Type = %q, want JSON like every other failure", method, ct)
		}
		if len(h.seenSettings) != 0 {
			t.Errorf("%s /api/evaluate reached the evaluator", method)
		}
	}
}

// Two paths are served, /api/evaluate and /api/health. Every other path,
// whatever the method, falls through to the catch-all and answers in the same
// JSON shape as every other failure rather than in the mux's own prose.
func TestUnknownPathIs404(t *testing.T) {
	h := newHarness(t, nil)
	// "/api/providers" is in this list deliberately. It was a real route once,
	// listing the endpoint presets that configuration no longer has, and a
	// route that used to exist is the one an editor is most likely to restore
	// by reflex. It must 404 like any other unknown path.
	//
	// "/api/evaluate/extra" and "/api/evaluate/batch/extra" are the other
	// reason this list is worth keeping: /api/evaluate/batch is a path below
	// a path, and a pattern written with a trailing slash would turn either
	// of those into a silent 200 against the wrong handler.
	for _, path := range []string{
		"/api/nope", "/", "/api/evaluate/extra", "/api/evaluate/batch/extra",
		"/api/meta", "/api/models", "/api/providers",
	} {
		for _, method := range []string{http.MethodGet, http.MethodPost} {
			w := h.send(httptest.NewRequest(method, path, strings.NewReader("")))
			h.expectError(w, http.StatusNotFound, "not_found")
			if ct := w.Header().Get("Content-Type"); ct != "application/json; charset=utf-8" {
				t.Errorf("%s %s: Content-Type = %q, want JSON like every other failure", method, path, ct)
			}
		}
	}
}

// The 404 body must not echo the path back: a response is not a mirror.
func TestNotFoundDoesNotEchoThePath(t *testing.T) {
	h := newHarness(t, nil)
	w := h.get("/api/<script>alert(1)</script>")
	message := h.expectError(w, http.StatusNotFound, "not_found")
	if strings.Contains(message, "script") {
		t.Errorf("the 404 message echoed the caller's path: %q", message)
	}
}

func TestPanicInAHandlerIs500WithALogLine(t *testing.T) {
	h := newHarness(t, nil)
	h.evaluate = func(context.Context, classifier.Request) (classifier.Response, error) {
		panic("the handler came apart")
	}

	w := h.post(validBody)

	message := h.expectError(w, http.StatusInternalServerError, codeInternal)
	if strings.Contains(message, "came apart") {
		t.Errorf("the panic value reached the caller: %q", message)
	}
	line := findLog(t, h.logs, "request failed")
	detail, _ := line["error"].(string)
	if !strings.Contains(detail, "the handler came apart") {
		t.Errorf("the log line does not carry the panic: %v", line)
	}
	if !strings.Contains(detail, "api.") {
		t.Errorf("the log line does not carry a stack: %q", detail)
	}
	if line["status"] != float64(http.StatusInternalServerError) {
		t.Errorf("status = %v, want 500", line["status"])
	}
	if line["method"] != http.MethodPost || line["path"] != "/api/evaluate" {
		t.Errorf("the access log line lost the request: %v", line)
	}
}

// http.ErrAbortHandler is net/http's way of saying "drop this connection on
// purpose". It must travel on, or a handler loses the only panic value that
// means something.
func TestErrAbortHandlerIsNotSwallowed(t *testing.T) {
	h := newHarness(t, nil)
	h.evaluate = func(context.Context, classifier.Request) (classifier.Response, error) {
		panic(http.ErrAbortHandler)
	}

	var recovered any
	func() {
		defer func() { recovered = recover() }()
		h.post(validBody)
	}()

	if recovered != http.ErrAbortHandler {
		t.Errorf("recovered %v, want http.ErrAbortHandler to be re-panicked", recovered)
	}
	// The request is still counted, so an aborted connection is not invisible.
	findLog(t, h.logs, "request")
}

// The panic value's own text is a credential risk like any other 5xx detail.
func TestAPanicIsScrubbedBeforeItIsLogged(t *testing.T) {
	h := newHarness(t, nil)
	h.evaluate = func(context.Context, classifier.Request) (classifier.Response, error) {
		panic("upstream rejected " + testKey)
	}

	h.post(validBody)

	if logs := h.logs.String(); strings.Contains(logs, testKey) {
		t.Errorf("a panic put the API key in the log:\n%s", logs)
	}
}

func TestALongModeIsClippedInTheLog(t *testing.T) {
	h := newHarness(t, nil)
	long := strings.Repeat("m", 20_000)

	h.post(fmt.Sprintf(`{"state":"s","questions":{"q":{"type":"noul"}},"mode":%q}`, long))

	line := findLog(t, h.logs, "request")
	mode, _ := line["mode"].(string)
	if mode == "" {
		t.Fatalf("no mode in the log line: %v", line)
	}
	if len(mode) > 128 {
		t.Errorf("the log line carries %d bytes of caller-supplied mode", len(mode))
	}
	// The classifier still sees the whole thing, so the rejection is honest.
	if got := string(h.request().Mode); got != long {
		t.Errorf("the evaluator was handed a clipped mode (%d bytes)", len(got))
	}
}

// writeJSON's marshal failure is the one branch a handler cannot reach with a
// well-typed response, and the branch that decides what a caller sees if one
// ever grows a field that cannot be encoded.
func TestWriteJSONReportsAValueItCannotEncode(t *testing.T) {
	h := newHarness(t, nil)
	rec := &statusRecorder{ResponseWriter: httptest.NewRecorder(), status: http.StatusOK}

	h.server.writeJSON(rec, http.StatusOK, make(chan int))

	inner := rec.ResponseWriter.(*httptest.ResponseRecorder)
	if inner.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", inner.Code)
	}
	var body errorResponse
	if err := json.Unmarshal(inner.Body.Bytes(), &body); err != nil {
		t.Fatalf("decoding %q: %v", inner.Body.String(), err)
	}
	if body.Code != codeInternal {
		t.Errorf("code = %q, want %q", body.Code, codeInternal)
	}
	if !strings.Contains(rec.detail, "encoding the response") {
		t.Errorf("detail = %q, want the encoding failure kept for the log", rec.detail)
	}
}

func TestEvaluateLogsOneLinePerRequest(t *testing.T) {
	h := newHarness(t, nil)
	h.answer(sampleResponse())
	h.post(sampleBody)

	line := findLog(t, h.logs, "request")
	if line["status"] != float64(http.StatusOK) {
		t.Errorf("status = %v, want 200", line["status"])
	}
	if line["method"] != http.MethodPost || line["path"] != "/api/evaluate" {
		t.Errorf("method/path = %v %v", line["method"], line["path"])
	}
	if line["questions"] != float64(3) {
		t.Errorf("questions = %v, want 3", line["questions"])
	}
	if line["mode"] != "parallel" {
		t.Errorf("mode = %v, want parallel", line["mode"])
	}
	if _, ok := line["duration"]; !ok {
		t.Error("the log line has no duration")
	}
}

func TestEvaluateNeverLogsTheCallersData(t *testing.T) {
	h := newHarness(t, nil)
	h.answer(sampleResponse())
	h.post(sampleBody)

	logs := h.logs.String()
	for _, secret := range []string{"Charged twice again", "Charges, refunds, invoices", "Which team should handle this?"} {
		if strings.Contains(logs, secret) {
			t.Errorf("the log contains the caller's data %q:\n%s", secret, logs)
		}
	}
}

// findLog returns the first log line with the given message.
func findLog(t *testing.T, buf *bytes.Buffer, msg string) map[string]any {
	t.Helper()
	for line := range strings.SplitSeq(strings.TrimSpace(buf.String()), "\n") {
		if line == "" {
			continue
		}
		var entry map[string]any
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			t.Fatalf("log line %q is not JSON: %v", line, err)
		}
		if entry["msg"] == msg {
			return entry
		}
	}
	t.Fatalf("no log line with msg=%q in:\n%s", msg, buf.String())
	return nil
}

// slicesEqual compares two string slices.
func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// The softmax temperature is the operator's, like the reasoning effort and
// the scorer: it is what every reported probability is calibrated by, and a
// caller who could change it between two requests would change what a
// threshold means. It has to reach the evaluator's settings from the
// configuration and from nowhere else.
func TestEvaluateSettingsCarryTheConfiguredSoftmaxTemperature(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.SoftmaxTemperature = 5.04 })
	if w := h.post(validBody); w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}
	if got := h.settings().SoftmaxTemperature; got != 5.04 {
		t.Errorf("Settings.SoftmaxTemperature = %v, want 5.04", got)
	}
}

// The batch endpoint resolves its settings on its own line, so it gets its
// own check rather than being assumed to follow.
func TestBatchSettingsCarryTheConfiguredSoftmaxTemperature(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.SoftmaxTemperature = 5.04 })
	if w := h.postBatch(sampleBatchBody); w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}
	if got := h.settings().SoftmaxTemperature; got != 5.04 {
		t.Errorf("Settings.SoftmaxTemperature = %v, want 5.04", got)
	}
}

// A Config that names no softmax temperature — a zero value, or one a caller
// assembled in Go and left out — takes the default rather than the zero,
// which the softmax would floor to 0.05 and report a near one-hot
// distribution from. NaN takes it too: the loader refuses one, but this
// struct is exported.
func TestNewServerDefaultsTheSoftmaxTemperature(t *testing.T) {
	for _, bad := range []float64{0, -1, math.NaN()} {
		h := newHarness(t, func(c *config.Config) { c.SoftmaxTemperature = bad })
		if got := h.server.Config().SoftmaxTemperature; got != config.DefaultSoftmaxTemperature {
			t.Errorf("a configured %v resolved to %v, want the default %v",
				bad, got, config.DefaultSoftmaxTemperature)
		}
		if w := h.post(validBody); w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		if got := h.settings().SoftmaxTemperature; got != config.DefaultSoftmaxTemperature {
			t.Errorf("a configured %v reached the evaluator as %v, want the default %v",
				bad, got, config.DefaultSoftmaxTemperature)
		}
	}
}

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
	"github.com/titusai-io/perceptea/provider/inference"
)

// postBatch sends a body to /api/evaluate/batch.
func (h *harness) postBatch(body string) *httptest.ResponseRecorder {
	h.t.Helper()
	return h.send(httptest.NewRequest(http.MethodPost, "/api/evaluate/batch", strings.NewReader(body)))
}

// answerBatch sets the response the fake evaluator returns for a batch.
func (h *harness) answerBatch(resp classifier.BatchResponse) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.evaluateBatch = func(context.Context, classifier.BatchRequest) (classifier.BatchResponse, error) {
		return resp, nil
	}
}

// batch returns the batch request the server last submitted.
func (h *harness) batch() classifier.BatchRequest {
	h.t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.seenBatches) == 0 {
		h.t.Fatal("the server never called EvaluateBatch")
	}
	return h.seenBatches[len(h.seenBatches)-1]
}

// sampleBatchBody is two states against one shared question.
const sampleBatchBody = `{
  "items": [
    {"id": "t-1", "state": "Charged twice again!! Second month in a row."},
    {"id": "t-2", "state": "Thanks, all sorted."}
  ],
  "questions": {
    "angry": {"type": "noul", "instructions": "Strong frustration or anger?"}
  },
  "model": "probe-1",
  "mode": "parallel"
}`

// noulAnswers is one answered noul question, as a classifier would return it.
func noulAnswers(name string, p float64) classifier.Answers {
	answers := classifier.NewOrderedMap[classifier.Answer]()
	answers.Set(name, classifier.Answer{Type: classifier.TypeNoul, Noul: p})
	return *answers
}

// sampleBatchResponse is one answered item and one that failed, which is the
// shape the endpoint exists to be able to return.
func sampleBatchResponse() classifier.BatchResponse {
	return classifier.BatchResponse{
		Model: "probe-1",
		Results: []classifier.BatchResult{
			{
				ID:      "t-1",
				Index:   0,
				Answers: noulAnswers("angry", 0.88),
				Usage:   classifier.Usage{InputTokens: ptr(120), OutputTokens: ptr(6)},
			},
			{
				ID:    "t-2",
				Index: 1,
				Error: "upstream provider error (status 502): no message",
			},
		},
		Usage: classifier.Usage{InputTokens: ptr(120), OutputTokens: ptr(6)},
		Meta: &classifier.BatchMeta{
			Mode:          classifier.ModeParallel,
			LatencyMS:     640,
			ParallelCalls: 2,
			Items:         2,
			Succeeded:     1,
			Failed:        1,
		},
	}
}

func TestBatchHappyPath(t *testing.T) {
	h := newHarness(t, nil)
	h.answerBatch(sampleBatchResponse())

	w := h.postBatch(sampleBatchBody)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/json; charset=utf-8" {
		t.Errorf("Content-Type = %q", ct)
	}

	const want = `{"model":"probe-1","results":[` +
		`{"id":"t-1","index":0,"answers":{"angry":{"type":"noul","noul":0.88}},` +
		`"usage":{"input_tokens":120,"output_tokens":6}},` +
		`{"id":"t-2","index":1,"answers":{},"usage":{"input_tokens":null,"output_tokens":null},` +
		`"error":"upstream provider error (status 502): no message"}],` +
		`"usage":{"input_tokens":120,"output_tokens":6},` +
		`"meta":{"mode":"parallel","latency_ms":640,"parallel_calls":2,"items":2,"succeeded":1,"failed":1}}`
	if got := strings.TrimSpace(w.Body.String()); got != want {
		t.Errorf("response JSON:\n got %s\nwant %s", got, want)
	}
}

func TestBatchPassesTheRequestThrough(t *testing.T) {
	h := newHarness(t, nil)
	h.answerBatch(sampleBatchResponse())

	if w := h.postBatch(sampleBatchBody); w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}

	req := h.batch()
	if len(req.Items) != 2 {
		t.Fatalf("the evaluator saw %d items, want 2", len(req.Items))
	}
	for i, want := range []struct{ id, state string }{
		{"t-1", "Charged twice again!! Second month in a row."},
		{"t-2", "Thanks, all sorted."},
	} {
		if req.Items[i].ID != want.id {
			t.Errorf("items[%d].ID = %q, want %q", i, req.Items[i].ID, want.id)
		}
		state, err := req.Items[i].State.Text()
		if err != nil {
			t.Fatalf("items[%d].State.Text(): %v", i, err)
		}
		if state != want.state {
			t.Errorf("items[%d] state = %q, want %q", i, state, want.state)
		}
	}
	if got := req.Questions.Keys(); !slicesEqual(got, []string{"angry"}) {
		t.Errorf("questions = %v", got)
	}
	if req.Model != "probe-1" {
		t.Errorf("model = %q", req.Model)
	}
	if req.Mode != classifier.ModeParallel {
		t.Errorf("mode = %q", req.Mode)
	}
	// The single endpoint was not used on the way.
	if len(h.seenRequests) != 0 {
		t.Errorf("a batch request also called Evaluate %d times", len(h.seenRequests))
	}
}

// The batch endpoint resolves the model, temperature and mode by the same
// rules as the single one, from the same fields.
func TestBatchResolvesModeModelAndTemperature(t *testing.T) {
	tests := []struct {
		name            string
		body            string
		wantModel       string
		wantTemperature float64
		wantMode        classifier.Mode
	}{
		{
			name:            "defaults come from the configuration",
			body:            `{"items":[{"state":"s"}],"questions":{"q":{"type":"noul"}}}`,
			wantModel:       "configured-model",
			wantTemperature: 0.3,
			wantMode:        classifier.ModeParallel,
		},
		{
			name: "the body wins",
			body: `{"items":[{"state":"s"}],"questions":{"q":{"type":"noul"}},` +
				`"model":"body-model","temperature":0.9,"mode":"oneshot"}`,
			wantModel:       "body-model",
			wantTemperature: 0.9,
			wantMode:        classifier.ModeOneshot,
		},
		{
			name:            "an explicit zero temperature is honoured",
			body:            `{"items":[{"state":"s"}],"questions":{"q":{"type":"noul"}},"temperature":0}`,
			wantModel:       "configured-model",
			wantTemperature: 0,
			wantMode:        classifier.ModeParallel,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newHarness(t, func(c *config.Config) {
				c.Model = "configured-model"
				c.Temperature = 0.3
			})
			if w := h.postBatch(tt.body); w.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
			}
			req := h.batch()
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

// The same credential rules, from the same fields, with the same rejection of
// a base URL the caller could not have meant.
func TestBatchUsesTheSameCredentialRules(t *testing.T) {
	t.Run("the body's credentials are taken when allowed", func(t *testing.T) {
		h := newHarness(t, nil)
		body := `{"items":[{"state":"s"}],"questions":{"q":{"type":"noul"}},` +
			`"api_key":"sk-caller-0123456789","inference_base_url":"https://openrouter.ai/api/v1"}`
		if w := h.postBatch(body); w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
		}
		st := h.settings()
		if st.APIKey != "sk-caller-0123456789" {
			t.Errorf("Settings.APIKey = %q, want the one from the body", st.APIKey)
		}
		if st.BaseURL != "https://openrouter.ai/api/v1" {
			t.Errorf("Settings.BaseURL = %q, want the one from the body", st.BaseURL)
		}
	})

	t.Run("they are ignored when the server forbids them", func(t *testing.T) {
		h := newHarness(t, func(c *config.Config) { c.AllowRequestCredentials = false })
		body := `{"items":[{"state":"s"}],"questions":{"q":{"type":"noul"}},` +
			`"api_key":"sk-caller-0123456789","inference_base_url":"https://openrouter.ai/api/v1"}`
		if w := h.postBatch(body); w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
		}
		if st := h.settings(); st.APIKey != testKey {
			t.Errorf("Settings.APIKey = %q, want the server's own", st.APIKey)
		}
	})

	t.Run("an unusable base URL is the caller's 400", func(t *testing.T) {
		h := newHarness(t, nil)
		body := `{"items":[{"state":"s"}],"questions":{"q":{"type":"noul"}},` +
			`"inference_base_url":"openrouter.ai/api/v1"}`
		msg := h.expectError(h.postBatch(body), http.StatusBadRequest, codeInvalidRequest)
		if !strings.Contains(msg, "inference_base_url") {
			t.Errorf("message %q does not name the field at fault", msg)
		}
		if len(h.seenSettings) != 0 {
			t.Error("the request reached the provider anyway")
		}
	})
}

// ----------------------------------------------------------- the rejections --

func TestBatchRejectsAnEmptyItemList(t *testing.T) {
	for _, body := range []string{
		`{"questions":{"q":{"type":"noul"}}}`,
		`{"items":[],"questions":{"q":{"type":"noul"}}}`,
		`{"items":null,"questions":{"q":{"type":"noul"}}}`,
	} {
		h := newHarness(t, nil)
		msg := h.expectError(h.postBatch(body), http.StatusBadRequest, codeInvalidRequest)
		if !strings.Contains(msg, "items") {
			t.Errorf("message %q does not name the field at fault", msg)
		}
		if len(h.seenSettings) != 0 {
			t.Errorf("%s reached the evaluator", body)
		}
	}
}

func TestBatchRejectsAnEmptyQuestionSet(t *testing.T) {
	h := newHarness(t, nil)
	msg := h.expectError(h.postBatch(`{"items":[{"state":"s"}]}`), http.StatusBadRequest, codeInvalidRequest)
	if !strings.Contains(msg, "questions") {
		t.Errorf("message %q does not name the field at fault", msg)
	}
	if len(h.seenSettings) != 0 {
		t.Error("a batch with no questions reached the evaluator")
	}
}

// The ceiling is what stops one request committing to unbounded provider
// spend, so the rejection names the setting and both numbers: an operator
// reading a caller's bug report needs to know what was asked for and which
// variable to raise.
func TestBatchRejectsTooManyItems(t *testing.T) {
	const limit = 3
	h := newHarness(t, func(c *config.Config) { c.MaxBatchItems = limit })

	items := make([]string, limit+1)
	for i := range items {
		items[i] = fmt.Sprintf(`{"id":"t-%d","state":"s"}`, i)
	}
	body := `{"items":[` + strings.Join(items, ",") + `],"questions":{"q":{"type":"noul"}}}`

	msg := h.expectError(h.postBatch(body), http.StatusBadRequest, codeInvalidRequest)
	for _, want := range []string{config.EnvMaxBatchItems, "3", "4"} {
		if !strings.Contains(msg, want) {
			t.Errorf("message %q does not carry %q", msg, want)
		}
	}
	if len(h.seenSettings) != 0 {
		t.Error("an over-sized batch reached the evaluator")
	}
}

// Exactly the limit is not over it.
func TestBatchAcceptsExactlyTheLimit(t *testing.T) {
	const limit = 3
	h := newHarness(t, func(c *config.Config) { c.MaxBatchItems = limit })

	items := make([]string, limit)
	for i := range items {
		items[i] = fmt.Sprintf(`{"id":"t-%d","state":"s"}`, i)
	}
	body := `{"items":[` + strings.Join(items, ",") + `],"questions":{"q":{"type":"noul"}}}`

	if w := h.postBatch(body); w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}
	if got := len(h.batch().Items); got != limit {
		t.Errorf("the evaluator saw %d items, want %d", got, limit)
	}
}

// A server built without a ceiling gets the documented default rather than
// rejecting every batch it is ever sent.
func TestBatchCeilingDefaultsWhenUnset(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.MaxBatchItems = 0 })
	if got := h.server.Config().MaxBatchItems; got != config.DefaultMaxBatchItems {
		t.Fatalf("MaxBatchItems = %d, want the default %d", got, config.DefaultMaxBatchItems)
	}
	if w := h.postBatch(`{"items":[{"state":"s"}],"questions":{"q":{"type":"noul"}}}`); w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}
}

// The same rule as the single endpoint: a field nobody reads is a 400 naming
// it, not a silently ignored typo that spends the server's key.
func TestBatchRejectsAnUnknownField(t *testing.T) {
	h := newHarness(t, nil)
	body := `{"items":[{"state":"s"}],"questions":{"q":{"type":"noul"}},"apikey":"sk-oops-0123456789"}`
	msg := h.expectError(h.postBatch(body), http.StatusBadRequest, codeInvalidJSON)
	if !strings.Contains(msg, "apikey") {
		t.Errorf("message %q does not name the field it rejected", msg)
	}
	if len(h.seenSettings) != 0 {
		t.Error("a rejected request still reached the evaluator")
	}
}

// And the same rule about a body carrying more than one document.
func TestBatchRejectsTrailingContent(t *testing.T) {
	const valid = `{"items":[{"state":"s"}],"questions":{"q":{"type":"noul"}}}`
	for _, suffix := range []string{`{"items":[]}`, "garbage", "]"} {
		h := newHarness(t, nil)
		h.expectError(h.postBatch(valid+suffix), http.StatusBadRequest, codeInvalidJSON)
		if len(h.seenSettings) != 0 {
			t.Errorf("a body with %q after it reached the evaluator", suffix)
		}
	}
}

// The body limit is shared with the single endpoint, and a batch is the body
// most likely to reach it.
func TestBatchHonoursTheSharedBodyLimit(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.MaxBodyBytes = 512 })

	items := make([]string, 50)
	for i := range items {
		items[i] = fmt.Sprintf(`{"id":"t-%d","state":%q}`, i, strings.Repeat("x", 100))
	}
	body := `{"items":[` + strings.Join(items, ",") + `],"questions":{"q":{"type":"noul"}}}`

	msg := h.expectError(h.postBatch(body), http.StatusRequestEntityTooLarge, codePayloadTooLarge)
	if !strings.Contains(msg, "512") {
		t.Errorf("message %q does not name the limit", msg)
	}
}

func TestBatchRejectsAWrongMethod(t *testing.T) {
	h := newHarness(t, nil)
	for _, method := range []string{http.MethodGet, http.MethodPut, http.MethodDelete} {
		w := h.send(httptest.NewRequest(method, "/api/evaluate/batch", nil))
		h.expectError(w, http.StatusMethodNotAllowed, codeMethodNotAllowed)
		if got := w.Header().Get("Allow"); got != http.MethodPost {
			t.Errorf("%s /api/evaluate/batch: Allow = %q, want %q", method, got, http.MethodPost)
		}
		if len(h.seenSettings) != 0 {
			t.Errorf("%s /api/evaluate/batch reached the evaluator", method)
		}
	}
}

// ------------------------------------------------- partial and total failure --

// A batch in which every single item failed is still a batch that was served.
// The request succeeded; the items did not, and saying so is exactly what the
// per-item errors are for. Only a whole-request failure uses the error table.
func TestBatchWhereEveryItemFailedIsStill200(t *testing.T) {
	h := newHarness(t, nil)
	h.answerBatch(classifier.BatchResponse{
		Model: "probe-1",
		Results: []classifier.BatchResult{
			{ID: "t-1", Index: 0, Error: "upstream provider error (status 502): no message"},
			{ID: "t-2", Index: 1, Error: "upstream provider error (status 502): no message"},
		},
		Meta: &classifier.BatchMeta{Mode: classifier.ModeParallel, Items: 2, Failed: 2},
	})

	w := h.postBatch(sampleBatchBody)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: the request was served even though no item was (body: %s)",
			w.Code, w.Body.String())
	}
	var body classifier.BatchResponse
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decoding %q: %v", w.Body.String(), err)
	}
	if len(body.Results) != 2 {
		t.Fatalf("got %d results, want 2", len(body.Results))
	}
	for _, r := range body.Results {
		if r.Error == "" {
			t.Errorf("result %d lost its error on the way out", r.Index)
		}
	}
	// The error body shape is for failures of the request, and this was not
	// one: nothing in the response should look like one.
	if strings.Contains(w.Body.String(), `"code"`) {
		t.Errorf("a served batch answered in the error shape: %s", w.Body.String())
	}
}

// A whole-request failure is reported from the error table like any other.
func TestBatchWholeRequestFailuresUseTheErrorTable(t *testing.T) {
	tests := []struct {
		name       string
		err        error
		wantStatus int
		wantCode   string
	}{
		{
			name:       "no questions",
			err:        classifier.ErrNoQuestions,
			wantStatus: http.StatusBadRequest,
			wantCode:   codeInvalidRequest,
		},
		{
			name:       "no items",
			err:        classifier.ErrNoItems,
			wantStatus: http.StatusBadRequest,
			wantCode:   codeInvalidRequest,
		},
		{
			name:       "an unknown mode",
			err:        fmt.Errorf("%w %q", classifier.ErrUnknownMode, "batched"),
			wantStatus: http.StatusBadRequest,
			wantCode:   codeUnsupportedMode,
		},
		{
			name:       "a question that does not validate",
			err:        &classifier.ValidationError{Question: "urgency", Field: "criteria", Message: "needs levels"},
			wantStatus: http.StatusBadRequest,
			wantCode:   codeInvalidRequest,
		},
		{
			name:       "the provider rate-limited the job",
			err:        &inference.APIError{StatusCode: http.StatusTooManyRequests, Message: "slow down"},
			wantStatus: http.StatusTooManyRequests,
			wantCode:   codeUpstreamRateLimited,
		},
		{
			name:       "the caller hung up",
			err:        context.Canceled,
			wantStatus: StatusClientClosedRequest,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := newHarness(t, nil)
			h.mu.Lock()
			h.evaluateBatch = func(context.Context, classifier.BatchRequest) (classifier.BatchResponse, error) {
				return classifier.BatchResponse{}, tt.err
			}
			h.mu.Unlock()

			w := h.postBatch(sampleBatchBody)
			if tt.wantCode == "" {
				if w.Code != tt.wantStatus {
					t.Fatalf("status = %d, want %d", w.Code, tt.wantStatus)
				}
				if w.Body.Len() != 0 {
					t.Errorf("a %d carried a body: %s", tt.wantStatus, w.Body.String())
				}
				return
			}
			h.expectError(w, tt.wantStatus, tt.wantCode)
		})
	}
}

// A per-item error is a message on its way to a caller like any other: an
// upstream failure can quote the endpoint it was talking to, and that URL can
// carry a credential.
func TestBatchScrubsPerItemErrors(t *testing.T) {
	const callerKey = "sk-caller-9876543210"
	h := newHarness(t, nil)
	h.mu.Lock()
	h.evaluateBatch = func(_ context.Context, req classifier.BatchRequest) (classifier.BatchResponse, error) {
		return classifier.BatchResponse{
			Results: []classifier.BatchResult{
				{ID: "t-1", Index: 0, Error: "calling https://gw.example/v1 with " + testKey},
				{ID: "t-2", Index: 1, Error: "calling https://gw.example/v1 with " + callerKey},
			},
			Meta: &classifier.BatchMeta{Items: 2, Failed: 2},
		}, nil
	}
	h.mu.Unlock()

	body := `{"items":[{"id":"t-1","state":"s"},{"id":"t-2","state":"s"}],` +
		`"questions":{"q":{"type":"noul"}},"api_key":"` + callerKey + `"}`
	w := h.postBatch(body)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", w.Code, w.Body.String())
	}
	out := w.Body.String()
	for _, secret := range []string{testKey, callerKey} {
		if strings.Contains(out, secret) {
			t.Errorf("a per-item error carried the credential %q to the caller:\n%s", secret, out)
		}
	}
	if !strings.Contains(out, "[redacted]") {
		t.Errorf("nothing was redacted, so the scrubber never ran:\n%s", out)
	}
}

// The batch path is the one place where an evaluator might not be able to do
// what was asked of it, because Options.NewEvaluator promises only Evaluator.
func TestBatchWithAnEvaluatorThatCannotBatch(t *testing.T) {
	h := newHarness(t, nil)
	h.mu.Lock()
	h.singleOnly = true
	h.mu.Unlock()

	message := h.expectError(h.postBatch(sampleBatchBody), http.StatusInternalServerError, codeInternal)
	if strings.Contains(message, "evaluateFunc") {
		t.Errorf("the caller was told the server's own type: %q", message)
	}
	line := findLog(t, h.logs, "request failed")
	if detail, _ := line["error"].(string); !strings.Contains(detail, "BatchEvaluator") {
		t.Errorf("the log line does not say what was wrong: %v", line)
	}
	if len(h.seenBatches) != 0 {
		t.Error("an evaluator that cannot batch was asked to")
	}
}

// ------------------------------------------------------------------ logging --

func TestBatchLogsTheItemCount(t *testing.T) {
	h := newHarness(t, nil)
	h.answerBatch(sampleBatchResponse())
	h.postBatch(sampleBatchBody)

	line := findLog(t, h.logs, "request")
	if line["status"] != float64(http.StatusOK) {
		t.Errorf("status = %v, want 200", line["status"])
	}
	if line["path"] != "/api/evaluate/batch" {
		t.Errorf("path = %v", line["path"])
	}
	if line["items"] != float64(2) {
		t.Errorf("items = %v, want 2: a hundred-state batch must not log like a single evaluation", line["items"])
	}
	if line["questions"] != float64(1) {
		t.Errorf("questions = %v, want 1", line["questions"])
	}
}

// A single evaluation has no items, so the field stays off its log line.
func TestSingleEvaluationLogsNoItemCount(t *testing.T) {
	h := newHarness(t, nil)
	h.answer(sampleResponse())
	h.post(sampleBody)

	line := findLog(t, h.logs, "request")
	if _, ok := line["items"]; ok {
		t.Errorf("a single evaluation logged an item count: %v", line)
	}
}

// ------------------------------------------------------------------- docs --

// The README states the batch ceiling as a number, in two places, and a
// number written down in prose is a number that goes stale. This ties both to
// the constant the server actually applies.
//
// It reads only where a *default* is being stated — the configuration table's
// Default column, and the "(default `n`)" in the endpoint's own section — not
// every figure on those lines. The row also names the status an over-sized
// batch gets, and a test that could not tell 400 the status from 100 the
// ceiling would fail on a sentence that is perfectly correct.
func TestREADMEDocumentsTheBatchCeilingThatIsApplied(t *testing.T) {
	readme, err := os.ReadFile(filepath.Join("..", "README.md"))
	if err != nil {
		t.Fatalf("reading the README: %v", err)
	}

	want := "`" + strconv.Itoa(config.DefaultMaxBatchItems) + "`"
	variable := "`" + config.EnvMaxBatchItems + "`"
	// "…(default `100`)…", the only other place a default is written out.
	prose := regexp.MustCompile("default (`[0-9]+`)")

	stated := 0
	for line := range strings.SplitSeq(string(readme), "\n") {
		if !strings.Contains(line, variable) {
			continue
		}
		var found []string
		if cells := strings.Split(line, "|"); strings.HasPrefix(strings.TrimSpace(line), "|") && len(cells) > 3 &&
			strings.TrimSpace(cells[1]) == variable {
			found = append(found, strings.TrimSpace(cells[2]))
		}
		for _, m := range prose.FindAllStringSubmatch(line, -1) {
			found = append(found, m[1])
		}
		for _, got := range found {
			stated++
			if got != want {
				t.Errorf("the README documents %s as %s; the server applies %s:\n  %s",
					config.EnvMaxBatchItems, got, want, strings.TrimSpace(line))
			}
		}
	}
	// Two: the configuration table and the endpoint's section. A drop to one
	// means a statement of the default was deleted or reworded past this
	// check, which is how the other one goes stale unnoticed.
	if stated != 2 {
		t.Errorf("the README states what %s defaults to %d times, want 2",
			config.EnvMaxBatchItems, stated)
	}
	// And the endpoint itself is listed, so the route and the documentation
	// cannot part company silently.
	if !strings.Contains(string(readme), "/api/evaluate/batch") {
		t.Error("the README does not mention the batch endpoint")
	}
}

func TestBatchNeverLogsTheCallersData(t *testing.T) {
	h := newHarness(t, nil)
	h.answerBatch(sampleBatchResponse())
	h.postBatch(sampleBatchBody)

	logs := h.logs.String()
	for _, secret := range []string{"Charged twice again", "Thanks, all sorted", "Strong frustration"} {
		if strings.Contains(logs, secret) {
			t.Errorf("the log contains the caller's data %q:\n%s", secret, logs)
		}
	}
}

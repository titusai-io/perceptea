package classifier

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeGenerator is a Scorer that can also generate. Score is never expected to
// be called in oneshot mode, so it fails the test if it is.
type fakeGenerator struct {
	t            *testing.T
	content      string
	err          error
	inputTokens  int
	outputTokens int

	mu     sync.Mutex
	seen   []GenerateRequest
	scored int
}

func (f *fakeGenerator) Score(_ context.Context, req ScoreRequest) (ScoreResult, error) {
	f.mu.Lock()
	f.scored++
	f.mu.Unlock()
	f.t.Errorf("oneshot mode called Score for %q", req.Statement)
	return ScoreResult{Probability: 0.5}, nil
}

func (f *fakeGenerator) Generate(_ context.Context, req GenerateRequest) (GenerateResult, error) {
	f.mu.Lock()
	f.seen = append(f.seen, req)
	f.mu.Unlock()
	if f.err != nil {
		return GenerateResult{}, f.err
	}
	return GenerateResult{
		Content:      f.content,
		InputTokens:  f.inputTokens,
		OutputTokens: f.outputTokens,
	}, nil
}

func (f *fakeGenerator) lastRequest(t *testing.T) GenerateRequest {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.seen) != 1 {
		t.Fatalf("the generator was called %d times, want exactly 1", len(f.seen))
	}
	return f.seen[0]
}

// The oneshot question set the expected prompts below were written against:
// one question of each type, and a "boolean" that has to survive into the
// prompt as "boolean".
const oneshotQuestions = `{
	"department": {
		"type": "choice",
		"instructions": "Which team should handle this?",
		"criteria": {"billing": "Charges, refunds", "technical": "Bugs"}
	},
	"urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Low", "Medium", "High"]},
	"angry": {"type": "boolean", "instructions": "Is the customer angry?"}
}`

// Both prompts below were written out by hand from the shape the prompt is
// specified to have, not captured from the builder, so they can catch the
// builder drifting. Note the "angry" line: the type shown is the one the
// document declared, so a question written as "boolean" is described to the
// model as boolean even though its answer comes back as a noul.
const wantOneshotSystem = "You are a precise decision engine. Answer every question based only on the provided state. " +
	"Return probabilities that reflect genuine uncertainty. Do not invent information."

const wantOneshotUser = "STATE:\nCharged twice\n\nQUESTIONS:\n" +
	"- department (choice): Which team should handle this?\n" +
	"  Options → billing: Charges, refunds, technical: Bugs\n" +
	"- urgency (score): How urgent is this?\n" +
	"  Levels → 0=Low | 1=Medium | 2=High\n" +
	"- angry (boolean): Is the customer angry?\n\n" +
	"Respond with JSON matching this shape:\n" +
	"For each choice question: { \"choice\": \"<key>\", \"confidence\": 0-1, \"probabilities\": { \"<key>\": 0-1, ... } }\n" +
	"For each score question: { \"score\": <float>, \"confidence\": 0-1 }\n" +
	"For each noul question: { \"noul\": 0-1 }\n" +
	"Top-level keys must be the question names."

func oneshotRequest(t *testing.T) Request {
	t.Helper()
	return Request{
		State:       StringState("Charged twice"),
		Questions:   mustQuestions(t, oneshotQuestions),
		Model:       "probe-1",
		Temperature: 0.4,
		Mode:        ModeOneshot,
	}
}

func TestEvaluateOneshotGolden(t *testing.T) {
	const content = `{
		"department": {"choice": "technical", "confidence": 0.8, "probabilities": {"technical": 0.75}},
		"urgency": {"score": 2.5, "confidence": 0.6},
		"angry": {"probability": 0.9}
	}`

	// The missing declared option is appended at zero after the keys the model
	// did report, and a score answer carries a legend but no distribution.
	const want = `{"model":"probe-1","answers":{` +
		`"department":{"type":"choice","choice":"technical","confidence":0.8,` +
		`"probabilities":{"technical":0.75,"billing":0}},` +
		`"urgency":{"type":"score","score":2.5,"confidence":0.6,` +
		`"legend":{"0":"Low","1":"Medium","2":"High"}},` +
		`"angry":{"type":"noul","noul":0.9}},` +
		`"usage":{"input_tokens":120,"output_tokens":45},` +
		`"meta":{"mode":"oneshot","latency_ms":250,"parallel_calls":1}}`

	generator := &fakeGenerator{t: t, content: content, inputTokens: 120, outputTokens: 45}
	e := New(generator,
		WithClock(stepClock(time.Date(2026, 9, 20, 12, 0, 0, 0, time.UTC), 250*time.Millisecond)))

	resp, err := e.Evaluate(context.Background(), oneshotRequest(t))
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}

	encoded, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshalling the response: %v", err)
	}
	if string(encoded) != want {
		t.Errorf("response JSON differs from the golden document:\n got %s\nwant %s", encoded, want)
	}

	req := generator.lastRequest(t)
	if req.System != wantOneshotSystem {
		t.Errorf("system prompt:\n got %q\nwant %q", req.System, wantOneshotSystem)
	}
	if req.User != wantOneshotUser {
		t.Errorf("user prompt:\n got %q\nwant %q", req.User, wantOneshotUser)
	}
	if !req.JSONObject {
		t.Error("the one-shot call did not ask for a JSON object")
	}
	if req.Model != "probe-1" {
		t.Errorf("Model = %q, want %q", req.Model, "probe-1")
	}
	if req.Temperature != 0.4 {
		t.Errorf("Temperature = %v, want 0.4", req.Temperature)
	}
}

func TestEvaluateOneshotDefaults(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    string
	}{
		{
			name:    "no answers at all",
			content: `{}`,
			want: `{"department":{"type":"choice","choice":"billing","confidence":0.5,` +
				`"probabilities":{"billing":0.5,"technical":0}},` +
				`"urgency":{"type":"score","score":0,"confidence":0.5,` +
				`"legend":{"0":"Low","1":"Medium","2":"High"}},` +
				`"angry":{"type":"noul","noul":0.5}}`,
		},
		{
			name:    "empty content stands in for no reply",
			content: "",
			want: `{"department":{"type":"choice","choice":"billing","confidence":0.5,` +
				`"probabilities":{"billing":0.5,"technical":0}},` +
				`"urgency":{"type":"score","score":0,"confidence":0.5,` +
				`"legend":{"0":"Low","1":"Medium","2":"High"}},` +
				`"angry":{"type":"noul","noul":0.5}}`,
		},
		{
			name:    "valid JSON that is not an object answers nothing",
			content: `["department", "urgency"]`,
			want: `{"department":{"type":"choice","choice":"billing","confidence":0.5,` +
				`"probabilities":{"billing":0.5,"technical":0}},` +
				`"urgency":{"type":"score","score":0,"confidence":0.5,` +
				`"legend":{"0":"Low","1":"Medium","2":"High"}},` +
				`"angry":{"type":"noul","noul":0.5}}`,
		},
		{
			name:    "one question answered, the rest defaulted",
			content: `{"angry": {"noul": 0.1}}`,
			want: `{"department":{"type":"choice","choice":"billing","confidence":0.5,` +
				`"probabilities":{"billing":0.5,"technical":0}},` +
				`"urgency":{"type":"score","score":0,"confidence":0.5,` +
				`"legend":{"0":"Low","1":"Medium","2":"High"}},` +
				`"angry":{"type":"noul","noul":0.1}}`,
		},
		{
			name: "a choice with a confidence but no distribution",
			content: `{"department": {"choice": "technical", "confidence": 0.7},
				"urgency": {"score": 1}, "angry": {"noul": 0}}`,
			want: `{"department":{"type":"choice","choice":"technical","confidence":0.7,` +
				`"probabilities":{"technical":0.7,"billing":0}},` +
				`"urgency":{"type":"score","score":1,"confidence":0.5,` +
				`"legend":{"0":"Low","1":"Medium","2":"High"}},` +
				`"angry":{"type":"noul","noul":0}}`,
		},
		{
			name: "a distribution the model reported is kept, in its own order",
			content: `{"department": {"choice": "billing", "confidence": 0.55,
				"probabilities": {"technical": 0.45, "billing": 0.55}}}`,
			want: `{"department":{"type":"choice","choice":"billing","confidence":0.55,` +
				`"probabilities":{"technical":0.45,"billing":0.55}},` +
				`"urgency":{"type":"score","score":0,"confidence":0.5,` +
				`"legend":{"0":"Low","1":"Medium","2":"High"}},` +
				`"angry":{"type":"noul","noul":0.5}}`,
		},
		{
			name:    "noul prefers noul over probability",
			content: `{"angry": {"noul": 0.2, "probability": 0.9}}`,
			want: `{"department":{"type":"choice","choice":"billing","confidence":0.5,` +
				`"probabilities":{"billing":0.5,"technical":0}},` +
				`"urgency":{"type":"score","score":0,"confidence":0.5,` +
				`"legend":{"0":"Low","1":"Medium","2":"High"}},` +
				`"angry":{"type":"noul","noul":0.2}}`,
		},
		{
			name:    "a question answered with something that is not an object",
			content: `{"angry": "very"}`,
			want: `{"department":{"type":"choice","choice":"billing","confidence":0.5,` +
				`"probabilities":{"billing":0.5,"technical":0}},` +
				`"urgency":{"type":"score","score":0,"confidence":0.5,` +
				`"legend":{"0":"Low","1":"Medium","2":"High"}},` +
				`"angry":{"type":"noul","noul":0.5}}`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			generator := &fakeGenerator{t: t, content: tc.content}
			resp, err := New(generator).Evaluate(context.Background(), oneshotRequest(t))
			if err != nil {
				t.Fatalf("Evaluate returned %v", err)
			}
			encoded, err := json.Marshal(resp.Answers)
			if err != nil {
				t.Fatalf("marshalling the answers: %v", err)
			}
			if string(encoded) != tc.want {
				t.Fatalf("answers:\n got %s\nwant %s", encoded, tc.want)
			}
			if resp.Meta.ParallelCalls != 1 {
				t.Errorf("ParallelCalls = %d, want 1", resp.Meta.ParallelCalls)
			}
			if resp.Meta.Mode != ModeOneshot {
				t.Errorf("Meta.Mode = %q, want %q", resp.Meta.Mode, ModeOneshot)
			}
		})
	}
}

func TestEvaluateOneshotNonJSON(t *testing.T) {
	const content = "I'm sorry, I can't answer that."

	generator := &fakeGenerator{t: t, content: content}
	_, err := New(generator).Evaluate(context.Background(), oneshotRequest(t))
	if err == nil {
		t.Fatal("Evaluate accepted a non-JSON reply")
	}
	if !strings.Contains(err.Error(), "non-JSON") {
		t.Errorf("error %q should say the reply was not JSON", err)
	}
	if !strings.Contains(err.Error(), content) {
		t.Errorf("error %q should quote the reply", err)
	}
}

func TestEvaluateOneshotNonJSONIsTruncated(t *testing.T) {
	content := strings.Repeat("é", 500) // multi-byte, so a byte cut would split a rune

	generator := &fakeGenerator{t: t, content: content}
	_, err := New(generator).Evaluate(context.Background(), oneshotRequest(t))
	if err == nil {
		t.Fatal("Evaluate accepted a non-JSON reply")
	}
	quoted := strings.TrimPrefix(err.Error(), "classifier: one-shot model returned non-JSON: ")
	if got := len([]rune(quoted)); got != 200 {
		t.Fatalf("the error quoted %d characters of the reply, want 200", got)
	}
	if strings.ContainsRune(quoted, '�') {
		t.Error("the quoted reply was cut in the middle of a character")
	}
}

func TestEvaluateOneshotNeedsAGenerator(t *testing.T) {
	scoreOnly := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		return ScoreResult{Probability: 0.5}, nil
	})

	_, err := New(scoreOnly).Evaluate(context.Background(), oneshotRequest(t))
	if !errors.Is(err, ErrOneshotUnsupported) {
		t.Fatalf("Evaluate returned %v, want ErrOneshotUnsupported", err)
	}
}

func TestEvaluateOneshotPropagatesGeneratorErrors(t *testing.T) {
	wantErr := errors.New("provider is down")
	generator := &fakeGenerator{t: t, err: wantErr}

	_, err := New(generator).Evaluate(context.Background(), oneshotRequest(t))
	if !errors.Is(err, wantErr) {
		t.Fatalf("Evaluate returned %v, want the generator's error", err)
	}
}

// TestEvaluateOneshotValidatesFirst checks a bad request never reaches the
// model, in oneshot mode as much as in parallel mode.
func TestEvaluateOneshotValidatesFirst(t *testing.T) {
	generator := &fakeGenerator{t: t, content: `{}`}

	_, err := New(generator).Evaluate(context.Background(), Request{
		State:     StringState("s"),
		Questions: one("urgency", levelsOf(1)),
		Mode:      ModeOneshot,
	})
	var verr *ValidationError
	if !errors.As(err, &verr) {
		t.Fatalf("Evaluate returned %v, want a *ValidationError", err)
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.seen) != 0 {
		t.Fatalf("an invalid request still reached the model %d times", len(generator.seen))
	}
}

// TestOneshotQuestionBlock pins the question description block on its own,
// including a question with no instruction and one with no options listed.
// The expected block is spelled out in full rather than assembled from the
// pieces the builder itself uses.
func TestOneshotQuestionBlock(t *testing.T) {
	qs := mustQuestions(t, `{
		"department": {"type": "choice", "criteria": {"billing": "Charges", "technical": "Bugs"}},
		"urgency": {"type": "score", "criteria": ["Low", "High"]},
		"angry": {"type": "boolean"}
	}`)

	want := "- department (choice): \n  Options → billing: Charges, technical: Bugs\n" +
		"- urgency (score): \n  Levels → 0=Low | 1=High\n" +
		"- angry (boolean): "

	if got := oneshotQuestionBlock(qs); got != want {
		t.Fatalf("question block:\n got %q\nwant %q", got, want)
	}
}

// TestOneshotQuestionBlockUsesTheDeclaredType covers both directions at once:
// a question written as "boolean" is described as boolean, and one written as
// "noul" is described as noul. A block built from the normalised type would
// fail the first; one built from a hardcoded synonym would fail the second.
func TestOneshotQuestionBlockUsesTheDeclaredType(t *testing.T) {
	for declared, want := range map[string]string{
		"boolean": "- angry (boolean): Is the customer angry?",
		"noul":    "- angry (noul): Is the customer angry?",
	} {
		qs := mustQuestions(t, `{"angry": {"type": "`+declared+`", "instructions": "Is the customer angry?"}}`)
		if got := oneshotQuestionBlock(qs); got != want {
			t.Errorf("a question declared %q rendered as %q, want %q", declared, got, want)
		}
	}

	// A question built in Go declares nothing, so it falls back to its type.
	programmatic := one("angry", Question{Type: TypeNoul, Instructions: "Is the customer angry?"})
	if got, want := oneshotQuestionBlock(programmatic), "- angry (noul): Is the customer angry?"; got != want {
		t.Errorf("a programmatic question rendered as %q, want %q", got, want)
	}
}

// TestOneshotAnswerForDoesNotMutateTheReply checks the reported distribution
// is filled in on a copy. Writing through an assignment copy of an OrderedMap
// reaches the original's map without appending to its key slice, which leaves
// the source able to Get a key that Keys does not list.
func TestOneshotAnswerForDoesNotMutateTheReply(t *testing.T) {
	reported := NewOrderedMap[float64]()
	reported.Set("technical", 0.75)

	q := Question{Type: TypeChoice, Options: *func() *OrderedMap[string] {
		o := NewOrderedMap[string]()
		o.Set("billing", "Charges")
		o.Set("technical", "Bugs")
		return o
	}()}

	answer := oneshotAnswerFor(q, oneshotAnswer{Probabilities: reported})

	if got := reported.Keys(); !equalStrings(got, []string{"technical"}) {
		t.Errorf("the reply's own keys are now %v, want [technical]", got)
	}
	if _, ok := reported.Get("billing"); ok {
		t.Error("filling in the missing option wrote through into the reply's own map")
	}
	if got := answer.Probabilities.Keys(); !equalStrings(got, []string{"technical", "billing"}) {
		t.Errorf("answer probabilities = %v, want the reported key then the missing one", got)
	}
}

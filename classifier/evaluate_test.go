package classifier

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ---------------------------------------------------------------- helpers --

// mustQuestions decodes a questions document, which is also how a real request
// arrives, so declaration order comes from the document itself.
func mustQuestions(t *testing.T, doc string) Questions {
	t.Helper()
	var qs Questions
	if err := json.Unmarshal([]byte(doc), &qs); err != nil {
		t.Fatalf("decoding questions: %v", err)
	}
	return qs
}

// noulQuestions builds n noul questions named q0…q(n-1).
func noulQuestions(n int) Questions {
	qs := NewOrderedMap[Question]()
	for i := range n {
		qs.Set(fmt.Sprintf("q%d", i), Question{Type: TypeNoul})
	}
	return *qs
}

// stepClock returns a clock whose first reading is start and whose every later
// reading is one step further on, so a latency is exactly one step.
func stepClock(start time.Time, step time.Duration) func() time.Time {
	var mu sync.Mutex
	reads := 0
	return func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		t := start.Add(time.Duration(reads) * step)
		reads++
		return t
	}
}

// scriptedScorer answers a fixed statement-to-probability script and records
// every request it was given. An unscripted statement fails the test.
type scriptedScorer struct {
	t            *testing.T
	script       map[string]float64
	inputTokens  int
	outputTokens int

	mu   sync.Mutex
	seen []ScoreRequest
}

func (s *scriptedScorer) Score(_ context.Context, req ScoreRequest) (ScoreResult, error) {
	s.mu.Lock()
	s.seen = append(s.seen, req)
	s.mu.Unlock()

	p, ok := s.script[req.Statement]
	if !ok {
		s.t.Errorf("scorer was asked an unscripted statement: %q", req.Statement)
		return ScoreResult{}, fmt.Errorf("unscripted statement %q", req.Statement)
	}
	return ScoreResult{Probability: p, InputTokens: s.inputTokens, OutputTokens: s.outputTokens}, nil
}

func (s *scriptedScorer) requests() []ScoreRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]ScoreRequest, len(s.seen))
	copy(out, s.seen)
	return out
}

// constantScorer answers every statement with the same probability and counts
// the calls.
type constantScorer struct {
	probability float64
	mu          sync.Mutex
	calls       int
}

func (s *constantScorer) Score(_ context.Context, _ ScoreRequest) (ScoreResult, error) {
	s.mu.Lock()
	s.calls++
	s.mu.Unlock()
	return ScoreResult{Probability: s.probability}, nil
}

func (s *constantScorer) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

// ------------------------------------------------------------------ tests --

// The golden document below pins ordering, rounding and field selection in one
// assertion. Every probability, score and confidence in it was computed from
// the formulas by hand over the same scripted inputs, not printed from this
// implementation's output — a golden recorded from the code under test would
// agree with that code however wrong it was. The working:
//
//	scoresToDistribution([0.62, 0.55, 0.30], 1) → [0.497073, 0.372359, 0.130568]
//	                                              at 6dp, confidence 0.560893
//	scoresToDistribution([0.20, 0.55, 0.50], 1) → [0.101124, 0.494382, 0.404494]
//	                                              at 6dp,
//	                                              score 1.3033707865168538 → 1.3,
//	                                              confidence 0.542135
//	round6(0.7123) → 0.7123
//
// The probabilities and confidences were three decimals until reporting moved
// to six; the unrounded distributions behind them are unchanged, because the
// softmax temperature this runs at is 1 and dividing by 1 is the identity. The
// score keeps two places: a score is a position on a declared scale, not a
// probability, and nothing about it can be mistaken for a certainty.
const goldenQuestions = `{
	"department": {
		"type": "choice",
		"instructions": "Which team should handle this?",
		"criteria": {
			"billing": "Charges, refunds, invoices",
			"technical": "Bugs or product issues",
			"other": "Doesn't fit above"
		}
	},
	"urgency": {
		"type": "score",
		"instructions": "How urgent is this?",
		"criteria": ["Low", "Medium", "High"]
	},
	"angry": {
		"type": "boolean",
		"instructions": "Is the customer expressing strong frustration or anger?"
	}
}`

var goldenScript = map[string]float64{
	`The correct which team should handle this is "billing" (Charges, refunds, invoices).`:      0.62,
	`The correct which team should handle this is "technical" (Bugs or product issues).`:        0.55,
	`The correct which team should handle this is "other" (Doesn't fit above).`:                 0.30,
	`On the scale for "How urgent is this?", the most appropriate rating is level 0: "Low".`:    0.20,
	`On the scale for "How urgent is this?", the most appropriate rating is level 1: "Medium".`: 0.55,
	`On the scale for "How urgent is this?", the most appropriate rating is level 2: "High".`:   0.50,
	`Is the customer expressing strong frustration or anger?`:                                   0.7123,
}

const goldenResponse = `{"model":"probe-1","answers":{` +
	`"department":{"type":"choice","choice":"billing","confidence":0.560893,` +
	`"probabilities":{"billing":0.497073,"technical":0.372359,"other":0.130568}},` +
	`"urgency":{"type":"score","score":1.3,"confidence":0.542135,` +
	`"legend":{"0":"Low","1":"Medium","2":"High"},` +
	`"probabilities":{"0":0.101124,"1":0.494382,"2":0.404494}},` +
	`"angry":{"type":"noul","noul":0.7123}},` +
	`"usage":{"input_tokens":21,"output_tokens":7},` +
	`"meta":{"mode":"parallel","latency_ms":250,"parallel_calls":7}}`

func TestEvaluateParallelGolden(t *testing.T) {
	scorer := &scriptedScorer{t: t, script: goldenScript, inputTokens: 3, outputTokens: 1}
	e := New(scorer,
		WithClock(stepClock(time.Date(2026, 9, 20, 12, 0, 0, 0, time.UTC), 250*time.Millisecond)))

	req := Request{
		State:       StringState("Charged twice again!! Billed twice for the Pro plan."),
		Questions:   mustQuestions(t, goldenQuestions),
		Model:       "probe-1",
		Temperature: 0.25,
	}

	resp, err := e.Evaluate(context.Background(), req)
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}

	encoded, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshalling the response: %v", err)
	}
	if string(encoded) != goldenResponse {
		t.Errorf("response JSON differs from the golden document:\n got %s\nwant %s", encoded, goldenResponse)
	}

	// Every call has to carry the request's own model, state and temperature.
	requests := scorer.requests()
	if len(requests) != 7 {
		t.Fatalf("scorer saw %d calls, want 7", len(requests))
	}
	for _, r := range requests {
		if r.Model != "probe-1" {
			t.Errorf("call for %q carried model %q, want %q", r.Statement, r.Model, "probe-1")
		}
		if r.Temperature != 0.25 {
			t.Errorf("call for %q carried temperature %v, want 0.25", r.Statement, r.Temperature)
		}
		if r.State != "Charged twice again!! Billed twice for the Pro plan." {
			t.Errorf("call for %q carried state %q", r.Statement, r.State)
		}
	}
}

// TestEvaluateRendersAJSONState checks the state reaches the scorer the way
// State.Text renders it, with its declared key order intact.
func TestEvaluateRendersAJSONState(t *testing.T) {
	var state State
	if err := json.Unmarshal([]byte(`{"subject":"Charged twice","attempts":2}`), &state); err != nil {
		t.Fatalf("decoding state: %v", err)
	}
	// The state re-indented with two spaces, its declared key order intact.
	const want = "{\n  \"subject\": \"Charged twice\",\n  \"attempts\": 2\n}"

	seen := make(chan string, 1)
	recording := ScorerFunc(func(_ context.Context, req ScoreRequest) (ScoreResult, error) {
		seen <- req.State
		return ScoreResult{Probability: 0.5}, nil
	})

	if _, err := New(recording).Evaluate(context.Background(), Request{
		State:     state,
		Questions: noulQuestions(1),
	}); err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}
	if got := <-seen; got != want {
		t.Fatalf("scorer saw state %q, want %q", got, want)
	}
}

// TestEvaluateChoiceTieGoesToTheFirstKey pins the tie-break: two equally
// likely options resolve to the one declared first, so the same request always
// returns the same option key.
func TestEvaluateChoiceTieGoesToTheFirstKey(t *testing.T) {
	// scoresToDistribution([0.6, 0.6, 0.2], 1) → [0.461538, 0.461538, 0.076923]
	// at 6dp, confidence 0.480769. The first two are equal before rounding as
	// well as after — the scores are identical — so this is a real tie and not
	// one the reporting invented.
	script := map[string]float64{
		`The correct select the best label for "department" is "zeta".`:  0.6,
		`The correct select the best label for "department" is "alpha".`: 0.6,
		`The correct select the best label for "department" is "omega".`: 0.2,
	}
	scorer := &scriptedScorer{t: t, script: script}

	resp, err := New(scorer).Evaluate(context.Background(), Request{
		State: StringState("anything"),
		Questions: mustQuestions(t, `{
			"department": {"type": "choice", "criteria": {"zeta": "", "alpha": "", "omega": ""}}
		}`),
	})
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}

	answer, _ := resp.Answers.Get("department")
	if answer.Choice != "zeta" {
		t.Errorf("choice = %q, want the first declared key of the tie, %q", answer.Choice, "zeta")
	}
	if answer.Confidence != 0.480769 {
		t.Errorf("confidence = %v, want 0.480769", answer.Confidence)
	}
	wantProbs := map[string]float64{"zeta": 0.461538, "alpha": 0.461538, "omega": 0.076923}
	for key, want := range wantProbs {
		if got, _ := answer.Probabilities.Get(key); got != want {
			t.Errorf("probability of %q = %v, want %v", key, got, want)
		}
	}
	if keys := answer.Probabilities.Keys(); strings.Join(keys, ",") != "zeta,alpha,omega" {
		t.Errorf("probability keys = %v, want the declared order", keys)
	}
}

// TestEvaluateChoiceArgmaxUsesTheUnroundedDistribution pins that the winner is
// picked before the reported probabilities are rounded.
//
// The scores are a counterexample rebuilt for six decimals. The reviewer's
// original one separated the two leaders by 4.1e-4, which three decimals
// flattened into a tie and six report apart, so at the new precision it no
// longer distinguishes anything: it passes whether the argmax is taken before
// or after rounding. These scores put the leaders 4.5e-8 apart instead.
//
// scoresToDistribution([0.5744669624608281, 0.23147521650098238,
// 0.574466986906295], 1) is [0.44982071193981865, 0.10035853113828926,
// 0.44982075692189205], which is [0.449821, 0.100359, 0.449821] at six
// places. The third option wins; on the rounded numbers the first and third
// tie and the tie goes to the first.
func TestEvaluateChoiceArgmaxUsesTheUnroundedDistribution(t *testing.T) {
	script := map[string]float64{
		`The correct select the best label for "pick" is "first".`:  0.5744669624608281,
		`The correct select the best label for "pick" is "second".`: 0.23147521650098238,
		`The correct select the best label for "pick" is "third".`:  0.574466986906295,
	}

	resp, err := New(&scriptedScorer{t: t, script: script}).Evaluate(context.Background(), Request{
		State:     StringState("s"),
		Questions: mustQuestions(t, `{"pick": {"type": "choice", "criteria": {"first": "", "second": "", "third": ""}}}`),
	})
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}

	answer, _ := resp.Answers.Get("pick")
	if answer.Choice != "third" {
		t.Errorf("choice = %q, want %q: the winner is decided before rounding", answer.Choice, "third")
	}
	for key, want := range map[string]float64{"first": 0.449821, "second": 0.100359, "third": 0.449821} {
		if got, _ := answer.Probabilities.Get(key); got != want {
			t.Errorf("probability of %q = %v, want %v", key, got, want)
		}
	}
}

// TestEvaluateConfidenceUsesTheUnroundedDistribution is the same guard for the
// other consumer of the distribution, and its scores were rebuilt for six
// decimals for the same reason: the old triple's two confidences agree once
// the distribution is reported to six places, so it could no longer tell the
// two orders of operations apart.
//
// scoresToDistribution([0.72500192, 0.4072496, 0.73644896], 1) is
// [0.43093946024142732, 0.11230409152642162, 0.45675644823215111], whose
// confidence is 0.491287. Recomputed from the reported
// [0.430939, 0.112304, 0.456756] it is 0.491286.
func TestEvaluateConfidenceUsesTheUnroundedDistribution(t *testing.T) {
	script := map[string]float64{
		`The correct select the best label for "pick" is "first".`:  0.72500192,
		`The correct select the best label for "pick" is "second".`: 0.4072496,
		`The correct select the best label for "pick" is "third".`:  0.73644896,
	}

	resp, err := New(&scriptedScorer{t: t, script: script}).Evaluate(context.Background(), Request{
		State:     StringState("s"),
		Questions: mustQuestions(t, `{"pick": {"type": "choice", "criteria": {"first": "", "second": "", "third": ""}}}`),
	})
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}

	answer, _ := resp.Answers.Get("pick")
	if answer.Confidence != 0.491287 {
		t.Errorf("confidence = %v, want 0.491287: 0.491286 is what the rounded distribution gives",
			answer.Confidence)
	}
	for key, want := range map[string]float64{"first": 0.430939, "second": 0.112304, "third": 0.456756} {
		if got, _ := answer.Probabilities.Get(key); got != want {
			t.Errorf("probability of %q = %v, want %v", key, got, want)
		}
	}
}

// TestEvaluateSanitisesScorerProbabilities checks a Scorer cannot put a value
// outside [0,1] into an answer, and cannot make the response unencodable. A
// Scorer is an interface, so its answer is input like any other, and the
// clamp on the way in is the only thing holding the range.
func TestEvaluateSanitisesScorerProbabilities(t *testing.T) {
	const questions = `{
		"angry": {"type": "noul"},
		"department": {"type": "choice", "criteria": {"x": "", "y": ""}},
		"urgency": {"type": "score", "criteria": ["Low", "High"]}
	}`

	for _, tc := range []struct {
		name        string
		probability float64
		wantNoul    float64
	}{
		{name: "above one", probability: 1.5, wantNoul: 1},
		{name: "far above one", probability: 12, wantNoul: 1},
		{name: "below zero", probability: -0.2, wantNoul: 0},
		{name: "positive infinity", probability: math.Inf(1), wantNoul: 1},
		{name: "negative infinity", probability: math.Inf(-1), wantNoul: 0},
		{name: "not a number", probability: math.NaN(), wantNoul: 0.5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			scorer := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
				return ScoreResult{Probability: tc.probability}, nil
			})
			resp, err := New(scorer).Evaluate(context.Background(), Request{
				State:     StringState("s"),
				Questions: mustQuestions(t, questions),
			})
			if err != nil {
				t.Fatalf("Evaluate returned %v", err)
			}

			// A NaN reaches here as a NaN probability and a NaN confidence,
			// and json.Marshal refuses both: without the sanitising step the
			// caller gets an encoding error rather than an answer.
			encoded, err := json.Marshal(resp)
			if err != nil {
				t.Fatalf("marshalling the response: %v", err)
			}
			if strings.Contains(string(encoded), "NaN") || strings.Contains(string(encoded), "Inf") {
				t.Fatalf("the response carries a non-finite number: %s", encoded)
			}

			angry, _ := resp.Answers.Get("angry")
			if angry.Noul != tc.wantNoul {
				t.Errorf("noul = %v, want %v", angry.Noul, tc.wantNoul)
			}
			for _, name := range resp.Answers.Keys() {
				answer, _ := resp.Answers.Get(name)
				for key, p := range answer.Probabilities.All() {
					if p < 0 || p > 1 || math.IsNaN(p) {
						t.Errorf("%s probability %q = %v, outside [0,1]", name, key, p)
					}
				}
				if c := answer.Confidence; c < 0 || c > 1 || math.IsNaN(c) {
					t.Errorf("%s confidence = %v, outside [0,1]", name, c)
				}
			}
		})
	}
}

func TestEvaluateUsageIsNullWhenNothingWasCounted(t *testing.T) {
	tests := []struct {
		name         string
		inputTokens  int
		outputTokens int
		want         string
	}{
		{name: "nothing reported", want: `{"input_tokens":null,"output_tokens":null}`},
		{name: "both reported", inputTokens: 4, outputTokens: 2, want: `{"input_tokens":8,"output_tokens":4}`},
		{name: "only output reported", outputTokens: 3, want: `{"input_tokens":null,"output_tokens":6}`},
		{name: "only input reported", inputTokens: 5, want: `{"input_tokens":10,"output_tokens":null}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			scorer := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
				return ScoreResult{Probability: 0.5, InputTokens: tc.inputTokens, OutputTokens: tc.outputTokens}, nil
			})
			resp, err := New(scorer).Evaluate(context.Background(), Request{
				State:     StringState("s"),
				Questions: noulQuestions(2), // two calls, so the sum is visible
			})
			if err != nil {
				t.Fatalf("Evaluate returned %v", err)
			}
			encoded, err := json.Marshal(resp.Usage)
			if err != nil {
				t.Fatalf("marshalling usage: %v", err)
			}
			if string(encoded) != tc.want {
				t.Fatalf("usage = %s, want %s", encoded, tc.want)
			}
		})
	}
}

func TestEvaluateParallelCallsCountsEveryCandidate(t *testing.T) {
	tests := []struct {
		name      string
		questions string
		want      int
	}{
		{name: "one noul is one call", questions: `{"a": {"type": "noul"}}`, want: 1},
		{
			name:      "a choice costs one call per option",
			questions: `{"a": {"type": "choice", "criteria": {"x": "", "y": "", "z": ""}}}`,
			want:      3,
		},
		{
			name:      "a score costs one call per level",
			questions: `{"a": {"type": "score", "criteria": ["p", "q", "r", "s"]}}`,
			want:      4,
		},
		{
			name: "the whole request adds up",
			questions: `{
				"a": {"type": "choice", "criteria": {"x": "", "y": ""}},
				"b": {"type": "score", "criteria": ["p", "q", "r"]},
				"c": {"type": "noul"},
				"d": {"type": "boolean"}
			}`,
			want: 7,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			scorer := &constantScorer{probability: 0.5}
			resp, err := New(scorer).Evaluate(context.Background(), Request{
				State:     StringState("s"),
				Questions: mustQuestions(t, tc.questions),
			})
			if err != nil {
				t.Fatalf("Evaluate returned %v", err)
			}
			if resp.Meta == nil {
				t.Fatal("response carried no meta")
			}
			if resp.Meta.ParallelCalls != tc.want {
				t.Errorf("ParallelCalls = %d, want %d", resp.Meta.ParallelCalls, tc.want)
			}
			if got := scorer.callCount(); got != tc.want {
				t.Errorf("the scorer was called %d times, want %d", got, tc.want)
			}
			if resp.Meta.Mode != ModeParallel {
				t.Errorf("Meta.Mode = %q, want %q", resp.Meta.Mode, ModeParallel)
			}
		})
	}
}

// TestEvaluateAnswersFollowQuestionOrder checks the answers come back in the
// order the questions were declared, not in map order and not in the order the
// concurrent calls happened to finish.
func TestEvaluateAnswersFollowQuestionOrder(t *testing.T) {
	const doc = `{
		"zulu": {"type": "noul"},
		"alpha": {"type": "choice", "criteria": {"x": "", "y": ""}},
		"mike": {"type": "score", "criteria": ["p", "q"]},
		"bravo": {"type": "noul"}
	}`
	want := []string{"zulu", "alpha", "mike", "bravo"}

	// Finishing out of order: later questions answer first.
	scorer := ScorerFunc(func(_ context.Context, req ScoreRequest) (ScoreResult, error) {
		if strings.Contains(req.Statement, "zulu") {
			time.Sleep(20 * time.Millisecond)
		}
		return ScoreResult{Probability: 0.5}, nil
	})

	resp, err := New(scorer).Evaluate(context.Background(), Request{
		State:     StringState("s"),
		Questions: mustQuestions(t, doc),
	})
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}
	if got := resp.Answers.Keys(); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("answers came back as %v, want %v", got, want)
	}
}

// TestEvaluateBoundsConcurrency counts how many scorer calls are in flight at
// once and checks the semaphore never lets more than the configured limit
// through.
func TestEvaluateBoundsConcurrency(t *testing.T) {
	const limit = 3
	const total = 24

	var (
		mu          sync.Mutex
		inFlight    int
		maxInFlight int
	)
	scorer := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		mu.Lock()
		inFlight++
		if inFlight > maxInFlight {
			maxInFlight = inFlight
		}
		mu.Unlock()

		// Long enough that everything admitted overlaps: without a bound all
		// 24 calls would be counted in flight together.
		time.Sleep(10 * time.Millisecond)

		mu.Lock()
		inFlight--
		mu.Unlock()
		return ScoreResult{Probability: 0.5}, nil
	})

	resp, err := New(scorer, WithMaxConcurrency(limit)).Evaluate(context.Background(), Request{
		State:     StringState("s"),
		Questions: noulQuestions(total),
	})
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}
	if resp.Meta.ParallelCalls != total {
		t.Fatalf("ParallelCalls = %d, want %d", resp.Meta.ParallelCalls, total)
	}

	mu.Lock()
	observed := maxInFlight
	mu.Unlock()
	if observed > limit {
		t.Fatalf("saw %d scorer calls in flight at once, want at most %d", observed, limit)
	}
	if observed < 2 {
		t.Fatalf("saw at most %d call in flight; the wave is not running concurrently at all", observed)
	}
}

// TestEvaluateUnlimitedConcurrency uses a rendezvous no bounded wave could get
// through: every call blocks until all of them have arrived. If any limit were
// applied, the calls would time out and the evaluation would fail.
func TestEvaluateUnlimitedConcurrency(t *testing.T) {
	const total = 12

	var mu sync.Mutex
	arrived := 0
	everyoneHere := make(chan struct{})

	scorer := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		mu.Lock()
		arrived++
		if arrived == total {
			close(everyoneHere)
		}
		mu.Unlock()

		select {
		case <-everyoneHere:
			return ScoreResult{Probability: 0.5}, nil
		case <-time.After(5 * time.Second):
			return ScoreResult{}, errors.New("the wave never had every call in flight at once")
		}
	})

	for _, limit := range []int{0, -1} {
		t.Run(fmt.Sprintf("limit %d", limit), func(t *testing.T) {
			mu.Lock()
			arrived = 0
			everyoneHere = make(chan struct{})
			mu.Unlock()

			_, err := New(scorer, WithMaxConcurrency(limit)).Evaluate(context.Background(), Request{
				State:     StringState("s"),
				Questions: noulQuestions(total),
			})
			if err != nil {
				t.Fatalf("Evaluate returned %v", err)
			}
		})
	}
}

// TestEvaluateDefaultConcurrency documents the default bound, which is what
// the undecorated constructor applies.
func TestEvaluateDefaultConcurrency(t *testing.T) {
	var (
		mu          sync.Mutex
		inFlight    int
		maxInFlight int
	)
	scorer := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		mu.Lock()
		inFlight++
		if inFlight > maxInFlight {
			maxInFlight = inFlight
		}
		mu.Unlock()
		time.Sleep(10 * time.Millisecond)
		mu.Lock()
		inFlight--
		mu.Unlock()
		return ScoreResult{Probability: 0.5}, nil
	})

	if _, err := New(scorer).Evaluate(context.Background(), Request{
		State:     StringState("s"),
		Questions: noulQuestions(DefaultMaxConcurrency * 3),
	}); err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}
	mu.Lock()
	observed := maxInFlight
	mu.Unlock()
	if observed > DefaultMaxConcurrency {
		t.Fatalf("saw %d calls in flight at once, want at most the default %d", observed, DefaultMaxConcurrency)
	}
}

// TestEvaluateFirstErrorWins checks the all-or-nothing rule: the first failure
// is returned and the rest of the wave is abandoned rather than issued.
func TestEvaluateFirstErrorWins(t *testing.T) {
	const total = 20
	scorerErr := errors.New("provider refused the request")

	var mu sync.Mutex
	calls := 0

	scorer := ScorerFunc(func(ctx context.Context, _ ScoreRequest) (ScoreResult, error) {
		mu.Lock()
		calls++
		first := calls == 1
		mu.Unlock()

		if first {
			return ScoreResult{}, scorerErr
		}
		// Anything already in flight waits to be cancelled, so the error the
		// evaluation reports can only be the first one.
		select {
		case <-ctx.Done():
			return ScoreResult{}, ctx.Err()
		case <-time.After(2 * time.Second):
			return ScoreResult{}, errors.New("was never cancelled")
		}
	})

	resp, err := New(scorer, WithMaxConcurrency(2)).Evaluate(context.Background(), Request{
		State:     StringState("s"),
		Questions: noulQuestions(total),
	})
	if !errors.Is(err, scorerErr) {
		t.Fatalf("Evaluate returned %v, want the scorer's own error", err)
	}
	if resp.Answers.Len() != 0 || resp.Meta != nil {
		t.Errorf("a failed evaluation returned a populated response: %+v", resp)
	}

	mu.Lock()
	issued := calls
	mu.Unlock()
	if issued >= total {
		t.Fatalf("the scorer was called %d times; the remaining work should have been cancelled", issued)
	}
}

func TestEvaluateHonoursContextCancellation(t *testing.T) {
	t.Run("cancelled before the call", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		scorer := &constantScorer{probability: 0.5}
		_, err := New(scorer).Evaluate(ctx, Request{
			State:     StringState("s"),
			Questions: noulQuestions(12),
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Evaluate returned %v, want context.Canceled", err)
		}
		if got := scorer.callCount(); got != 0 {
			t.Fatalf("the scorer was called %d times under a cancelled context, want 0", got)
		}
	})

	t.Run("cancelled mid-flight", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		var once sync.Once

		scorer := ScorerFunc(func(ctx context.Context, _ ScoreRequest) (ScoreResult, error) {
			once.Do(func() { go cancel() })
			select {
			case <-ctx.Done():
				return ScoreResult{}, ctx.Err()
			case <-time.After(5 * time.Second):
				return ScoreResult{}, errors.New("was never cancelled")
			}
		})
		defer cancel()

		_, err := New(scorer).Evaluate(ctx, Request{
			State:     StringState("s"),
			Questions: noulQuestions(4),
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Evaluate returned %v, want context.Canceled", err)
		}
	})
}

// A scorer error that names the fault is the point of the diagnosable
// sentinels — the one the provider package raises when it gets no logprobs,
// the one this package raises for a truncated one-shot reply — and the API
// layer turns both into a 502 whose message says which setting to change.
//
// The candidate that fails fails fast; its siblings take a moment to unwind an
// in-flight round trip or a retry backoff, and the caller's deadline can land
// in that window. The diagnosis is what the caller needs, and a deadline that
// arrived after the wave had already been abandoned for a better reason must
// not replace it — a 502 naming the setting would become a bare 504.
func TestEvaluateReportsTheScorerErrorAndNotTheDeadlineItRanInto(t *testing.T) {
	const total = 4
	refused := errors.New("the provider returned no token logprobs")

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	var arrived atomic.Int64
	inFlight := make(chan struct{}, total)

	scorer := ScorerFunc(func(callCtx context.Context, _ ScoreRequest) (ScoreResult, error) {
		if arrived.Add(1) == 1 {
			// Fail only once every sibling is inside the scorer, so none of
			// them can be turned away at the gate and every one of them is
			// still unwinding when the caller's deadline lands.
			for range total - 1 {
				<-inFlight
			}
			return ScoreResult{}, refused
		}
		// A call that does not abandon itself the instant the wave is
		// cancelled. The deadline is what ends it.
		inFlight <- struct{}{}
		<-ctx.Done()
		return ScoreResult{}, callCtx.Err()
	})

	_, err := New(scorer, WithMaxConcurrency(total)).Evaluate(ctx, Request{
		State:     StringState("s"),
		Questions: noulQuestions(total),
	})
	if !errors.Is(err, refused) {
		t.Fatalf("Evaluate returned %v, want the scorer's own error: the deadline landed while the wave unwound and must not stand in for the diagnosis", err)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Evaluate returned %v, which a caller classifies as a timeout rather than as the upstream fault it was", err)
	}
}

// An evaluation whose every call came back has an answer, and it was paid for.
// A cancellation that lands after the last one has nothing left to stop, and
// throwing the answers away for it discards finished work.
func TestEvaluateKeepsAnAnswerFinishedBeforeTheCancellation(t *testing.T) {
	const total = 4

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var done atomic.Int64
	scorer := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		// The last call hangs the caller up before returning its own result,
		// so the context is certainly expired by the time the wave is
		// collected, and every call has certainly succeeded.
		if done.Add(1) == total {
			cancel()
		}
		return ScoreResult{Probability: 0.5}, nil
	})

	resp, err := New(scorer, WithMaxConcurrency(total)).Evaluate(ctx, Request{
		State:     StringState("s"),
		Questions: noulQuestions(total),
	})
	if err != nil {
		t.Fatalf("Evaluate returned %v; every call had come back before the cancellation", err)
	}
	if got := resp.Answers.Len(); got != total {
		t.Fatalf("got %d answers, want %d", got, total)
	}
}

func TestEvaluateRejectsBadRequests(t *testing.T) {
	ok := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		return ScoreResult{Probability: 0.5}, nil
	})

	t.Run("no scorer", func(t *testing.T) {
		e := New(nil) // must not panic
		if e == nil {
			t.Fatal("New(nil) returned nil")
		}
		_, err := e.Evaluate(context.Background(), Request{
			State:     StringState("s"),
			Questions: noulQuestions(1),
		})
		if !errors.Is(err, ErrNoScorer) {
			t.Fatalf("Evaluate returned %v, want ErrNoScorer", err)
		}
	})

	t.Run("no questions", func(t *testing.T) {
		_, err := New(ok).Evaluate(context.Background(), Request{State: StringState("s")})
		if !errors.Is(err, ErrNoQuestions) {
			t.Fatalf("Evaluate returned %v, want ErrNoQuestions", err)
		}
	})

	t.Run("an invalid question", func(t *testing.T) {
		_, err := New(ok).Evaluate(context.Background(), Request{
			State:     StringState("s"),
			Questions: one("urgency", levelsOf(1)),
		})
		var verr *ValidationError
		if !errors.As(err, &verr) {
			t.Fatalf("Evaluate returned %v, want a *ValidationError", err)
		}
		if verr.Question != "urgency" || verr.Field != "criteria" {
			t.Fatalf("validation error named %q/%q, want urgency/criteria", verr.Question, verr.Field)
		}
	})

	t.Run("an unknown mode", func(t *testing.T) {
		_, err := New(ok).Evaluate(context.Background(), Request{
			State:     StringState("s"),
			Questions: noulQuestions(1),
			Mode:      "batched",
		})
		if !errors.Is(err, ErrUnknownMode) {
			t.Fatalf("Evaluate returned %v, want ErrUnknownMode", err)
		}
		if !strings.Contains(err.Error(), `"batched"`) {
			t.Errorf("error %q should quote the mode that was asked for", err)
		}
	})

	t.Run("the default mode is parallel", func(t *testing.T) {
		resp, err := New(ok).Evaluate(context.Background(), Request{
			State:     StringState("s"),
			Questions: noulQuestions(1),
		})
		if err != nil {
			t.Fatalf("Evaluate returned %v", err)
		}
		if resp.Meta.Mode != ModeParallel {
			t.Fatalf("Meta.Mode = %q, want %q", resp.Meta.Mode, ModeParallel)
		}
	})
}

// TestEvaluateLatencyUsesTheInjectedClock checks the measurement comes from the
// clock the caller supplied, which is what makes the golden document stable.
func TestEvaluateLatencyUsesTheInjectedClock(t *testing.T) {
	ok := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		return ScoreResult{Probability: 0.5}, nil
	})
	e := New(ok, WithClock(stepClock(time.Unix(0, 0), 1500*time.Millisecond)))

	resp, err := e.Evaluate(context.Background(), Request{
		State:     StringState("s"),
		Questions: noulQuestions(1),
	})
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}
	if resp.Meta.LatencyMS != 1500 {
		t.Fatalf("LatencyMS = %d, want 1500", resp.Meta.LatencyMS)
	}
}

func TestOptionsIgnoreNilArguments(t *testing.T) {
	e := New(ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		return ScoreResult{Probability: 0.5}, nil
	}), nil, WithClock(nil))

	if e.now == nil {
		t.Fatal("WithClock(nil) cleared the clock instead of being ignored")
	}
	if _, err := e.Evaluate(context.Background(), Request{
		State:     StringState("s"),
		Questions: noulQuestions(1),
	}); err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}
}

// TestRequestJSONRoundTrip pins that a request survives a decode and re-encode
// byte for byte, with its criteria order intact — including the "boolean"
// synonym, which normalises for evaluation but is written back as declared.
func TestRequestJSONRoundTrip(t *testing.T) {
	const doc = `{"state":{"subject":"Charged twice","attempts":2},` +
		`"questions":{` +
		`"department":{"type":"choice","instructions":"Which team should handle this?",` +
		`"criteria":{"zeta":"Last declared","alpha":"First declared","middle":"In between"}},` +
		`"urgency":{"type":"score","criteria":["Low","High"]},` +
		`"angry":{"type":"boolean"}},` +
		`"model":"probe-1","temperature":0.3,"mode":"oneshot"}`

	// The document comes back exactly as it went in.
	const want = doc

	var req Request
	if err := json.Unmarshal([]byte(doc), &req); err != nil {
		t.Fatalf("decoding the request: %v", err)
	}

	if got := req.Questions.Keys(); strings.Join(got, ",") != "department,urgency,angry" {
		t.Errorf("questions decoded in order %v, want the document's order", got)
	}
	department, _ := req.Questions.Get("department")
	if got := department.Options.Keys(); strings.Join(got, ",") != "zeta,alpha,middle" {
		t.Errorf("criteria decoded in order %v, want the document's order", got)
	}
	angry, _ := req.Questions.Get("angry")
	if angry.Type != TypeNoul {
		t.Errorf("the boolean synonym decoded as %q, want %q", angry.Type, TypeNoul)
	}

	encoded, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("re-encoding the request: %v", err)
	}
	if string(encoded) != want {
		t.Fatalf("round trip changed the document:\n got %s\nwant %s", encoded, want)
	}
}

// ------------------------------------------------- the softmax temperature --

// temperatureQuestions is one choice of three options whose scores genuinely
// differ, which is what a temperature test needs: equal scores are uniform at
// every temperature and three options redistribute mass visibly where two
// only trade it.
const temperatureQuestions = `{
	"pick": {"type": "choice", "criteria": {"high": "", "middle": "", "low": ""}}
}`

// temperatureScript scores the three options 0.9, 0.6 and 0.2.
var temperatureScript = map[string]float64{
	`The correct select the best label for "pick" is "high".`:   0.9,
	`The correct select the best label for "pick" is "middle".`: 0.6,
	`The correct select the best label for "pick" is "low".`:    0.2,
}

// pickProbabilities runs the shared fixture through an Evaluator built with
// opts and returns the answer it reported.
func pickProbabilities(t *testing.T, opts ...Option) Answer {
	t.Helper()
	resp, err := New(&scriptedScorer{t: t, script: temperatureScript}, opts...).
		Evaluate(context.Background(), Request{
			State:     StringState("s"),
			Questions: mustQuestions(t, temperatureQuestions),
		})
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}
	answer, ok := resp.Answers.Get("pick")
	if !ok {
		t.Fatal("no answer for the question")
	}
	return answer
}

// probabilitiesOf reads a distribution out in declaration order.
func probabilitiesOf(t *testing.T, a Answer, keys ...string) []float64 {
	t.Helper()
	out := make([]float64, len(keys))
	for i, key := range keys {
		p, ok := a.Probabilities.Get(key)
		if !ok {
			t.Fatalf("no probability reported for %q", key)
		}
		out[i] = p
	}
	return out
}

// TestEvaluateDefaultSoftmaxTemperatureIsTheIdentity pins the default at 1 and
// pins what 1 reports, so a default that drifted would be caught by the
// numbers and not only by the constant.
//
//	scoresToDistribution([0.9, 0.6, 0.2], 1) = [0.83720930232558155,
//	0.1395348837209302, 0.023255813953488379] → [0.837209, 0.139535, 0.023256]
func TestEvaluateDefaultSoftmaxTemperatureIsTheIdentity(t *testing.T) {
	if DefaultSoftmaxTemperature != 1 {
		t.Fatalf("DefaultSoftmaxTemperature = %v, want 1: the default has to be the identity "+
			"so that no existing deployment's numbers move", DefaultSoftmaxTemperature)
	}
	if got := New(nil).SoftmaxTemperature(); got != 1 {
		t.Fatalf("an Evaluator built with no options normalises at %v, want 1", got)
	}

	want := []float64{0.837209, 0.139535, 0.023256}
	byDefault := probabilitiesOf(t, pickProbabilities(t), "high", "middle", "low")
	if !equalFloats(byDefault, want) {
		t.Fatalf("with no option the distribution is %v, want %v", byDefault, want)
	}
	// Asking for 1 explicitly has to give the very same bits.
	explicit := probabilitiesOf(t,
		pickProbabilities(t, WithSoftmaxTemperature(1)), "high", "middle", "low")
	if !equalFloats(explicit, byDefault) {
		t.Fatalf("temperature 1 gave %v where the default gave %v; they must agree to the bit",
			explicit, byDefault)
	}
}

// TestEvaluateSoftmaxTemperatureReachesTheDistribution is the whole point of
// the option: a configured temperature has to arrive at the softmax, not just
// be parsed and stored. A different temperature must produce a different
// distribution — and the same choice, because it cannot reorder anything.
//
//	scoresToDistribution([0.9, 0.6, 0.2], 5.04) = [0.45621217132150665,
//	0.31972145323729606, 0.22406637544119723] → [0.456212, 0.319721, 0.224066]
func TestEvaluateSoftmaxTemperatureReachesTheDistribution(t *testing.T) {
	const fitted = 5.04

	hot := pickProbabilities(t, WithSoftmaxTemperature(fitted))
	cold := pickProbabilities(t)

	wantHot := []float64{0.456212, 0.319721, 0.224066}
	gotHot := probabilitiesOf(t, hot, "high", "middle", "low")
	if !equalFloats(gotHot, wantHot) {
		t.Fatalf("at temperature %v the distribution is %v, want %v", fitted, gotHot, wantHot)
	}

	gotCold := probabilitiesOf(t, cold, "high", "middle", "low")
	if equalFloats(gotHot, gotCold) {
		t.Fatalf("temperature %v reported the same distribution as the default, %v: "+
			"the configured value never reached the softmax", fitted, gotCold)
	}
	// Flatter, in every direction: the winner gives mass up and both losers
	// take it.
	if !(gotHot[0] < gotCold[0] && gotHot[1] > gotCold[1] && gotHot[2] > gotCold[2]) {
		t.Fatalf("a higher temperature gave %v against the default's %v, which is not flatter",
			gotHot, gotCold)
	}

	// And the answer itself is untouched, which is what makes the knob safe.
	if hot.Choice != cold.Choice || hot.Choice != "high" {
		t.Fatalf("choice at temperature %v is %q and at the default %q; want %q either way",
			fitted, hot.Choice, cold.Choice, "high")
	}
	if hot.Confidence == cold.Confidence {
		t.Fatalf("confidence is %v at both temperatures; it is computed from the "+
			"distribution and should have moved with it", hot.Confidence)
	}
}

// TestEvaluateSoftmaxTemperatureReachesAScoreQuestion is the same check for
// the other question type that normalises, since each builds its distribution
// on its own line.
func TestEvaluateSoftmaxTemperatureReachesAScoreQuestion(t *testing.T) {
	script := map[string]float64{
		`On the scale for "How urgent is this?", the most appropriate rating is level 0: "Low".`:    0.2,
		`On the scale for "How urgent is this?", the most appropriate rating is level 1: "Medium".`: 0.55,
		`On the scale for "How urgent is this?", the most appropriate rating is level 2: "High".`:   0.5,
	}
	questions := `{"urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Low", "Medium", "High"]}}`

	run := func(opts ...Option) Answer {
		t.Helper()
		resp, err := New(&scriptedScorer{t: t, script: script}, opts...).
			Evaluate(context.Background(), Request{
				State:     StringState("s"),
				Questions: mustQuestions(t, questions),
			})
		if err != nil {
			t.Fatalf("Evaluate returned %v", err)
		}
		a, _ := resp.Answers.Get("urgency")
		return a
	}

	base := run()
	warm := run(WithSoftmaxTemperature(5.04))
	if equalFloats(
		probabilitiesOf(t, base, "0", "1", "2"),
		probabilitiesOf(t, warm, "0", "1", "2"),
	) {
		t.Fatal("a score question reported the same distribution at both temperatures: " +
			"the configured value never reached its softmax")
	}
	if base.Score == warm.Score {
		t.Fatalf("the expected value is %v at both temperatures, though the distribution it "+
			"is taken over changed", base.Score)
	}
}

// TestWithSoftmaxTemperatureIgnoresWhatIsNotATemperature pins the option's
// side of the contract. The configuration loader refuses these at startup;
// a library caller who passes one gets the default rather than an Evaluator
// that reports NaN for every probability it will ever produce.
func TestWithSoftmaxTemperatureIgnoresWhatIsNotATemperature(t *testing.T) {
	for _, bad := range []float64{0, -1, -0.0001, math.NaN(), math.Inf(1)} {
		e := New(nil, WithSoftmaxTemperature(bad))
		if got := e.SoftmaxTemperature(); got != DefaultSoftmaxTemperature {
			t.Errorf("WithSoftmaxTemperature(%v) left the temperature at %v, want the default %v",
				bad, got, DefaultSoftmaxTemperature)
		}
	}
	// A small positive value is a temperature and is kept, even though the
	// softmax will floor it: the option's job is to reject what is not a
	// temperature, not to second-guess one.
	if got := New(nil, WithSoftmaxTemperature(0.01)).SoftmaxTemperature(); got != 0.01 {
		t.Errorf("WithSoftmaxTemperature(0.01) gave %v, want 0.01", got)
	}
}

// TestEvaluatorBuiltAsALiteralStillNormalisesAtTheDefault covers the zero
// value. An Evaluator assembled as a struct literal has a zero temperature,
// which the softmax would floor to 0.05 and report a near one-hot distribution
// from.
func TestEvaluatorBuiltAsALiteralStillNormalisesAtTheDefault(t *testing.T) {
	literal := &Evaluator{scorer: &scriptedScorer{t: t, script: temperatureScript}}
	resp, err := literal.Evaluate(context.Background(), Request{
		State:     StringState("s"),
		Questions: mustQuestions(t, temperatureQuestions),
	})
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}
	answer, _ := resp.Answers.Get("pick")
	want := []float64{0.837209, 0.139535, 0.023256}
	if got := probabilitiesOf(t, answer, "high", "middle", "low"); !equalFloats(got, want) {
		t.Fatalf("a zero-valued Evaluator reported %v, want the default temperature's %v", got, want)
	}
}

// TestEvaluateReportsNoCertainty is the end-to-end form of the reporting
// property: a model that saturated every score still produces an answer whose
// every published number can be fed back through a log.
func TestEvaluateReportsNoCertainty(t *testing.T) {
	script := map[string]float64{
		`The correct select the best label for "pick" is "yes".`: 1,
		`The correct select the best label for "pick" is "no".`:  0,
	}
	resp, err := New(&scriptedScorer{t: t, script: script}).Evaluate(context.Background(), Request{
		State:     StringState("s"),
		Questions: mustQuestions(t, `{"pick": {"type": "choice", "criteria": {"yes": "", "no": ""}}}`),
	})
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}
	answer, _ := resp.Answers.Get("pick")
	for _, key := range []string{"yes", "no"} {
		p, _ := answer.Probabilities.Get(key)
		if p == 0 || p == 1 {
			t.Errorf("probability of %q is %v; the estimator clamps its inputs to "+
				"(0.001, 0.999) and can never mean a certainty", key, p)
		}
	}
	if want := 0.999999; func() float64 { p, _ := answer.Probabilities.Get("yes"); return p }() != want {
		p, _ := answer.Probabilities.Get("yes")
		t.Errorf("probability of the winner is %v, want %v", p, want)
	}

	// A noul is the one reported probability that may be 0 or 1: it is the
	// scorer's own number, with no softmax between it and the wire. What the
	// finer rounding buys there is that a number just short of 1 stays just
	// short of it.
	noul, err := New(&scriptedScorer{t: t, script: map[string]float64{`The proposition "almost" is true.`: 0.9996}}).
		Evaluate(context.Background(), Request{
			State:     StringState("s"),
			Questions: mustQuestions(t, `{"almost": {"type": "noul"}}`),
		})
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}
	a, _ := noul.Answers.Get("almost")
	if a.Noul != 0.9996 {
		t.Errorf("noul = %v, want 0.9996 reported as itself and not rounded up to 1", a.Noul)
	}
}

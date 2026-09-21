package calibration

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/titusai-io/perceptea/classifier"
)

// The runner is exercised through a real *classifier.Evaluator over a fake
// Scorer, rather than a fake evaluator, so that the plumbing between a case
// and an answer is the plumbing the service actually uses. Nothing here
// reaches the network.

// stateScorer answers by state: whatever probability the script holds for the
// state a call was made about. An error in the script is returned instead.
type stateScorer struct {
	probabilities map[string]float64
	errs          map[string]error
	delays        map[string]time.Duration

	mu    sync.Mutex
	calls int
	// onCall runs after every call is counted, so a test can interrupt a run
	// at a known point.
	onCall func(n int)
}

func (s *stateScorer) Score(ctx context.Context, req classifier.ScoreRequest) (classifier.ScoreResult, error) {
	s.mu.Lock()
	s.calls++
	n := s.calls
	hook := s.onCall
	s.mu.Unlock()
	if hook != nil {
		hook(n)
	}

	if d, ok := s.delays[req.State]; ok {
		select {
		case <-time.After(d):
		case <-ctx.Done():
			return classifier.ScoreResult{}, ctx.Err()
		}
	}
	if err, ok := s.errs[req.State]; ok {
		return classifier.ScoreResult{}, err
	}
	p, ok := s.probabilities[req.State]
	if !ok {
		return classifier.ScoreResult{}, fmt.Errorf("unscripted state %q", req.State)
	}
	return classifier.ScoreResult{Probability: p}, nil
}

// noulDataset builds one noul case per state, each named after its state.
func noulDataset(t *testing.T, states ...string) []Case {
	t.Helper()
	lines := make([]string, 0, len(states))
	for _, state := range states {
		lines = append(lines, fmt.Sprintf(
			`{"id":%q,"name":"holds","state":%q,"question":{"type":"noul","instructions":"It holds."},"answer":true}`,
			state, state))
	}
	cases, err := ParseDataset(strings.NewReader(strings.Join(lines, "\n") + "\n"))
	if err != nil {
		t.Fatalf("building the dataset: %v", err)
	}
	return cases
}

func TestRunnerAnswersEveryCase(t *testing.T) {
	scorer := &stateScorer{probabilities: map[string]float64{"clouds": 0.8, "sun": 0.1}}
	runner := Runner{Evaluator: classifier.New(scorer)}

	run, err := runner.Run(context.Background(), noulDataset(t, "clouds", "sun"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if run.Cases != 2 || run.Repeat != 1 || len(run.Outcomes) != 2 {
		t.Fatalf("run = %d cases, %d repeat, %d outcomes; want 2, 1, 2",
			run.Cases, run.Repeat, len(run.Outcomes))
	}
	// A noul answer is the scored probability itself, so these are the
	// scorer's own numbers arriving intact.
	for i, want := range []float64{0.8, 0.1} {
		o := run.Outcomes[i]
		if o.Err != nil {
			t.Fatalf("outcome %d failed: %v", i, o.Err)
		}
		if o.Answer.Noul != want {
			t.Errorf("outcome %d noul = %v, want %v", i, o.Answer.Noul, want)
		}
		if o.Attempt != 1 {
			t.Errorf("outcome %d attempt = %d, want 1", i, o.Attempt)
		}
	}
}

func TestRunnerRecordsAFailureAndKeepsGoing(t *testing.T) {
	scorer := &stateScorer{
		probabilities: map[string]float64{"clouds": 0.8, "fog": 0.4},
		errs:          map[string]error{"sun": errors.New("upstream said no")},
	}
	runner := Runner{Evaluator: classifier.New(scorer)}

	run, err := runner.Run(context.Background(), noulDataset(t, "clouds", "sun", "fog"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if len(run.Outcomes) != 3 {
		t.Fatalf("got %d outcomes, want 3", len(run.Outcomes))
	}
	if run.Outcomes[0].Err != nil {
		t.Errorf("first case failed: %v", run.Outcomes[0].Err)
	}
	if run.Outcomes[1].Err == nil {
		t.Fatal("the failing case reported no error")
	}
	if !strings.Contains(run.Outcomes[1].Err.Error(), "upstream said no") {
		t.Errorf("failure = %v, want the provider's reason", run.Outcomes[1].Err)
	}
	// The case after the failure is the point: one bad call must not cost the
	// measurements that came after it.
	if run.Outcomes[2].Err != nil {
		t.Errorf("the case after the failure did not run: %v", run.Outcomes[2].Err)
	}
	if run.Outcomes[2].Answer.Noul != 0.4 {
		t.Errorf("third noul = %v, want 0.4", run.Outcomes[2].Answer.Noul)
	}
}

func TestRunnerRepeatsEveryCase(t *testing.T) {
	scorer := &stateScorer{probabilities: map[string]float64{"clouds": 0.8, "sun": 0.1}}
	runner := Runner{Evaluator: classifier.New(scorer), Repeat: 3}

	run, err := runner.Run(context.Background(), noulDataset(t, "clouds", "sun"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if run.Repeat != 3 || len(run.Outcomes) != 6 {
		t.Fatalf("run = %d repeat, %d outcomes; want 3, 6", run.Repeat, len(run.Outcomes))
	}
	// Each case's repeats stay together and are numbered from one, so a
	// report can say which pass a failure came from.
	wantIDs := []string{"clouds", "clouds", "clouds", "sun", "sun", "sun"}
	wantAttempts := []int{1, 2, 3, 1, 2, 3}
	for i, o := range run.Outcomes {
		if o.Case.ID != wantIDs[i] || o.Attempt != wantAttempts[i] {
			t.Errorf("outcome %d = %s attempt %d, want %s attempt %d",
				i, o.Case.ID, o.Attempt, wantIDs[i], wantAttempts[i])
		}
	}
}

func TestRunnerKeepsDatasetOrderUnderConcurrency(t *testing.T) {
	const n = 8
	states := make([]string, n)
	probabilities := make(map[string]float64, n)
	delays := make(map[string]time.Duration, n)
	for i := range n {
		state := fmt.Sprintf("s%d", i)
		states[i] = state
		probabilities[state] = float64(i) / 10
		// Earlier cases are slower, so a runner that collected results as
		// they finished would report them very nearly backwards.
		delays[state] = time.Duration(n-i) * 2 * time.Millisecond
	}
	scorer := &stateScorer{probabilities: probabilities, delays: delays}
	runner := Runner{Evaluator: classifier.New(scorer), Concurrency: n}

	run, err := runner.Run(context.Background(), noulDataset(t, states...))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if len(run.Outcomes) != n {
		t.Fatalf("got %d outcomes, want %d", len(run.Outcomes), n)
	}
	for i, o := range run.Outcomes {
		if o.Case.ID != states[i] {
			t.Fatalf("outcome %d is %s, want %s: the report is not in dataset order",
				i, o.Case.ID, states[i])
		}
	}
}

func TestRunnerNeedsAnEvaluator(t *testing.T) {
	_, err := Runner{}.Run(context.Background(), noulDataset(t, "clouds"))

	if !errors.Is(err, ErrNoEvaluator) {
		t.Errorf("Run error = %v, want %v", err, ErrNoEvaluator)
	}
}

func TestRunnerStopsWhenTheContextIsCancelled(t *testing.T) {
	const n = 10
	states := make([]string, n)
	probabilities := make(map[string]float64, n)
	for i := range n {
		states[i] = fmt.Sprintf("s%d", i)
		probabilities[states[i]] = 0.5
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	scorer := &stateScorer{
		probabilities: probabilities,
		onCall: func(callNumber int) {
			if callNumber == 2 {
				cancel()
			}
		},
	}
	// One case at a time, so the cancellation lands with most of the dataset
	// still undispatched.
	runner := Runner{Evaluator: classifier.New(scorer), Concurrency: 1}

	run, err := runner.Run(ctx, noulDataset(t, states...))

	if !errors.Is(err, context.Canceled) {
		t.Errorf("Run error = %v, want %v", err, context.Canceled)
	}
	// Undispatched attempts are dropped rather than reported as empty
	// successes, so a cancelled run cannot claim to have measured the rest.
	if len(run.Outcomes) >= n {
		t.Errorf("got %d outcomes, want fewer than %d", len(run.Outcomes), n)
	}
	for i, o := range run.Outcomes {
		if o.Attempt == 0 {
			t.Errorf("outcome %d was never dispatched but is in the report", i)
		}
	}
}

// answerEvaluator returns a fixed response. It exists for the two guards a
// real evaluator cannot be made to trip: an answer filed under the wrong name
// and an answer of the wrong type. Both would silently score one metric
// against another's label, so both have to be reachable in a test.
type answerEvaluator struct {
	name   string
	answer classifier.Answer
}

func (e answerEvaluator) Evaluate(context.Context, classifier.Request) (classifier.Response, error) {
	answers := classifier.NewOrderedMap[classifier.Answer]()
	answers.Set(e.name, e.answer)
	return classifier.Response{Answers: *answers}, nil
}

func TestRunnerFailsACaseWithNoAnswerOfItsOwn(t *testing.T) {
	runner := Runner{Evaluator: answerEvaluator{
		name:   "something else",
		answer: classifier.Answer{Type: classifier.TypeNoul, Noul: 0.5},
	}}

	run, err := runner.Run(context.Background(), noulDataset(t, "clouds"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if run.Outcomes[0].Err == nil {
		t.Fatal("a response with no answer for the question was accepted")
	}
	// The reason matters as much as the failure: "no answer for this
	// question" and "the wrong kind of answer" are different faults, and the
	// message is all the report will carry.
	for _, want := range []string{"no answer", `"holds"`} {
		if !strings.Contains(run.Outcomes[0].Err.Error(), want) {
			t.Errorf("failure = %v, want it to mention %q", run.Outcomes[0].Err, want)
		}
	}
}

func TestRunnerFailsACaseAnsweredWithTheWrongType(t *testing.T) {
	runner := Runner{Evaluator: answerEvaluator{
		name:   "holds",
		answer: classifier.Answer{Type: classifier.TypeChoice, Choice: "a"},
	}}

	run, err := runner.Run(context.Background(), noulDataset(t, "clouds"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if run.Outcomes[0].Err == nil {
		t.Fatal("a choice answer to a noul question was accepted")
	}
	for _, want := range []string{"noul", "choice"} {
		if !strings.Contains(run.Outcomes[0].Err.Error(), want) {
			t.Errorf("failure = %v, want it to mention %q", run.Outcomes[0].Err, want)
		}
	}
}

func TestRunnerWithNoCases(t *testing.T) {
	run, err := Runner{Evaluator: classifier.New(&stateScorer{})}.Run(context.Background(), nil)

	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if run.Cases != 0 || len(run.Outcomes) != 0 {
		t.Errorf("run = %d cases, %d outcomes; want 0, 0", run.Cases, len(run.Outcomes))
	}
}

// blockingScorer reports every call that starts and holds it until released.
type blockingScorer struct {
	started chan struct{}
	release chan struct{}
}

func (b *blockingScorer) Score(context.Context, classifier.ScoreRequest) (classifier.ScoreResult, error) {
	b.started <- struct{}{}
	<-b.release
	return classifier.ScoreResult{Probability: 0.5}, nil
}

func TestThrottleBoundsCallsInFlight(t *testing.T) {
	const (
		limit = 2
		calls = 5
	)
	inner := &blockingScorer{
		started: make(chan struct{}, calls),
		release: make(chan struct{}),
	}
	scorer := Throttle(inner, limit)

	done := make(chan struct{})
	for range calls {
		go func() {
			defer func() { done <- struct{}{} }()
			if _, err := scorer.Score(context.Background(), classifier.ScoreRequest{}); err != nil {
				t.Errorf("Score: %v", err)
			}
		}()
	}

	// Exactly the limit may start. Waiting for those is a blocking receive,
	// so it cannot pass early; the third is the one that must not arrive.
	for range limit {
		<-inner.started
	}
	select {
	case <-inner.started:
		t.Fatalf("a call beyond the limit of %d started", limit)
	case <-time.After(100 * time.Millisecond):
	}

	close(inner.release)
	for range calls {
		<-done
	}
	for range calls - limit {
		<-inner.started
	}
}

func TestThrottleGivesUpOnACancelledContext(t *testing.T) {
	inner := &blockingScorer{started: make(chan struct{}, 1), release: make(chan struct{})}
	defer close(inner.release)
	scorer := Throttle(inner, 1)

	go func() {
		_, _ = scorer.Score(context.Background(), classifier.ScoreRequest{})
	}()
	<-inner.started

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := scorer.Score(ctx, classifier.ScoreRequest{}); !errors.Is(err, context.Canceled) {
		t.Errorf("Score error = %v, want %v", err, context.Canceled)
	}
}

func TestThrottleWithoutALimitPassesThrough(t *testing.T) {
	inner := classifier.ScorerFunc(func(context.Context, classifier.ScoreRequest) (classifier.ScoreResult, error) {
		return classifier.ScoreResult{Probability: 0.25}, nil
	})

	got := Throttle(inner, 0)

	res, err := got.Score(context.Background(), classifier.ScoreRequest{})
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if res.Probability != 0.25 {
		t.Errorf("probability = %v, want 0.25", res.Probability)
	}
}

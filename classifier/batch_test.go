package classifier

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ---------------------------------------------------------------- helpers --

// batchOf builds n items named t-0…t-(n-1), each with its own state text.
func batchOf(n int) []BatchItem {
	items := make([]BatchItem, n)
	for i := range n {
		items[i] = BatchItem{ID: fmt.Sprintf("t-%d", i), State: StringState(fmt.Sprintf("state %d", i))}
	}
	return items
}

// countingScorer answers every call the same way and records the highest
// number of calls it ever had in flight at once.
type countingScorer struct {
	probability float64
	hold        time.Duration

	mu          sync.Mutex
	calls       int
	inFlight    int
	maxInFlight int
}

func (s *countingScorer) Score(_ context.Context, _ ScoreRequest) (ScoreResult, error) {
	s.mu.Lock()
	s.calls++
	s.inFlight++
	if s.inFlight > s.maxInFlight {
		s.maxInFlight = s.inFlight
	}
	s.mu.Unlock()

	time.Sleep(s.hold)

	s.mu.Lock()
	s.inFlight--
	s.mu.Unlock()
	return ScoreResult{Probability: s.probability}, nil
}

func (s *countingScorer) peak() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxInFlight
}

func (s *countingScorer) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

// resultFor returns the result carrying the given id.
func resultFor(t *testing.T, resp BatchResponse, id string) BatchResult {
	t.Helper()
	for _, r := range resp.Results {
		if r.ID == id {
			return r
		}
	}
	t.Fatalf("no result for item %q in %+v", id, resp.Results)
	return BatchResult{}
}

// ------------------------------------------------------- the shared budget --

// TestEvaluateBatchSharesOneConcurrencyBudget is the whole point of the batch:
// ten items of six candidates is sixty calls, and the limit bounds all sixty
// together. Given each item its own budget the peak would be ten times the
// limit, which is what this counts.
func TestEvaluateBatchSharesOneConcurrencyBudget(t *testing.T) {
	const (
		limit   = 4
		items   = 10
		perItem = 6
	)

	// Every admitted call holds its slot long enough that everything else
	// admitted alongside it is counted in flight at the same time.
	scorer := &countingScorer{probability: 0.5, hold: 15 * time.Millisecond}

	resp, err := New(scorer, WithMaxConcurrency(limit)).EvaluateBatch(context.Background(), BatchRequest{
		Items:     batchOf(items),
		Questions: noulQuestions(perItem),
	})
	if err != nil {
		t.Fatalf("EvaluateBatch returned %v", err)
	}
	if got := scorer.callCount(); got != items*perItem {
		t.Fatalf("the scorer was called %d times, want %d", got, items*perItem)
	}
	if resp.Meta.ParallelCalls != items*perItem {
		t.Errorf("Meta.ParallelCalls = %d, want %d", resp.Meta.ParallelCalls, items*perItem)
	}

	peak := scorer.peak()
	if peak > limit {
		t.Fatalf("saw %d scorer calls in flight at once across the batch, want at most %d: "+
			"the budget is being applied per item rather than per batch", peak, limit)
	}
	if peak < 2 {
		t.Fatalf("saw at most %d call in flight; the batch is not running concurrently at all", peak)
	}
}

// TestEvaluateBatchRunsItemsConcurrently is the other half of the same claim.
// Four items of one candidate each, with a budget of four, must all be in
// flight together: a batch that ran its items one after another — or gave each
// one a budget it could only use within itself — would never get four calls to
// the rendezvous and would fail on the timeout.
func TestEvaluateBatchRunsItemsConcurrently(t *testing.T) {
	const items = 4

	var mu sync.Mutex
	arrived := 0
	everyoneHere := make(chan struct{})

	scorer := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		mu.Lock()
		arrived++
		if arrived == items {
			close(everyoneHere)
		}
		mu.Unlock()

		select {
		case <-everyoneHere:
			return ScoreResult{Probability: 0.5}, nil
		case <-time.After(5 * time.Second):
			return ScoreResult{}, errors.New("the batch never had one call per item in flight at once")
		}
	})

	resp, err := New(scorer, WithMaxConcurrency(items)).EvaluateBatch(context.Background(), BatchRequest{
		Items:     batchOf(items),
		Questions: noulQuestions(1),
	})
	if err != nil {
		t.Fatalf("EvaluateBatch returned %v", err)
	}
	for _, r := range resp.Results {
		if r.Error != "" {
			t.Fatalf("item %d failed: %s", r.Index, r.Error)
		}
	}
}

// TestEvaluateBatchUnlimitedConcurrency checks the nil-semaphore branch is
// reached for a batch as it is for a single evaluation: with no limit, every
// call of every item runs at once.
func TestEvaluateBatchUnlimitedConcurrency(t *testing.T) {
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

	if _, err := New(scorer, WithMaxConcurrency(0)).EvaluateBatch(context.Background(), BatchRequest{
		Items:     batchOf(4),
		Questions: noulQuestions(3),
	}); err != nil {
		t.Fatalf("EvaluateBatch returned %v", err)
	}
}

// ------------------------------------------------------------ per-item errors --

// TestEvaluateBatchOneFailedItemLeavesTheRestIntact is the rule a batch exists
// for: the items are independent, so one state that the provider refuses must
// not discard the states it answered.
func TestEvaluateBatchOneFailedItemLeavesTheRestIntact(t *testing.T) {
	refused := errors.New("provider refused the request")

	scorer := ScorerFunc(func(ctx context.Context, req ScoreRequest) (ScoreResult, error) {
		if req.State == "state 1" {
			return ScoreResult{}, refused
		}
		return ScoreResult{Probability: 0.5, InputTokens: 3, OutputTokens: 1}, nil
	})

	resp, err := New(scorer, WithMaxConcurrency(4)).EvaluateBatch(context.Background(), BatchRequest{
		Items:     batchOf(3),
		Questions: noulQuestions(2),
	})
	if err != nil {
		t.Fatalf("EvaluateBatch returned %v, want a batch that served what it could", err)
	}

	for _, id := range []string{"t-0", "t-2"} {
		r := resultFor(t, resp, id)
		if r.Error != "" {
			t.Errorf("item %q failed with %q; one failing item took the others with it", id, r.Error)
		}
		if r.Answers.Len() != 2 {
			t.Errorf("item %q has %d answers, want 2", id, r.Answers.Len())
		}
	}

	bad := resultFor(t, resp, "t-1")
	if !strings.Contains(bad.Error, refused.Error()) {
		t.Errorf("the failed item's error = %q, want the scorer's own", bad.Error)
	}
	if bad.Answers.Len() != 0 {
		t.Errorf("the failed item carries %d answers", bad.Answers.Len())
	}

	if resp.Meta.Items != 3 || resp.Meta.Succeeded != 2 || resp.Meta.Failed != 1 {
		t.Errorf("meta counts = %d items, %d succeeded, %d failed; want 3/2/1",
			resp.Meta.Items, resp.Meta.Succeeded, resp.Meta.Failed)
	}
}

// TestEvaluateBatchFailedItemStopsOnlyItsOwnCalls checks the failure is
// contained the other way too: the failing item abandons its own remaining
// candidates rather than running them, and no other item's are touched.
func TestEvaluateBatchFailedItemStopsOnlyItsOwnCalls(t *testing.T) {
	refused := errors.New("provider refused the request")

	var mu sync.Mutex
	perState := map[string]int{}

	scorer := ScorerFunc(func(ctx context.Context, req ScoreRequest) (ScoreResult, error) {
		mu.Lock()
		perState[req.State]++
		n := perState[req.State]
		mu.Unlock()

		if req.State == "state 0" {
			if n == 1 {
				return ScoreResult{}, refused
			}
			// Anything of the failing item already in flight waits to be
			// cancelled, so the item's reported error is the first one.
			select {
			case <-ctx.Done():
				return ScoreResult{}, ctx.Err()
			case <-time.After(2 * time.Second):
				return ScoreResult{}, errors.New("was never cancelled")
			}
		}
		return ScoreResult{Probability: 0.5}, nil
	})

	const perItem = 20
	resp, err := New(scorer, WithMaxConcurrency(4)).EvaluateBatch(context.Background(), BatchRequest{
		Items:     batchOf(2),
		Questions: noulQuestions(perItem),
	})
	if err != nil {
		t.Fatalf("EvaluateBatch returned %v", err)
	}

	if got := resultFor(t, resp, "t-0").Error; !strings.Contains(got, refused.Error()) {
		t.Errorf("item t-0 error = %q, want the first failure, not a cancellation", got)
	}
	if got := resultFor(t, resp, "t-1").Error; got != "" {
		t.Errorf("item t-1 failed with %q; the other item's cancellation reached it", got)
	}

	mu.Lock()
	failing, healthy := perState["state 0"], perState["state 1"]
	mu.Unlock()
	if failing >= perItem {
		t.Errorf("the failing item issued %d of its %d calls; the rest should have been abandoned", failing, perItem)
	}
	if healthy != perItem {
		t.Errorf("the healthy item issued %d of its %d calls; it was cancelled along with the other", healthy, perItem)
	}
}

// A state is the one part of a batch that belongs to the item rather than to
// the request, so a missing one is that item's failure and not the batch's.
func TestEvaluateBatchItemWithoutAState(t *testing.T) {
	for _, tc := range []struct {
		name  string
		state State
	}{
		{name: "absent", state: State{}},
		{name: "null", state: RawState(json.RawMessage("null"))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			scorer := &constantScorer{probability: 0.5}
			resp, err := New(scorer).EvaluateBatch(context.Background(), BatchRequest{
				Items: []BatchItem{
					{ID: "empty", State: tc.state},
					{ID: "fine", State: StringState("something")},
				},
				Questions: noulQuestions(2),
			})
			if err != nil {
				t.Fatalf("EvaluateBatch returned %v", err)
			}
			if got := resultFor(t, resp, "empty").Error; !strings.Contains(got, "state") {
				t.Errorf("the stateless item's error = %q, want it to name the missing state", got)
			}
			if got := resultFor(t, resp, "fine").Error; got != "" {
				t.Errorf("the other item failed with %q", got)
			}
			// The stateless item is never put to the provider: two questions,
			// one usable item, two calls.
			if got := scorer.callCount(); got != 2 {
				t.Errorf("the scorer was called %d times, want 2: the stateless item was still dispatched", got)
			}
			if resp.Meta.Failed != 1 || resp.Meta.Succeeded != 1 {
				t.Errorf("meta = %d succeeded, %d failed; want 1/1", resp.Meta.Succeeded, resp.Meta.Failed)
			}
		})
	}
}

// ------------------------------------------------------- shape and ordering --

// TestEvaluateBatchKeepsRequestOrder checks the results come back in the order
// the items were sent, each carrying its own index and id, however the
// concurrent calls finished.
func TestEvaluateBatchKeepsRequestOrder(t *testing.T) {
	const items = 6

	// The first item answers last, so a batch that collected results as they
	// arrived would return them in a different order.
	scorer := ScorerFunc(func(_ context.Context, req ScoreRequest) (ScoreResult, error) {
		if req.State == "state 0" {
			time.Sleep(25 * time.Millisecond)
		}
		return ScoreResult{Probability: 0.5}, nil
	})

	resp, err := New(scorer, WithMaxConcurrency(items)).EvaluateBatch(context.Background(), BatchRequest{
		Items:     batchOf(items),
		Questions: noulQuestions(1),
	})
	if err != nil {
		t.Fatalf("EvaluateBatch returned %v", err)
	}
	if len(resp.Results) != items {
		t.Fatalf("got %d results, want %d", len(resp.Results), items)
	}
	for i, r := range resp.Results {
		if r.Index != i {
			t.Errorf("results[%d].Index = %d", i, r.Index)
		}
		if want := fmt.Sprintf("t-%d", i); r.ID != want {
			t.Errorf("results[%d].ID = %q, want %q", i, r.ID, want)
		}
	}
}

// An item without an id still gets a result, and its index is what places it.
func TestEvaluateBatchResultsPlaceThemselvesWithoutIDs(t *testing.T) {
	resp, err := New(&constantScorer{probability: 0.5}).EvaluateBatch(context.Background(), BatchRequest{
		Items: []BatchItem{
			{State: StringState("first")},
			{State: StringState("second")},
		},
		Questions: noulQuestions(1),
	})
	if err != nil {
		t.Fatalf("EvaluateBatch returned %v", err)
	}
	for i, r := range resp.Results {
		if r.ID != "" {
			t.Errorf("results[%d].ID = %q, want the empty id echoed back", i, r.ID)
		}
		if r.Index != i {
			t.Errorf("results[%d].Index = %d", i, r.Index)
		}
	}
}

// TestEvaluateBatchOfOneMatchesASingleEvaluation is the equivalence the two
// paths are built to keep: one item, the same questions, the same scorer, and
// the answers, the usage and the calls put to the provider are the same.
func TestEvaluateBatchOfOneMatchesASingleEvaluation(t *testing.T) {
	questions := mustQuestions(t, goldenQuestions)
	const state = "Charged twice again!! Billed twice for the Pro plan."

	single := &scriptedScorer{t: t, script: goldenScript, inputTokens: 3, outputTokens: 1}
	one, err := New(single).Evaluate(context.Background(), Request{
		State:       StringState(state),
		Questions:   questions,
		Model:       "probe-1",
		Temperature: 0.25,
	})
	if err != nil {
		t.Fatalf("Evaluate returned %v", err)
	}

	batched := &scriptedScorer{t: t, script: goldenScript, inputTokens: 3, outputTokens: 1}
	many, err := New(batched).EvaluateBatch(context.Background(), BatchRequest{
		Items:       []BatchItem{{ID: "only", State: StringState(state)}},
		Questions:   questions,
		Model:       "probe-1",
		Temperature: 0.25,
	})
	if err != nil {
		t.Fatalf("EvaluateBatch returned %v", err)
	}

	if len(many.Results) != 1 {
		t.Fatalf("got %d results, want 1", len(many.Results))
	}
	got := many.Results[0]
	if got.Error != "" {
		t.Fatalf("the only item failed: %s", got.Error)
	}

	wantAnswers, err := json.Marshal(one.Answers)
	if err != nil {
		t.Fatalf("marshalling the single evaluation's answers: %v", err)
	}
	gotAnswers, err := json.Marshal(got.Answers)
	if err != nil {
		t.Fatalf("marshalling the batched answers: %v", err)
	}
	if string(gotAnswers) != string(wantAnswers) {
		t.Errorf("a batch of one answered differently:\n got %s\nwant %s", gotAnswers, wantAnswers)
	}

	wantUsage, _ := json.Marshal(one.Usage)
	gotUsage, _ := json.Marshal(got.Usage)
	if string(gotUsage) != string(wantUsage) {
		t.Errorf("a batch of one cost differently: got %s, want %s", gotUsage, wantUsage)
	}
	if string(gotUsage) != string(mustJSON(t, many.Usage)) {
		t.Errorf("the batch total %s is not the only item's usage %s", mustJSON(t, many.Usage), gotUsage)
	}
	if many.Model != one.Model {
		t.Errorf("model = %q, want %q", many.Model, one.Model)
	}
	if many.Meta.Mode != one.Meta.Mode {
		t.Errorf("mode = %q, want %q", many.Meta.Mode, one.Meta.Mode)
	}
	if many.Meta.ParallelCalls != one.Meta.ParallelCalls {
		t.Errorf("ParallelCalls = %d, want %d", many.Meta.ParallelCalls, one.Meta.ParallelCalls)
	}

	// The provider saw the same calls, not merely the same number of them.
	if got, want := scoreRequestKeys(batched), scoreRequestKeys(single); !slices.Equal(got, want) {
		t.Errorf("a batch of one put different calls to the scorer:\n got %v\nwant %v", got, want)
	}
}

// scoreRequestKeys renders every call a scorer saw, sorted, so two runs can be
// compared without depending on the order the wave finished in.
func scoreRequestKeys(s *scriptedScorer) []string {
	reqs := s.requests()
	keys := make([]string, 0, len(reqs))
	for _, r := range reqs {
		keys = append(keys, fmt.Sprintf("%s|%s|%v|%s", r.Model, r.State, r.Temperature, r.Statement))
	}
	slices.Sort(keys)
	return keys
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshalling %T: %v", v, err)
	}
	return b
}

// ------------------------------------------------------------------- meta --

func TestEvaluateBatchMetaAndUsageTotals(t *testing.T) {
	scorer := ScorerFunc(func(_ context.Context, req ScoreRequest) (ScoreResult, error) {
		return ScoreResult{Probability: 0.5, InputTokens: 3, OutputTokens: 1}, nil
	})

	resp, err := New(scorer, WithClock(stepClock(time.Unix(0, 0), 700*time.Millisecond))).
		EvaluateBatch(context.Background(), BatchRequest{
			Items:     batchOf(3),
			Questions: mustQuestions(t, `{"a": {"type": "choice", "criteria": {"x": "", "y": ""}}, "b": {"type": "noul"}}`),
			Model:     "probe-1",
		})
	if err != nil {
		t.Fatalf("EvaluateBatch returned %v", err)
	}

	// Three items of three candidates each.
	if resp.Meta.ParallelCalls != 9 {
		t.Errorf("Meta.ParallelCalls = %d, want 9", resp.Meta.ParallelCalls)
	}
	if resp.Meta.Items != 3 || resp.Meta.Succeeded != 3 || resp.Meta.Failed != 0 {
		t.Errorf("meta counts = %d/%d/%d, want 3/3/0", resp.Meta.Items, resp.Meta.Succeeded, resp.Meta.Failed)
	}
	if resp.Meta.LatencyMS != 700 {
		t.Errorf("LatencyMS = %d, want the injected clock's 700", resp.Meta.LatencyMS)
	}
	if resp.Meta.Mode != ModeParallel {
		t.Errorf("Meta.Mode = %q, want %q", resp.Meta.Mode, ModeParallel)
	}
	if got, want := string(mustJSON(t, resp.Usage)), `{"input_tokens":27,"output_tokens":9}`; got != want {
		t.Errorf("batch usage = %s, want %s", got, want)
	}
	for _, r := range resp.Results {
		if got, want := string(mustJSON(t, r.Usage)), `{"input_tokens":9,"output_tokens":3}`; got != want {
			t.Errorf("item %d usage = %s, want %s", r.Index, got, want)
		}
	}
}

// A call that failed late still cost what it cost, so the tokens the completed
// calls of a failed item reported are counted, on the item and in the total.
func TestEvaluateBatchCountsAFailedItemsUsage(t *testing.T) {
	failAfter := make(chan struct{})
	var once sync.Once

	scorer := ScorerFunc(func(_ context.Context, req ScoreRequest) (ScoreResult, error) {
		if req.State == "state 0" && strings.Contains(req.Statement, "q1") {
			// Let the item's other call report its tokens first.
			<-failAfter
			return ScoreResult{}, errors.New("refused")
		}
		if req.State == "state 0" {
			defer once.Do(func() { close(failAfter) })
		}
		return ScoreResult{Probability: 0.5, InputTokens: 5, OutputTokens: 2}, nil
	})

	resp, err := New(scorer, WithMaxConcurrency(4)).EvaluateBatch(context.Background(), BatchRequest{
		Items:     batchOf(2),
		Questions: noulQuestions(2),
	})
	if err != nil {
		t.Fatalf("EvaluateBatch returned %v", err)
	}

	failed := resultFor(t, resp, "t-0")
	if failed.Error == "" {
		t.Fatal("the item that was refused reports no error")
	}
	if got, want := string(mustJSON(t, failed.Usage)), `{"input_tokens":5,"output_tokens":2}`; got != want {
		t.Errorf("the failed item's usage = %s, want %s: the call that did come back cost something", got, want)
	}
	if got, want := string(mustJSON(t, resp.Usage)), `{"input_tokens":15,"output_tokens":6}`; got != want {
		t.Errorf("batch usage = %s, want %s: a failed item's cost belongs in the total", got, want)
	}
}

// parallel_calls is what reached the provider, not what was planned, and the
// two are equal in every batch where nothing fails — which is every other
// batch in this file, so none of them can tell the two apart.
//
// Here they cannot be equal. One item fails on its first call and abandons the
// rest of its own, so the count is short of the plan by however many were
// abandoned; a concurrency of one makes that exactly five, because no two
// calls of the failing item can be in flight at once and every one after the
// first meets a group that has already been cancelled.
func TestEvaluateBatchParallelCallsCountsWhatWasIssued(t *testing.T) {
	const (
		items   = 2
		perItem = 6
		planned = items * perItem
		// One call of the failing item, and all of the other item's.
		wantIssued = 1 + perItem
	)

	var issued atomic.Int64
	scorer := ScorerFunc(func(_ context.Context, req ScoreRequest) (ScoreResult, error) {
		issued.Add(1)
		if req.State == "state 0" {
			return ScoreResult{}, errors.New("provider refused the request")
		}
		return ScoreResult{Probability: 0.5}, nil
	})

	resp, err := New(scorer, WithMaxConcurrency(1)).EvaluateBatch(context.Background(), BatchRequest{
		Items:     batchOf(items),
		Questions: noulQuestions(perItem),
	})
	if err != nil {
		t.Fatalf("EvaluateBatch returned %v", err)
	}
	if got := issued.Load(); got != wantIssued {
		t.Fatalf("the scorer was entered %d times, want %d: the fixture no longer abandons what it means to", got, wantIssued)
	}
	if resp.Meta.ParallelCalls != wantIssued {
		t.Errorf("Meta.ParallelCalls = %d, want %d — the calls that reached the provider, not the %d that were planned",
			resp.Meta.ParallelCalls, wantIssued, planned)
	}
}

// ----------------------------------------------------------------- stress --

// TestEvaluateBatchStressUnderRace hammers the hand-off between the wave's
// goroutines and the accounting that reads what they wrote, over many rounds
// of a many-item batch whose calls finish in an unpredictable order.
//
// What it can fail on, and what it cannot, are worth stating, because a
// stress fixture is the easiest kind of test to believe too much of.
//
// It can fail on an accounting mistake that only some interleavings produce:
// a result placed against the wrong item, an item's slots read as another's,
// a usage total that misses a late write, meta counts that disagree with the
// results they claim to summarise. Those are what the per-item token counts
// below are for — each item's calls report a count of its own, so a collector
// that reads the wrong item's slots is a wrong number rather than the same
// number twice. It can also fail, under -race, on a genuine data race
// introduced by an edit that reads a group's results before the whole wave
// has finished.
//
// It cannot fail on removing [batchWork]'s mutex. The wave's WaitGroup
// already orders every write before the collector's read, so the lock is not
// what makes the read safe today and -race sees nothing when it goes; the
// lock is there for the edit that stops waiting for the whole wave, and only
// that edit would show it up. The source comment on the field says the same.
//
// It is worth running as `go test ./classifier/ -race -run Stress -count=20`
// and with -cpu varied; the plain suite runs one pass of it.
func TestEvaluateBatchStressUnderRace(t *testing.T) {
	const (
		rounds  = 25
		items   = 12
		perItem = 4
		limit   = 5
	)
	refused := errors.New("provider refused the request")

	// The items whose state ends in 3 or 7 fail — two of the twelve — so each
	// round mixes items that complete with items abandoned part way through.
	failing := func(state string) bool {
		return strings.HasSuffix(state, "3") || strings.HasSuffix(state, "7")
	}

	// Each item's calls report a token count of its own, so that the numbers
	// a round checks can tell one item's results from another's. Uniform
	// counts would make a collector that read the wrong item's slots look
	// exactly like one that read the right ones.
	tokensFor := func(state string) int {
		n, err := strconv.Atoi(strings.TrimPrefix(state, "state "))
		if err != nil {
			t.Errorf("unexpected state %q", state)
		}
		return n + 1
	}

	var counter atomic.Int64
	scorer := ScorerFunc(func(ctx context.Context, req ScoreRequest) (ScoreResult, error) {
		// An uneven, non-deterministic amount of yielding, so the calls of one
		// item finish interleaved with another's rather than in lockstep.
		for range counter.Add(1) % 7 {
			runtime.Gosched()
		}
		if failing(req.State) {
			return ScoreResult{}, refused
		}
		return ScoreResult{Probability: 0.5, InputTokens: tokensFor(req.State), OutputTokens: 1}, nil
	})

	e := New(scorer, WithMaxConcurrency(limit))
	for round := range rounds {
		resp, err := e.EvaluateBatch(context.Background(), BatchRequest{
			Items:     batchOf(items),
			Questions: noulQuestions(perItem),
			Model:     "probe-1",
		})
		if err != nil {
			t.Fatalf("round %d: EvaluateBatch returned %v", round, err)
		}
		if len(resp.Results) != items {
			t.Fatalf("round %d: got %d results, want %d", round, len(resp.Results), items)
		}

		var succeeded, failed, in, out int
		for i, r := range resp.Results {
			if r.Index != i {
				t.Fatalf("round %d: results[%d].Index = %d", round, i, r.Index)
			}
			if r.Error != "" {
				failed++
				if !strings.Contains(r.Error, refused.Error()) {
					t.Fatalf("round %d: item %d failed with %q", round, i, r.Error)
				}
				continue
			}
			succeeded++
			if got := r.Answers.Len(); got != perItem {
				t.Fatalf("round %d: item %d has %d answers, want %d", round, i, got, perItem)
			}
			// Every candidate of a completed item reported its tokens, and
			// they are this item's own, so the arithmetic is exact: a torn
			// write, a missing one, or another item's slots all show up here.
			want := (i + 1) * perItem
			if r.Usage.InputTokens == nil || *r.Usage.InputTokens != want {
				t.Fatalf("round %d: item %d input tokens = %v, want %d", round, i, r.Usage.InputTokens, want)
			}
			in += *r.Usage.InputTokens
			out += *r.Usage.OutputTokens
		}
		if resp.Meta.Succeeded != succeeded || resp.Meta.Failed != failed {
			t.Fatalf("round %d: meta says %d/%d, the results say %d/%d",
				round, resp.Meta.Succeeded, resp.Meta.Failed, succeeded, failed)
		}
		if resp.Meta.Items != items {
			t.Fatalf("round %d: Meta.Items = %d, want %d", round, resp.Meta.Items, items)
		}
		if resp.Usage.InputTokens == nil || *resp.Usage.InputTokens < in {
			t.Fatalf("round %d: the batch total %v is less than the items' %d", round, resp.Usage.InputTokens, in)
		}
	}
}

// The same hammer on the one-shot path, where the item's whole outcome is
// written by its call and read by the collector.
func TestEvaluateBatchOneshotStressUnderRace(t *testing.T) {
	const (
		rounds = 25
		items  = 10
	)

	var counter atomic.Int64
	gen := &yieldingGenerator{counter: &counter}
	e := New(gen, WithMaxConcurrency(4))

	for round := range rounds {
		resp, err := e.EvaluateBatch(context.Background(), BatchRequest{
			Items:     batchOf(items),
			Questions: mustQuestions(t, `{"angry": {"type": "noul"}}`),
			Mode:      ModeOneshot,
		})
		if err != nil {
			t.Fatalf("round %d: EvaluateBatch returned %v", round, err)
		}
		for i, r := range resp.Results {
			if r.Error != "" {
				t.Fatalf("round %d: item %d failed: %s", round, i, r.Error)
			}
			// Reading the answers is the point: this is the field a one-shot
			// call writes from inside the wave.
			answer, ok := r.Answers.Get("angry")
			if !ok || answer.Noul != 0.75 {
				t.Fatalf("round %d: item %d answered %v (present: %v)", round, i, answer.Noul, ok)
			}
			if r.Usage.InputTokens == nil || *r.Usage.InputTokens != 11 {
				t.Fatalf("round %d: item %d input tokens = %v, want 11", round, i, r.Usage.InputTokens)
			}
		}
		if resp.Meta.ParallelCalls != items {
			t.Fatalf("round %d: Meta.ParallelCalls = %d, want %d", round, resp.Meta.ParallelCalls, items)
		}
	}
}

// yieldingGenerator answers every generation the same way after an uneven
// amount of yielding, so that items finish out of order.
type yieldingGenerator struct{ counter *atomic.Int64 }

func (g *yieldingGenerator) Score(context.Context, ScoreRequest) (ScoreResult, error) {
	return ScoreResult{}, errors.New("a one-shot batch must not score")
}

func (g *yieldingGenerator) Generate(context.Context, GenerateRequest) (GenerateResult, error) {
	for range g.counter.Add(1) % 5 {
		runtime.Gosched()
	}
	return GenerateResult{Content: `{"angry": {"noul": 0.75}}`, InputTokens: 11, OutputTokens: 3}, nil
}

// ---------------------------------------------------- whole-request errors --

func TestEvaluateBatchRejectsWholeRequestFaults(t *testing.T) {
	ok := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		return ScoreResult{Probability: 0.5}, nil
	})

	t.Run("no scorer", func(t *testing.T) {
		_, err := New(nil).EvaluateBatch(context.Background(), BatchRequest{
			Items:     batchOf(1),
			Questions: noulQuestions(1),
		})
		if !errors.Is(err, ErrNoScorer) {
			t.Fatalf("EvaluateBatch returned %v, want ErrNoScorer", err)
		}
	})

	t.Run("no items", func(t *testing.T) {
		for _, items := range [][]BatchItem{nil, {}} {
			_, err := New(ok).EvaluateBatch(context.Background(), BatchRequest{
				Items:     items,
				Questions: noulQuestions(1),
			})
			if !errors.Is(err, ErrNoItems) {
				t.Fatalf("EvaluateBatch returned %v, want ErrNoItems", err)
			}
		}
	})

	t.Run("no questions", func(t *testing.T) {
		_, err := New(ok).EvaluateBatch(context.Background(), BatchRequest{Items: batchOf(2)})
		if !errors.Is(err, ErrNoQuestions) {
			t.Fatalf("EvaluateBatch returned %v, want ErrNoQuestions", err)
		}
	})

	// The shared question set is checked once, and a fault in it fails the
	// request rather than every item separately.
	t.Run("an invalid question", func(t *testing.T) {
		scorer := &constantScorer{probability: 0.5}
		_, err := New(scorer).EvaluateBatch(context.Background(), BatchRequest{
			Items:     batchOf(50),
			Questions: one("urgency", levelsOf(1)),
		})
		var verr *ValidationError
		if !errors.As(err, &verr) {
			t.Fatalf("EvaluateBatch returned %v, want a *ValidationError", err)
		}
		if verr.Question != "urgency" || verr.Field != "criteria" {
			t.Fatalf("validation error named %q/%q, want urgency/criteria", verr.Question, verr.Field)
		}
		if got := scorer.callCount(); got != 0 {
			t.Fatalf("a request that does not validate still made %d calls", got)
		}
	})

	t.Run("an unknown mode", func(t *testing.T) {
		_, err := New(ok).EvaluateBatch(context.Background(), BatchRequest{
			Items:     batchOf(1),
			Questions: noulQuestions(1),
			Mode:      "batched",
		})
		if !errors.Is(err, ErrUnknownMode) {
			t.Fatalf("EvaluateBatch returned %v, want ErrUnknownMode", err)
		}
		if !strings.Contains(err.Error(), `"batched"`) {
			t.Errorf("error %q should quote the mode that was asked for", err)
		}
	})

	t.Run("the default mode is parallel", func(t *testing.T) {
		resp, err := New(ok).EvaluateBatch(context.Background(), BatchRequest{
			Items:     batchOf(1),
			Questions: noulQuestions(1),
		})
		if err != nil {
			t.Fatalf("EvaluateBatch returned %v", err)
		}
		if resp.Meta.Mode != ModeParallel {
			t.Fatalf("Meta.Mode = %q, want %q", resp.Meta.Mode, ModeParallel)
		}
	})
}

// A cancelled batch is a whole-request failure: nothing partial is returned,
// and nothing further is dispatched.
func TestEvaluateBatchHonoursContextCancellation(t *testing.T) {
	t.Run("cancelled before the batch", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		scorer := &constantScorer{probability: 0.5}
		resp, err := New(scorer).EvaluateBatch(ctx, BatchRequest{
			Items:     batchOf(10),
			Questions: noulQuestions(4),
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("EvaluateBatch returned %v, want context.Canceled", err)
		}
		if len(resp.Results) != 0 || resp.Meta != nil {
			t.Errorf("a cancelled batch returned a populated response: %+v", resp)
		}
		if got := scorer.callCount(); got != 0 {
			t.Fatalf("the scorer was called %d times under a cancelled context, want 0", got)
		}
	})

	t.Run("cancelled mid-flight", func(t *testing.T) {
		const items, perItem = 8, 8

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		var (
			mu    sync.Mutex
			calls int
			once  sync.Once
		)
		scorer := ScorerFunc(func(ctx context.Context, _ ScoreRequest) (ScoreResult, error) {
			mu.Lock()
			calls++
			mu.Unlock()
			once.Do(func() { go cancel() })
			select {
			case <-ctx.Done():
				return ScoreResult{}, ctx.Err()
			case <-time.After(5 * time.Second):
				return ScoreResult{}, errors.New("was never cancelled")
			}
		})

		_, err := New(scorer, WithMaxConcurrency(4)).EvaluateBatch(ctx, BatchRequest{
			Items:     batchOf(items),
			Questions: noulQuestions(perItem),
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("EvaluateBatch returned %v, want context.Canceled", err)
		}
		mu.Lock()
		issued := calls
		mu.Unlock()
		if issued >= items*perItem {
			t.Fatalf("the scorer was called %d times; a cancelled batch kept dispatching", issued)
		}
	})

	t.Run("a deadline is reported as one", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()

		scorer := ScorerFunc(func(ctx context.Context, _ ScoreRequest) (ScoreResult, error) {
			<-ctx.Done()
			return ScoreResult{}, ctx.Err()
		})
		_, err := New(scorer, WithMaxConcurrency(2)).EvaluateBatch(ctx, BatchRequest{
			Items:     batchOf(4),
			Questions: noulQuestions(2),
		})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("EvaluateBatch returned %v, want context.DeadlineExceeded", err)
		}
	})
}

// The other half of that rule: a cancellation stops a batch only while there
// is something left for it to stop. One that lands after the last call has
// come back discards finished, paid-for work, and one that lands after an item
// has already failed for a reason of its own hides that reason behind a
// timeout.
func TestEvaluateBatchKeepsWhatWasFinishedBeforeTheCancellation(t *testing.T) {
	t.Run("every item finished", func(t *testing.T) {
		const items, perItem = 3, 2

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		var done atomic.Int64
		scorer := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
			if done.Add(1) == items*perItem {
				cancel()
			}
			return ScoreResult{Probability: 0.5, InputTokens: 2, OutputTokens: 1}, nil
		})

		resp, err := New(scorer, WithMaxConcurrency(items*perItem)).EvaluateBatch(ctx, BatchRequest{
			Items:     batchOf(items),
			Questions: noulQuestions(perItem),
		})
		if err != nil {
			t.Fatalf("EvaluateBatch returned %v; every item had come back before the cancellation", err)
		}
		if resp.Meta == nil || resp.Meta.Succeeded != items || resp.Meta.Failed != 0 {
			t.Fatalf("meta = %+v, want %d succeeded and none failed", resp.Meta, items)
		}
		if resp.Usage.InputTokens == nil || *resp.Usage.InputTokens != 2*items*perItem {
			t.Fatalf("batch input tokens = %v, want %d: a cancelled batch threw away what the items cost",
				resp.Usage.InputTokens, 2*items*perItem)
		}
	})

	t.Run("an item's own failure is still its own", func(t *testing.T) {
		const items = 3
		refused := errors.New("provider refused the request")

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		// The first item fails on its only call, and the others wait for that
		// to have happened before finishing: the failure is on the group
		// before anything cancels, so what follows cannot be blamed on it.
		failed := make(chan struct{})
		var once sync.Once
		var done atomic.Int64

		scorer := ScorerFunc(func(_ context.Context, req ScoreRequest) (ScoreResult, error) {
			if req.State == "state 0" {
				once.Do(func() { close(failed) })
				return ScoreResult{}, refused
			}
			<-failed
			if done.Add(1) == items-1 {
				cancel()
			}
			return ScoreResult{Probability: 0.5}, nil
		})

		resp, err := New(scorer, WithMaxConcurrency(items)).EvaluateBatch(ctx, BatchRequest{
			Items:     batchOf(items),
			Questions: noulQuestions(1),
		})
		if err != nil {
			t.Fatalf("EvaluateBatch returned %v; the items that succeeded and the one that failed were all settled before the cancellation", err)
		}
		if got := resultFor(t, resp, "t-0"); !strings.Contains(got.Error, refused.Error()) {
			t.Fatalf("the failed item reports %q, want the provider's own refusal", got.Error)
		}
		if resp.Meta.Succeeded != items-1 || resp.Meta.Failed != 1 {
			t.Fatalf("meta says %d/%d, want %d succeeded and 1 failed",
				resp.Meta.Succeeded, resp.Meta.Failed, items-1)
		}
	})
}

// ---------------------------------------------------------------- oneshot --

// A one-shot batch is one generation per item, taking a slot of the same
// budget a parallel batch's candidates take.
func TestEvaluateBatchOneshot(t *testing.T) {
	const content = `{"angry": {"noul": 0.9}}`

	gen := &countingGenerator{content: content, inputTokens: 40, outputTokens: 8}
	resp, err := New(gen, WithMaxConcurrency(2)).EvaluateBatch(context.Background(), BatchRequest{
		Items:     batchOf(5),
		Questions: mustQuestions(t, `{"angry": {"type": "noul", "instructions": "Angry?"}}`),
		Model:     "probe-1",
		Mode:      ModeOneshot,
	})
	if err != nil {
		t.Fatalf("EvaluateBatch returned %v", err)
	}
	if resp.Meta.Mode != ModeOneshot {
		t.Errorf("Meta.Mode = %q, want %q", resp.Meta.Mode, ModeOneshot)
	}
	// One call per item, not one per candidate.
	if resp.Meta.ParallelCalls != 5 {
		t.Errorf("Meta.ParallelCalls = %d, want 5", resp.Meta.ParallelCalls)
	}
	if got := gen.peak(); got > 2 {
		t.Errorf("saw %d generations in flight at once, want at most the configured 2", got)
	}
	for _, r := range resp.Results {
		if r.Error != "" {
			t.Fatalf("item %d failed: %s", r.Index, r.Error)
		}
		answer, _ := r.Answers.Get("angry")
		if answer.Noul != 0.9 {
			t.Errorf("item %d noul = %v, want 0.9", r.Index, answer.Noul)
		}
	}
	if got, want := string(mustJSON(t, resp.Usage)), `{"input_tokens":200,"output_tokens":40}`; got != want {
		t.Errorf("batch usage = %s, want %s", got, want)
	}
}

// A provider that cannot generate fails every item of a one-shot batch, one
// error at a time, rather than the request.
// Whether the scorer can generate is a property of the Evaluator, not of any
// item, and it will be the same answer for the hundredth state as for the
// first. So it fails the request, like the other whole-request faults, rather
// than arriving as a 200 in which every single item failed for the same
// reason — and no call is made in the meantime.
func TestEvaluateBatchOneshotUnsupported(t *testing.T) {
	var calls atomic.Int64
	ok := ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		calls.Add(1)
		return ScoreResult{Probability: 0.5}, nil
	})
	resp, err := New(ok).EvaluateBatch(context.Background(), BatchRequest{
		Items:     batchOf(2),
		Questions: noulQuestions(1),
		Mode:      ModeOneshot,
	})
	if !errors.Is(err, ErrOneshotUnsupported) {
		t.Fatalf("EvaluateBatch returned %v, want ErrOneshotUnsupported", err)
	}
	if len(resp.Results) != 0 || resp.Meta != nil {
		t.Errorf("a request no item could have served returned a populated response: %+v", resp)
	}
	if n := calls.Load(); n != 0 {
		t.Errorf("the scorer was called %d times for a mode it cannot serve", n)
	}
}

// countingGenerator is a Scorer that generates, counting how many generations
// it has in flight at once.
type countingGenerator struct {
	content      string
	inputTokens  int
	outputTokens int

	mu          sync.Mutex
	inFlight    int
	maxInFlight int
}

func (g *countingGenerator) Score(context.Context, ScoreRequest) (ScoreResult, error) {
	return ScoreResult{}, errors.New("a one-shot batch must not score")
}

func (g *countingGenerator) Generate(context.Context, GenerateRequest) (GenerateResult, error) {
	g.mu.Lock()
	g.inFlight++
	if g.inFlight > g.maxInFlight {
		g.maxInFlight = g.inFlight
	}
	g.mu.Unlock()

	time.Sleep(10 * time.Millisecond)

	g.mu.Lock()
	g.inFlight--
	g.mu.Unlock()
	return GenerateResult{Content: g.content, InputTokens: g.inputTokens, OutputTokens: g.outputTokens}, nil
}

func (g *countingGenerator) peak() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.maxInFlight
}

// TestEvaluateBatchHonoursTheSoftmaxTemperature checks the batch path
// normalises at the configured temperature too. It folds its items' waves
// through the same function a single evaluation does, but from its own call
// site, so a plumbing change that reached one could miss the other.
//
// The three options score 0.9, 0.6 and 0.2 — distinct, because equal scores
// are uniform at every temperature and would report the same distribution
// whether the temperature arrived or not.
func TestEvaluateBatchHonoursTheSoftmaxTemperature(t *testing.T) {
	const fitted = 5.04

	run := func(opts ...Option) []float64 {
		t.Helper()
		resp, err := New(&scriptedScorer{t: t, script: temperatureScript}, opts...).
			EvaluateBatch(context.Background(), BatchRequest{
				Items:     []BatchItem{{ID: "a", State: StringState("s")}},
				Questions: mustQuestions(t, temperatureQuestions),
			})
		if err != nil {
			t.Fatalf("EvaluateBatch returned %v", err)
		}
		if len(resp.Results) != 1 || resp.Results[0].Error != "" {
			t.Fatalf("batch result: %+v", resp.Results)
		}
		answer, ok := resp.Results[0].Answers.Get("pick")
		if !ok {
			t.Fatal("the item has no answer for the question")
		}
		if answer.Choice != "high" {
			t.Fatalf("choice = %q, want %q at every temperature", answer.Choice, "high")
		}
		return probabilitiesOf(t, answer, "high", "middle", "low")
	}

	// The same literals the single-evaluation test pins, so the two paths are
	// held to one set of numbers rather than to each other.
	base, warm := run(), run(WithSoftmaxTemperature(fitted))
	if want := []float64{0.837209, 0.139535, 0.023256}; !equalFloats(base, want) {
		t.Fatalf("a batch at the default reported %v, want %v", base, want)
	}
	if want := []float64{0.456212, 0.319721, 0.224066}; !equalFloats(warm, want) {
		t.Fatalf("a batch at temperature %v reported %v, want %v", fitted, warm, want)
	}
}

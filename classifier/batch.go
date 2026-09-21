package classifier

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// ErrNoItems is returned when a batch carries no states to evaluate. It is the
// counterpart of [ErrNoQuestions]: a batch needs both halves.
var ErrNoItems = errors.New("classifier: at least one item is required")

// BatchItem is one state to be evaluated against the batch's shared questions.
type BatchItem struct {
	// ID is the caller's own identifier, echoed back on the matching result
	// so that results can be reassociated without relying on order. It is
	// optional; results are returned in request order either way.
	ID string `json:"id,omitempty"`
	// State is the material the questions are asked about.
	State State `json:"state"`
}

// BatchRequest evaluates many states against one set of questions.
//
// The point of the batch is that the whole job shares one concurrency budget
// rather than one per item, which is what lets a hundred states be classified
// without a hundred independent fan-outs competing for the same provider.
type BatchRequest struct {
	Items     []BatchItem `json:"items"`
	Questions Questions   `json:"questions"`

	Model       string  `json:"model,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	Mode        Mode    `json:"mode,omitempty"`
}

// BatchResult is one item's outcome. Exactly one of Answers and Error is
// meaningful.
type BatchResult struct {
	// ID echoes the item's identifier.
	ID string `json:"id,omitempty"`
	// Index is the item's position in the request, always present, so a
	// result can be placed even when no ID was given.
	Index int `json:"index"`
	// Answers is the evaluation, when it succeeded.
	Answers Answers `json:"answers,omitempty"`
	// Usage is what this item cost.
	Usage Usage `json:"usage"`
	// Error says why this item has no answers. It is per-item on purpose:
	// one state that fails must not discard the ones that succeeded.
	//
	// This is the opposite of a single evaluation, where the first error
	// cancels the whole wave, and the difference is deliberate. Within one
	// evaluation every candidate feeds the same normalisation, so a missing
	// one corrupts the answer; across a batch the items are independent and
	// a partial result is a useful result.
	Error string `json:"error,omitempty"`
}

// BatchMeta describes how a batch ran.
type BatchMeta struct {
	Mode Mode `json:"mode"`
	// LatencyMS is the elapsed time for the whole batch.
	LatencyMS int64 `json:"latency_ms"`
	// ParallelCalls is the total scorer calls issued across every item.
	ParallelCalls int `json:"parallel_calls"`
	// Items, Succeeded and Failed count the outcomes, so a caller can tell
	// a wholly successful batch from a partly successful one without
	// walking the results.
	Items     int `json:"items"`
	Succeeded int `json:"succeeded"`
	Failed    int `json:"failed"`
}

// BatchResponse is the result of a batch, in request order.
type BatchResponse struct {
	Model   string        `json:"model"`
	Results []BatchResult `json:"results"`
	// Usage totals every item's usage, failed ones included: a call that
	// failed late still cost what it cost.
	Usage Usage      `json:"usage"`
	Meta  *BatchMeta `json:"meta,omitempty"`
}

// batchWork is one item's place in the wave: the group carrying its calls, the
// slots its scores land in, and what they came to.
//
// Two different things are written from inside the wave and read after it.
// results is written slot by slot, each by the one call that owns that slot,
// and is read only once every call of every group has returned — the
// WaitGroup inside [Evaluator.runWave] is the edge, and it is a wave-wide one
// rather than a per-item one, so it holds however the items interleaved.
// outcome is written by the single call of a one-shot item, and takes a lock
// of its own for the reason waveGroup.err does: so that the field cannot be
// read unsynchronised even by an edit that stops waiting for the whole wave.
type batchWork struct {
	index   int
	group   *waveGroup
	results [][]ScoreResult

	mu      sync.Mutex
	outcome evaluation
}

// setOutcome records what a one-shot item's single call produced.
func (w *batchWork) setOutcome(ev evaluation) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.outcome = ev
}

// result reports what the item produced.
func (w *batchWork) result() evaluation {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.outcome
}

// EvaluateBatch answers the same questions about many states, sharing one
// concurrency budget across the whole job.
//
// The items are independent, so their failures are too: an item whose calls
// failed gets its [BatchResult.Error] set and the rest of the batch runs on.
// An error is returned only for something that invalidates the whole request —
// no items, no questions, a question set that does not validate, an unknown
// mode, no [Scorer], or a cancelled context — because none of those could have
// produced a useful result for any item.
//
// Every candidate of every item is one wave. That is the difference from
// calling [Evaluator.Evaluate] in a loop, where each call would fan out to the
// configured limit on its own: here the limit bounds the job.
func (e *Evaluator) EvaluateBatch(ctx context.Context, req BatchRequest) (BatchResponse, error) {
	if e == nil || e.scorer == nil {
		return BatchResponse{}, ErrNoScorer
	}
	now := e.now
	if now == nil {
		now = time.Now
	}
	start := now()

	mode := req.Mode
	if mode == "" {
		mode = ModeParallel
	}
	if mode != ModeParallel && mode != ModeOneshot {
		return BatchResponse{}, fmt.Errorf("%w %q; expected %q or %q",
			ErrUnknownMode, req.Mode, ModeParallel, ModeOneshot)
	}

	if len(req.Items) == 0 {
		return BatchResponse{}, ErrNoItems
	}
	// Once for the batch, not once per item: the question set is shared, so a
	// per-item check would report the same fault as many times as there are
	// states and charge the caller for the walk each time.
	if err := Validate(req.Questions); err != nil {
		return BatchResponse{}, err
	}

	results := make([]BatchResult, len(req.Items))
	work := make([]*batchWork, 0, len(req.Items))
	groups := make([]*waveGroup, 0, len(req.Items))

	for i, item := range req.Items {
		results[i] = BatchResult{ID: item.ID, Index: i}

		// A state is the one part of a batch that is the item's own, so a bad
		// one is the item's failure and not the request's.
		if item.State.IsZero() || item.State.IsNull() {
			results[i].Error = `classifier: item has no "state"`
			continue
		}
		state, err := item.State.Text()
		if err != nil {
			results[i].Error = fmt.Sprintf("classifier: rendering state: %v", err)
			continue
		}

		w := &batchWork{index: i}
		if mode == ModeOneshot {
			// One generate call per item, taking a slot of the same budget the
			// candidates of a parallel batch take: the point of the ceiling is
			// how much of the provider one request may occupy, and a one-shot
			// call occupies it too.
			w.group = &waveGroup{calls: []func(context.Context) error{
				func(ctx context.Context) error {
					out, err := e.evaluateOneshot(ctx, Request{
						Questions:   req.Questions,
						Model:       req.Model,
						Temperature: req.Temperature,
					}, state)
					if err != nil {
						return err
					}
					w.setOutcome(out)
					return nil
				},
			}}
		} else {
			tasks, slots, err := parallelTasks(req.Questions)
			if err != nil {
				// The question set is the batch's, not this item's, so a
				// block that will not render fails every item alike. It is
				// still recorded per item rather than failing the request:
				// the caller reads the same reason on each, in the place
				// they are already looking.
				results[i].Error = err.Error()
				continue
			}
			w.results = slots
			w.group = &waveGroup{calls: e.scoreCalls(req.Model, req.Temperature, state, tasks, slots)}
		}
		work = append(work, w)
		groups = append(groups, w.group)
	}

	if err := e.runWave(ctx, groups); err != nil {
		return BatchResponse{}, err
	}

	meta := BatchMeta{Mode: mode, Items: len(req.Items)}
	var totalIn, totalOut int
	for _, w := range work {
		r := &results[w.index]
		// The calls that did reach the provider, not the ones that were
		// planned: an item abandoned after its first failure still issued
		// some, and a caller reading this number is asking what was spent.
		issued := int(w.group.issued.Load())
		meta.ParallelCalls += issued

		if mode == ModeOneshot {
			if err := w.group.failure(); err != nil {
				r.Error = err.Error()
				continue
			}
			out := w.result()
			r.Answers = out.answers
			r.Usage = usage(out.inputTokens, out.outputTokens)
			totalIn += out.inputTokens
			totalOut += out.outputTokens
			continue
		}

		if err := w.group.failure(); err != nil {
			r.Error = err.Error()
			// A wave that failed part way through still paid for the calls
			// that came back before it did.
			in, out := tokensOf(w.results)
			r.Usage = usage(in, out)
			totalIn += in
			totalOut += out
			continue
		}
		out := parallelEvaluation(req.Questions, w.results, issued)
		r.Answers = out.answers
		r.Usage = usage(out.inputTokens, out.outputTokens)
		totalIn += out.inputTokens
		totalOut += out.outputTokens
	}

	for _, r := range results {
		if r.Error == "" {
			meta.Succeeded++
		}
	}
	meta.Failed = len(results) - meta.Succeeded
	meta.LatencyMS = now().Sub(start).Milliseconds()

	return BatchResponse{
		Model:   req.Model,
		Results: results,
		Usage:   usage(totalIn, totalOut),
		Meta:    &meta,
	}, nil
}

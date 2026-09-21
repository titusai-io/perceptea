package calibration

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/titusai-io/perceptea/classifier"
)

// ErrNoEvaluator is returned by [Runner.Run] when the runner has nothing to
// evaluate with.
var ErrNoEvaluator = errors.New("calibration: no Evaluator configured")

// Evaluator answers one request. A *[classifier.Evaluator] satisfies it, and
// so does a fake, which is how the runner is tested without a key.
type Evaluator interface {
	Evaluate(ctx context.Context, req classifier.Request) (classifier.Response, error)
}

// Outcome is one attempt at one case: the answer it produced, or the reason
// there is none.
//
// A failed attempt is kept rather than discarded because the count of them is
// half of what the report is for. An answer whose numbers are computed from
// two thirds of the dataset is a different claim from one computed from all
// of it, and only an outcome that survives can say so.
type Outcome struct {
	// Case is the labelled case this attempt was made against.
	Case Case
	// Attempt is the 1-based repeat number, so that a dataset run several
	// times can tell which pass a failure came from.
	Attempt int
	// Answer is what the service returned. It is meaningful only when Err is
	// nil.
	Answer classifier.Answer
	// Err is why the attempt produced no answer.
	Err error
}

// Run is everything one pass over a dataset produced.
type Run struct {
	// Dataset names the file the cases came from, carried through so that the
	// report says what it measured.
	Dataset string
	// Model is the model the run asked for, which may be empty when the
	// choice was left to the provider client.
	Model string
	// Cases is how many labelled cases were run, before repeats.
	Cases int
	// Repeat is how many times each case was evaluated.
	Repeat int
	// Outcomes holds one entry per attempt, in dataset order, each case's
	// repeats together.
	Outcomes []Outcome
}

// Runner evaluates a dataset of labelled cases.
type Runner struct {
	// Evaluator answers each case. Required.
	Evaluator Evaluator
	// Model and Temperature are put on every request, exactly as a caller
	// would put them on a request body.
	Model       string
	Temperature float64
	// Repeat evaluates every case this many times. Zero or less means once.
	// More than once is how a run at a non-zero temperature measures the
	// spread rather than one sample of it.
	Repeat int
	// Concurrency is how many cases may be in flight at once. Zero or less
	// means one at a time.
	//
	// This is cases, not provider calls: each case fans out into one call per
	// declared candidate of its own accord. Bounding the total spend is the
	// job of the [Scorer] underneath — see [Throttle] — because that is the
	// only place that sees every call.
	Concurrency int
	// Timeout bounds one attempt. Zero means the run's context is the only
	// deadline.
	Timeout time.Duration
}

// attempt names one unit of work: which case, and which repeat of it.
type attempt struct {
	caseIndex int
	number    int
}

// Run evaluates every case, Repeat times each.
//
// A case that fails is recorded and the run continues: one unreachable
// provider call must not cost the other ninety-nine measurements. A cancelled
// context is different — it means the run itself is over — so it stops the
// remaining work and is returned alongside whatever had already been
// collected.
func (r Runner) Run(ctx context.Context, cases []Case) (Run, error) {
	if r.Evaluator == nil {
		return Run{}, ErrNoEvaluator
	}
	repeat := max(r.Repeat, 1)
	workers := max(r.Concurrency, 1)

	run := Run{Model: r.Model, Cases: len(cases), Repeat: repeat}
	if len(cases) == 0 {
		return run, nil
	}

	attempts := make([]attempt, 0, len(cases)*repeat)
	for i := range cases {
		for n := 1; n <= repeat; n++ {
			attempts = append(attempts, attempt{caseIndex: i, number: n})
		}
	}
	// Written by index rather than collected from a channel so that the
	// report is in dataset order however the workers interleaved. A benchmark
	// whose output reorders itself between runs cannot be diffed.
	outcomes := make([]Outcome, len(attempts))

	next := make(chan int)
	var wg sync.WaitGroup
	for range min(workers, len(attempts)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				a := attempts[i]
				outcomes[i] = r.evaluate(ctx, cases[a.caseIndex], a.number)
			}
		}()
	}

	var dispatchErr error
dispatch:
	for i := range attempts {
		// Checked before the select rather than only inside it: with both a
		// waiting worker and a cancelled context ready, a select picks between
		// them at random, and a run that sometimes dispatched several more
		// attempts after a Ctrl-C would be a run nobody could reproduce.
		if err := ctx.Err(); err != nil {
			dispatchErr = err
			break dispatch
		}
		select {
		case next <- i:
		case <-ctx.Done():
			dispatchErr = ctx.Err()
			break dispatch
		}
	}
	close(next)
	wg.Wait()

	// An attempt that was never dispatched has a zero Outcome, which would
	// read as a success with an empty answer. Drop those rather than report
	// them: they are work that did not happen. The repeat number is what
	// identifies them, because it is set on every outcome the worker built
	// and is 1-based, so no dispatched attempt can be mistaken for one.
	run.Outcomes = make([]Outcome, 0, len(outcomes))
	for _, o := range outcomes {
		if o.Attempt == 0 {
			continue
		}
		run.Outcomes = append(run.Outcomes, o)
	}
	return run, dispatchErr
}

// evaluate makes one attempt and turns anything that went wrong into an
// outcome rather than an error, so that the caller keeps going.
func (r Runner) evaluate(ctx context.Context, c Case, number int) Outcome {
	out := Outcome{Case: c, Attempt: number}

	if r.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.Timeout)
		defer cancel()
	}

	resp, err := r.Evaluator.Evaluate(ctx, c.Request(r.Model, r.Temperature))
	if err != nil {
		out.Err = err
		return out
	}
	answer, ok := resp.Answers.Get(c.Name)
	if !ok {
		out.Err = fmt.Errorf("the evaluation returned no answer for question %q", c.Name)
		return out
	}
	// A type mismatch here would silently score the wrong metric — a choice
	// label read against a noul probability — so it is a failed case, not a
	// number to be quietly folded in.
	if answer.Type != c.Question.Type {
		out.Err = fmt.Errorf("question %q is a %q but the evaluation answered with a %q",
			c.Name, c.Question.Type, answer.Type)
		return out
	}
	out.Answer = answer
	return out
}

// Throttle bounds how many [classifier.Scorer] calls are in flight across
// everything that shares the returned scorer. A limit of less than one
// returns s unchanged.
//
// It exists because the bound that matters to a provider is the one over the
// whole run, and an evaluator's own concurrency limit is per evaluation: ten
// cases in flight, each allowed eight calls, is eighty calls however the
// evaluator is configured. Wrapping the scorer is the only place that sees
// every call and so the only place the real limit can be applied.
func Throttle(s classifier.Scorer, limit int) classifier.Scorer {
	if limit < 1 {
		return s
	}
	return &throttled{inner: s, sem: make(chan struct{}, limit)}
}

// throttled holds a buffered channel as a semaphore: a send takes a slot, a
// receive returns it.
type throttled struct {
	inner classifier.Scorer
	sem   chan struct{}
}

// Score waits for a slot, then calls the wrapped scorer. A context that is
// done while waiting gives up rather than queueing for a call nobody is
// waiting for any more.
func (t *throttled) Score(ctx context.Context, req classifier.ScoreRequest) (classifier.ScoreResult, error) {
	select {
	case t.sem <- struct{}{}:
		defer func() { <-t.sem }()
	case <-ctx.Done():
		return classifier.ScoreResult{}, ctx.Err()
	}
	return t.inner.Score(ctx, req)
}

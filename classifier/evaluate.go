package classifier

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// DefaultMaxConcurrency bounds how many scorer calls an evaluation has in
// flight when no other limit is configured.
const DefaultMaxConcurrency = 8

// options holds the settings an [Option] can change.
type options struct {
	maxConcurrency int
	now            func() time.Time
}

// Option configures an [Evaluator].
type Option func(*options)

// WithMaxConcurrency bounds the number of concurrent [Scorer] calls a single
// evaluation makes. Zero or negative means unlimited; the default is
// [DefaultMaxConcurrency].
func WithMaxConcurrency(n int) Option {
	return func(o *options) { o.maxConcurrency = n }
}

// WithClock replaces the source of the current time, which is used only to
// measure [Meta.LatencyMS]. A nil function is ignored.
func WithClock(now func() time.Time) Option {
	return func(o *options) {
		if now != nil {
			o.now = now
		}
	}
}

// Evaluator answers declared questions about a state by putting one statement
// at a time to a [Scorer].
//
// An Evaluator is safe for concurrent use as long as its Scorer is.
type Evaluator struct {
	scorer         Scorer
	maxConcurrency int
	now            func() time.Time
}

// New returns an Evaluator backed by s.
//
// A nil Scorer is accepted rather than panicking here; [Evaluator.Evaluate]
// then returns [ErrNoScorer], which keeps a misconfiguration a request-time
// error instead of a crash at wiring time.
func New(s Scorer, opts ...Option) *Evaluator {
	cfg := options{maxConcurrency: DefaultMaxConcurrency, now: time.Now}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return &Evaluator{scorer: s, maxConcurrency: cfg.maxConcurrency, now: cfg.now}
}

// Evaluate answers every question in req.
//
// In [ModeParallel], the default, every candidate answer of every question is
// scored independently and concurrently — one wave, bounded by the configured
// concurrency — and the independent probabilities are then normalised per
// question. The first scorer error cancels the rest of the wave and is
// returned, in preference to a ctx that expired while the rest of the wave was
// unwinding: the scorer's own error is the one that says what went wrong. A
// ctx cancelled while calls are still outstanding is returned as itself, and
// one cancelled after the last call came back is not returned at all — the
// answers are complete and there is nothing left for it to stop.
//
// In [ModeOneshot] the whole request goes to the model as one prompt, which
// needs a Scorer that also implements [Generator].
func (e *Evaluator) Evaluate(ctx context.Context, req Request) (Response, error) {
	if e == nil || e.scorer == nil {
		return Response{}, ErrNoScorer
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
		return Response{}, fmt.Errorf("%w %q; expected %q or %q",
			ErrUnknownMode, req.Mode, ModeParallel, ModeOneshot)
	}

	if err := Validate(req.Questions); err != nil {
		return Response{}, err
	}

	state, err := req.State.Text()
	if err != nil {
		return Response{}, fmt.Errorf("classifier: rendering state: %w", err)
	}

	var outcome evaluation
	if mode == ModeParallel {
		outcome, err = e.evaluateParallel(ctx, req, state)
	} else {
		outcome, err = e.evaluateOneshot(ctx, req, state)
	}
	if err != nil {
		return Response{}, err
	}

	return Response{
		Model:   req.Model,
		Answers: outcome.answers,
		Usage:   usage(outcome.inputTokens, outcome.outputTokens),
		Meta: &Meta{
			Mode:          mode,
			LatencyMS:     now().Sub(start).Milliseconds(),
			ParallelCalls: outcome.calls,
		},
	}, nil
}

// evaluation is what a mode produces before it is dressed up as a [Response].
type evaluation struct {
	answers      Answers
	inputTokens  int
	outputTokens int
	calls        int
}

// usage reports token counts, leaving a field null when nothing was counted. A
// zero and an unreported count are the same thing here: no call that reached a
// provider costs nothing, so a zero means the provider said nothing rather
// than that the evaluation was free.
func usage(in, out int) Usage {
	u := Usage{}
	if in != 0 {
		v := in
		u.InputTokens = &v
	}
	if out != 0 {
		v := out
		u.OutputTokens = &v
	}
	return u
}

// candidate is one statement waiting to be scored, tagged with where its
// result belongs.
type candidate struct {
	question  int
	index     int
	statement string
	// examples is the question's worked examples, already rendered as one
	// block. Every candidate of a question carries the same bytes, which is
	// what lets a backend keep the part of the prompt holding them identical
	// across the whole wave.
	examples string
}

// parallelTasks lists every candidate of every question, in declaration
// order, tagged with the slot its score goes in.
//
// The list depends on the questions alone and not on the state they are asked
// about, so a batch builds it once and reuses it for every item; see
// [resultSlots] for the part that is the item's own.
//
// The error is defensive. [Validate] renders every example's state before any
// of this runs and rejects a question whose examples cannot be shown, so a
// caller that validated — and both entry points do — cannot reach it.
func parallelTasks(qs Questions) ([]candidate, error) {
	var tasks []candidate
	for qi, name := range qs.Keys() {
		q, _ := qs.Get(name)
		// Rendered once per question, not once per candidate: the block is
		// the same for all of them, and rendering it per candidate would
		// invite an accidental difference into the part of the prompt whose
		// whole value is being identical.
		examples, err := ExamplesBlock(q)
		if err != nil {
			return nil, fmt.Errorf("classifier: question %q: rendering examples: %w", name, err)
		}
		for ci, statement := range statements(name, q) {
			tasks = append(tasks, candidate{question: qi, index: ci, statement: statement, examples: examples})
		}
	}
	return tasks, nil
}

// resultSlots allocates one slice of results per question and one slot per
// candidate, which is what the tasks' question and index fields address.
//
// This is the part of a wave that belongs to one state rather than to the
// question set: every item of a batch scores the same candidates and needs
// its own slots to put the scores in.
func resultSlots(qs Questions, tasks []candidate) [][]ScoreResult {
	counts := make([]int, qs.Len())
	for _, t := range tasks {
		counts[t.question]++
	}
	results := make([][]ScoreResult, len(counts))
	for qi, n := range counts {
		results[qi] = make([]ScoreResult, n)
	}
	return results
}

// tokensOf sums what every completed call reported. A wave that failed part
// way through leaves the rest of the slots zero, which is the same arithmetic:
// a call that never happened cost nothing.
func tokensOf(results [][]ScoreResult) (in, out int) {
	for _, question := range results {
		for _, r := range question {
			in += r.InputTokens
			out += r.OutputTokens
		}
	}
	return in, out
}

// parallelEvaluation folds a completed wave's results into an evaluation,
// normalising each question's candidates into its answer.
func parallelEvaluation(qs Questions, results [][]ScoreResult, calls int) evaluation {
	in, out := tokensOf(results)
	ev := evaluation{answers: *NewOrderedMap[Answer](), inputTokens: in, outputTokens: out, calls: calls}
	for qi, name := range qs.Keys() {
		q, _ := qs.Get(name)
		ev.answers.Set(name, answerFor(q, results[qi]))
	}
	return ev
}

// evaluateParallel scores every candidate of every question in one wave and
// normalises each question's results into an answer.
func (e *Evaluator) evaluateParallel(ctx context.Context, req Request, state string) (evaluation, error) {
	tasks, err := parallelTasks(req.Questions)
	if err != nil {
		return evaluation{}, err
	}
	results := resultSlots(req.Questions, tasks)
	group := &waveGroup{calls: e.scoreCalls(req.Model, req.Temperature, state, tasks, results)}

	if err := e.runWave(ctx, []*waveGroup{group}); err != nil {
		return evaluation{}, err
	}
	// One group, so its failure is the evaluation's: all or nothing, because
	// every candidate's score feeds the same normalisation and a missing one
	// would quietly skew the rest.
	if err := group.failure(); err != nil {
		return evaluation{}, err
	}
	return parallelEvaluation(req.Questions, results, len(tasks)), nil
}

// scoreCalls turns candidates into the calls that score them, each writing its
// own slot and none reading another's.
func (e *Evaluator) scoreCalls(model string, temperature float64, state string, tasks []candidate, results [][]ScoreResult) []func(context.Context) error {
	calls := make([]func(context.Context) error, 0, len(tasks))
	for _, task := range tasks {
		calls = append(calls, func(ctx context.Context) error {
			res, err := e.scorer.Score(ctx, ScoreRequest{
				Model:       model,
				State:       state,
				Statement:   task.statement,
				Examples:    task.examples,
				Temperature: temperature,
			})
			if err != nil {
				return err
			}
			results[task.question][task.index] = res
			return nil
		})
	}
	return calls
}

// waveGroup is one evaluation's worth of work inside a wave: the calls to
// issue, and what came of them.
//
// A group is the unit of failure. The calls within one group feed a single
// normalisation, so the first error abandons the rest of that group — and only
// that group. A wave may carry many independent evaluations, and one state
// that fails must not discard the ones that succeeded; see [BatchResult.Error].
type waveGroup struct {
	calls []func(context.Context) error

	// mu guards err, which the group's calls write from their own goroutines
	// and the collector reads once the wave is over.
	//
	// The wave's [sync.WaitGroup] already orders those two, so the lock is not
	// what makes the read safe today. It is here so that the field cannot be
	// read unsynchronised at all: the ordering that protects it otherwise
	// lives in another function, and an edit that reads a group's failure
	// before the whole wave has finished — to report an item the moment it is
	// done, say — would be a race with nothing nearby to say so.
	mu  sync.Mutex
	err error

	// issued counts the calls that actually reached the provider, which is
	// fewer than len(calls) for a group abandoned part way through.
	issued atomic.Int64
}

// fail records the first error one of the group's calls reported, and reports
// whether this was that first one. Later errors are dropped: they are almost
// always the cancellation the first one caused.
func (g *waveGroup) fail(err error) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.err != nil {
		return false
	}
	g.err = err
	return true
}

// failure reports the error that abandoned this group, or nil.
func (g *waveGroup) failure() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.err
}

// runWave issues every call of every group concurrently, at most
// maxConcurrency at a time — one budget for the whole wave, however many
// groups it carries. That is what lets a hundred states be classified without
// a hundred independent fan-outs competing for the same provider.
//
// A call's own failure is recorded on its group and cancels the rest of that
// group; its caller reads it from the group, where a batch can report it
// against the one item it belongs to. The wave's own error is reserved for a
// cancellation from outside that actually stopped work, and is returned only
// after every goroutine has finished; see [stoppedByContext].
func (e *Evaluator) runWave(ctx context.Context, groups []*waveGroup) error {
	total := 0
	for _, g := range groups {
		total += len(g.calls)
	}
	if total == 0 {
		return nil
	}

	// A buffered channel is the semaphore: a send takes a slot, a receive
	// returns it. Nil means unlimited, and a send on a nil channel blocks
	// forever, so the nil case is branched around rather than selected on.
	var sem chan struct{}
	if e.maxConcurrency > 0 {
		sem = make(chan struct{}, e.maxConcurrency)
	}

	var wg sync.WaitGroup

	for _, g := range groups {
		// Each group is cancelled on its own, so that one failure abandons its
		// own remaining calls and nobody else's. Cancelling ctx still stops
		// every group at once, because every group's context descends from it.
		groupCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		fail := func(err error) {
			if g.fail(err) {
				cancel()
			}
		}

		for _, call := range g.calls {
			wg.Add(1)
			go func() {
				defer wg.Done()

				if sem != nil {
					select {
					case sem <- struct{}{}:
						defer func() { <-sem }()
					case <-groupCtx.Done():
						fail(context.Cause(groupCtx))
						return
					}
				}
				// Cancelled while queued, or before the wave started: skip the
				// call rather than issue one that is already pointless.
				select {
				case <-groupCtx.Done():
					fail(context.Cause(groupCtx))
					return
				default:
				}

				g.issued.Add(1)
				if err := call(groupCtx); err != nil {
					fail(err)
				}
			}()
		}
	}

	wg.Wait()

	// Read after every goroutine has stopped, so a cancelled wave reports the
	// cancellation rather than whichever group noticed it first — and only
	// when the cancellation is what stopped something.
	if ctx.Err() != nil && stoppedByContext(groups) {
		return context.Cause(ctx)
	}
	return nil
}

// stoppedByContext reports whether the wave's context is what abandoned any of
// these groups, which is the only thing that makes it the wave's failure
// rather than an item's.
//
// Every call either filled its slot or recorded a failure on its group, so the
// groups are the whole record of what happened and the context adds nothing to
// it. A group that recorded no failure produced every answer it was asked for,
// even if the caller's deadline landed a moment later; a group that recorded a
// failure of its own has a reason worth reporting, and one that names a
// setting to change is worth far more to whoever reads it than the deadline
// that expired while its siblings were unwinding. Returning the context's
// cause regardless would discard both.
func stoppedByContext(groups []*waveGroup) bool {
	for _, g := range groups {
		err := g.failure()
		if err == nil {
			continue
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return true
		}
	}
	return false
}

// answerFor normalises one question's independent scores into its answer.
func answerFor(q Question, results []ScoreResult) Answer {
	raw := make([]float64, len(results))
	for i, r := range results {
		// A Scorer is an interface, so its answer is input, not a value this
		// package produced. Everything downstream may assume [0,1].
		raw[i] = sanitiseProbability(r.Probability)
	}

	switch q.Type {
	case TypeChoice:
		probs := scoresToDistribution(raw)
		keys := q.Options.Keys()
		distribution := NewOrderedMap[float64]()
		for i, key := range keys {
			distribution.Set(key, round3(probs[i]))
		}
		answer := Answer{
			Type:          TypeChoice,
			Confidence:    confidence(probs),
			Probabilities: *distribution,
		}
		if len(keys) > 0 {
			answer.Choice = keys[argmax(probs)]
		}
		return answer

	case TypeScore:
		probs := scoresToDistribution(raw)
		score := 0.0
		legend := NewOrderedMap[string]()
		distribution := NewOrderedMap[float64]()
		// probs has exactly one entry per declared level: the wave scored one
		// statement per level, so the two lengths cannot differ.
		for i, p := range probs {
			// The conversion is a barrier against a fused multiply-add, which
			// the compiler really does emit here on arm64. Fusing rounds once
			// where a separate multiply and add round twice, so without the
			// barrier the same inputs would sum differently depending on the
			// architecture the binary was built for; the two sums differ in
			// roughly one random case in eight. No observed difference has
			// ever survived round2, but the barrier is what makes that a
			// measurement rather than a hope.
			score += float64(p * float64(i))
			key := strconv.Itoa(i)
			legend.Set(key, q.Levels[i])
			distribution.Set(key, round3(p))
		}
		return Answer{
			Type:          TypeScore,
			Score:         round2(score),
			Confidence:    confidence(probs),
			Legend:        *legend,
			Probabilities: *distribution,
		}

	default:
		// A noul answer is the raw probability: there is nothing to normalise
		// against and so no confidence to report.
		var p float64
		if len(raw) > 0 {
			p = raw[0]
		}
		return Answer{Type: TypeNoul, Noul: round3(p)}
	}
}

package classifier

import (
	"context"
	"fmt"
	"strconv"
	"sync"
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
// returned; so is a cancelled ctx.
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
}

// evaluateParallel scores every candidate of every question in one wave and
// normalises each question's results into an answer.
func (e *Evaluator) evaluateParallel(ctx context.Context, req Request, state string) (evaluation, error) {
	names := req.Questions.Keys()

	results := make([][]ScoreResult, len(names))
	var tasks []candidate
	for qi, name := range names {
		q, _ := req.Questions.Get(name)
		stmts := statements(name, q)
		results[qi] = make([]ScoreResult, len(stmts))
		for ci, statement := range stmts {
			tasks = append(tasks, candidate{question: qi, index: ci, statement: statement})
		}
	}

	if err := e.runWave(ctx, req, state, tasks, results); err != nil {
		return evaluation{}, err
	}

	out := evaluation{answers: *NewOrderedMap[Answer](), calls: len(tasks)}
	for qi, name := range names {
		q, _ := req.Questions.Get(name)
		for _, r := range results[qi] {
			out.inputTokens += r.InputTokens
			out.outputTokens += r.OutputTokens
		}
		out.answers.Set(name, answerFor(q, results[qi]))
	}
	return out, nil
}

// runWave issues every task concurrently, at most maxConcurrency at a time,
// writing each result into its own slot. It returns the first error any call
// reported — or the context's, if it was cancelled — after cancelling the
// remaining work and waiting for the goroutines to finish. It is all or
// nothing: one failure fails the evaluation, because every candidate's score
// feeds the same normalisation and a missing one would quietly skew the rest.
func (e *Evaluator) runWave(ctx context.Context, req Request, state string, tasks []candidate, results [][]ScoreResult) error {
	if len(tasks) == 0 {
		return nil
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// A buffered channel is the semaphore: a send takes a slot, a receive
	// returns it. Nil means unlimited, and a send on a nil channel blocks
	// forever, so the nil case is branched around rather than selected on.
	var sem chan struct{}
	if e.maxConcurrency > 0 {
		sem = make(chan struct{}, e.maxConcurrency)
	}

	var (
		wg       sync.WaitGroup
		mu       sync.Mutex
		firstErr error
	)
	fail := func(err error) {
		mu.Lock()
		if firstErr == nil {
			firstErr = err
			cancel()
		}
		mu.Unlock()
	}

	for _, task := range tasks {
		wg.Add(1)
		go func(task candidate) {
			defer wg.Done()

			if sem != nil {
				select {
				case sem <- struct{}{}:
					defer func() { <-sem }()
				case <-ctx.Done():
					fail(context.Cause(ctx))
					return
				}
			}
			// Cancelled while queued, or before the wave started: skip the
			// call rather than issue one that is already pointless.
			select {
			case <-ctx.Done():
				fail(context.Cause(ctx))
				return
			default:
			}

			res, err := e.scorer.Score(ctx, ScoreRequest{
				Model:       req.Model,
				State:       state,
				Statement:   task.statement,
				Temperature: req.Temperature,
			})
			if err != nil {
				fail(err)
				return
			}
			results[task.question][task.index] = res
		}(task)
	}

	wg.Wait()
	return firstErr
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

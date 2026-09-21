package classifier

import "context"

// ScoreRequest asks for the probability that one statement holds given one
// state. It is the only question this package's backend is ever asked.
type ScoreRequest struct {
	// Model names the model to use. Empty leaves the choice to the scorer's
	// own configuration.
	Model string
	// State is the rendered state text.
	State string
	// Statement is the proposition to judge.
	Statement string
	// Examples is the question's worked examples, already rendered as one
	// labelled block by [ExamplesBlock], or empty when the question declares
	// none. A backend puts it in the part of the prompt that does not change
	// between candidates.
	//
	// It crosses the seam rendered rather than as []Example on purpose. Every
	// candidate of one question is handed the same bytes, which is what lets a
	// backend keep that part of the prompt identical across a whole wave, and
	// a backend needs to know nothing about question types to place a string.
	Examples string
	// Temperature is the sampling temperature for the call.
	Temperature float64
}

// ScoreResult is one probability and what it cost.
type ScoreResult struct {
	// Probability is the estimate, in [0,1]. A value outside that range is
	// clamped into it and a NaN is read as 0.5; see [Scorer].
	Probability float64
	// InputTokens and OutputTokens are 0 when the provider reports nothing.
	InputTokens  int
	OutputTokens int
}

// Scorer estimates the probability that a statement is true given a state.
//
// Implementations must be safe for concurrent use: a single evaluation issues
// many Score calls at once. A Scorer should honour the context's deadline and
// cancellation.
//
// An implementation is expected to return a probability in [0,1], but is not
// trusted to: the evaluator clamps what it gets into [0,1] and reads a NaN as
// 0.5 before anything else looks at it. The clamp means a scorer cannot make
// the package report a probability it could not have produced itself, and the
// NaN rule means one unreadable candidate blunts one score instead of poisoning
// every probability and confidence in the response and failing the encode.
// Returning an error is still the way to say a call went wrong; a number
// outside [0,1] only says the scorer is miscalibrated.
type Scorer interface {
	Score(ctx context.Context, req ScoreRequest) (ScoreResult, error)
}

// GenerateRequest is a plain chat completion, used only by ModeOneshot.
type GenerateRequest struct {
	Model       string
	System      string
	User        string
	Temperature float64
	// JSONObject asks the provider to constrain the reply to a JSON object
	// where it supports doing so.
	JSONObject bool
}

// GenerateResult is the model's reply and what it cost.
type GenerateResult struct {
	Content string
	// FinishReason is why the provider stopped generating, verbatim, or ""
	// when it said nothing. "length" means the reply was cut off at an
	// output token limit rather than finished, which is the difference
	// between a document the model got wrong and a document it never got
	// to the end of. See [ErrTruncatedReply].
	FinishReason string
	InputTokens  int
	OutputTokens int
}

// Generator runs an unconstrained chat completion. A Scorer that also
// implements Generator can serve ModeOneshot; one that does not will make
// [Evaluator.Evaluate] return ErrOneshotUnsupported for that mode.
type Generator interface {
	Generate(ctx context.Context, req GenerateRequest) (GenerateResult, error)
}

// ScorerFunc adapts a function to the Scorer interface.
type ScorerFunc func(ctx context.Context, req ScoreRequest) (ScoreResult, error)

// Score calls f.
func (f ScorerFunc) Score(ctx context.Context, req ScoreRequest) (ScoreResult, error) {
	return f(ctx, req)
}

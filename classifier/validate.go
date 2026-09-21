package classifier

import (
	"errors"
	"fmt"
	"strings"
)

// Sentinel errors a caller can test for with [errors.Is].
var (
	// ErrNoQuestions is returned when a request declares no questions at all.
	ErrNoQuestions = errors.New("classifier: at least one question is required")

	// ErrOneshotUnsupported is returned for [ModeOneshot] when the configured
	// [Scorer] does not also implement [Generator].
	ErrOneshotUnsupported = errors.New("classifier: oneshot mode needs a Scorer that also implements Generator")

	// ErrNoScorer is returned when an [Evaluator] was built without a Scorer.
	ErrNoScorer = errors.New("classifier: no Scorer configured")

	// ErrUnknownMode is returned when a request names a mode that is neither
	// [ModeParallel] nor [ModeOneshot].
	ErrUnknownMode = errors.New("classifier: unknown mode")

	// ErrTruncatedReply is returned for [ModeOneshot] when the provider cut
	// the answer document off at an output token limit. A document that
	// stops mid-way is not a model that answered badly, and reporting it as
	// unreadable JSON sends whoever reads the message looking for the wrong
	// fault.
	ErrTruncatedReply = errors.New("classifier: the one-shot reply was cut off at the output token limit before the answer document was complete")
)

// Limits on what a question may declare. They follow the published contract
// this API is modelled on: 255 distinct labels for a choice, and a scale of
// two to ten levels — one level carries no information, and the contract stops
// at ten.
const (
	MaxChoiceOptions = 255
	MinScoreLevels   = 2
	MaxScoreLevels   = 10
)

// ValidationError reports one thing wrong with a request's questions.
type ValidationError struct {
	// Question is the name of the offending question, empty when the problem
	// is with the request as a whole.
	Question string
	// Field names the part of the question at fault: "name", "type",
	// "criteria" or "examples".
	Field string
	// Message says what is wrong with it.
	Message string
}

// Error renders the location and the problem.
func (e *ValidationError) Error() string {
	var b strings.Builder
	b.WriteString("classifier: invalid request")
	if e.Question != "" {
		fmt.Fprintf(&b, ": question %q", e.Question)
	}
	if e.Field != "" {
		fmt.Fprintf(&b, ": %s", e.Field)
	}
	fmt.Fprintf(&b, ": %s", e.Message)
	return b.String()
}

// Validate checks that every question can actually be asked.
//
// It returns [ErrNoQuestions] when there are none, and otherwise a
// [*ValidationError] naming the first question and field at fault. Questions
// are checked in declaration order.
func Validate(qs Questions) error {
	if qs.Len() == 0 {
		return ErrNoQuestions
	}

	for name, q := range qs.All() {
		if strings.TrimSpace(name) == "" {
			return &ValidationError{
				Question: name,
				Field:    "name",
				Message:  "question name must not be empty",
			}
		}
		if err := validateQuestion(name, q); err != nil {
			return err
		}
	}
	return nil
}

// validateQuestion checks one question's criteria and then its worked
// examples, in that order: an example answers in the criteria's terms, so
// there is nothing to check an example against until the criteria are known
// to be sound.
func validateQuestion(name string, q Question) error {
	if err := validateCriteria(name, q); err != nil {
		return err
	}
	return validateExamples(name, q)
}

func validateCriteria(name string, q Question) error {
	switch q.Type {
	case TypeChoice:
		n := q.Options.Len()
		if n == 0 {
			return &ValidationError{
				Question: name,
				Field:    "criteria",
				Message:  "a choice question needs at least one option",
			}
		}
		if n > MaxChoiceOptions {
			return &ValidationError{
				Question: name,
				Field:    "criteria",
				Message: fmt.Sprintf("a choice question may declare at most %d options, got %d",
					MaxChoiceOptions, n),
			}
		}
		for i, key := range q.Options.Keys() {
			if strings.TrimSpace(key) == "" {
				return &ValidationError{
					Question: name,
					Field:    "criteria",
					Message:  fmt.Sprintf("option %d has an empty key", i),
				}
			}
		}
		return nil

	case TypeScore:
		n := len(q.Levels)
		if n < MinScoreLevels || n > MaxScoreLevels {
			return &ValidationError{
				Question: name,
				Field:    "criteria",
				Message: fmt.Sprintf("a score question needs between %d and %d levels, got %d",
					MinScoreLevels, MaxScoreLevels, n),
			}
		}
		for i, label := range q.Levels {
			if strings.TrimSpace(label) == "" {
				return &ValidationError{
					Question: name,
					Field:    "criteria",
					Message:  fmt.Sprintf("level %d has an empty label", i),
				}
			}
		}
		return nil

	case TypeNoul, TypeBoolean:
		// A noul question declares nothing beyond its instruction, and even
		// that has a fallback.
		return nil

	case "":
		return &ValidationError{
			Question: name,
			Field:    "type",
			Message: fmt.Sprintf("question type is missing; expected one of %q, %q, %q",
				TypeChoice, TypeScore, TypeNoul),
		}

	default:
		return &ValidationError{
			Question: name,
			Field:    "type",
			Message: fmt.Sprintf("unknown question type %q; expected one of %q, %q, %q",
				q.Type, TypeChoice, TypeScore, TypeNoul),
		}
	}
}

// validateExamples checks that every worked example a question declares can
// actually be shown: it is about some state, and its answer is one the
// question could have given.
//
// An example that names an undeclared option, or a level off the end of the
// scale, teaches the model an answer it is not allowed to reach — and an
// example is paid for on every call of the wave, so a wrong one is wrong many
// times over. It is caught here, before any call is made, rather than at
// prompt-building time, where there is nowhere good to report it.
//
// "Can actually be shown" includes the state rendering at all, not only being
// present. A state carrying bytes that are not JSON is not zero, so nothing
// else here objects to it, and [ExamplesBlock] is then the first thing to try
// them — which is exactly the prompt-building time this check exists to get
// ahead of. It is reachable only from a [State] built in Go, because the wire
// decoder rejects those bytes on the way in.
func validateExamples(name string, q Question) error {
	for i, ex := range q.Examples {
		if ex.State.IsZero() {
			return &ValidationError{
				Question: name,
				Field:    "examples",
				Message:  fmt.Sprintf("example %d has no state", i),
			}
		}
		if _, err := ex.State.Text(); err != nil {
			return &ValidationError{
				Question: name,
				Field:    "examples",
				Message:  fmt.Sprintf("example %d has a state that cannot be rendered: %v", i, err),
			}
		}
		switch q.Type {
		case TypeChoice:
			if _, ok := q.Options.Get(ex.Choice); !ok {
				return &ValidationError{
					Question: name,
					Field:    "examples",
					Message: fmt.Sprintf("example %d answers %q, which is not one of the declared option keys",
						i, ex.Choice),
				}
			}
		case TypeScore:
			if ex.Level < 0 || ex.Level >= len(q.Levels) {
				return &ValidationError{
					Question: name,
					Field:    "examples",
					Message: fmt.Sprintf("example %d answers with level %d, outside the declared scale of %d levels (0 to %d)",
						i, ex.Level, len(q.Levels), len(q.Levels)-1),
				}
			}
		}
	}
	return nil
}

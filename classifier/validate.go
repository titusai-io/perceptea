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
	// Field names the part of the question at fault: "name", "type" or
	// "criteria".
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

func validateQuestion(name string, q Question) error {
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

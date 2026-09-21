package classifier

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
)

// choiceWith builds a choice question with n generated option keys.
func choiceWith(n int) Question {
	options := NewOrderedMap[string]()
	for i := range n {
		options.Set(fmt.Sprintf("option-%d", i), "")
	}
	return Question{Type: TypeChoice, Options: *options}
}

// optionsOf builds an option map from the given keys, each with a description
// derived from its key so that nothing depends on a blank one.
func optionsOf(keys ...string) *OrderedMap[string] {
	options := NewOrderedMap[string]()
	for _, key := range keys {
		options.Set(key, "The "+key+" team")
	}
	return options
}

// levelsOf builds a score question with n generated level labels.
func levelsOf(n int) Question {
	levels := make([]string, n)
	for i := range levels {
		levels[i] = fmt.Sprintf("level-%d", i)
	}
	return Question{Type: TypeScore, Levels: levels}
}

// one wraps a single question into a Questions map under name.
func one(name string, q Question) Questions {
	qs := NewOrderedMap[Question]()
	qs.Set(name, q)
	return *qs
}

func TestValidateNoQuestions(t *testing.T) {
	var empty Questions
	if err := Validate(empty); !errors.Is(err, ErrNoQuestions) {
		t.Fatalf("Validate(zero value) = %v, want ErrNoQuestions", err)
	}
	if err := Validate(*NewOrderedMap[Question]()); !errors.Is(err, ErrNoQuestions) {
		t.Fatalf("Validate(empty map) = %v, want ErrNoQuestions", err)
	}
}

func TestValidateAccepts(t *testing.T) {
	tests := []struct {
		name string
		qs   Questions
	}{
		{name: "a single option choice", qs: one("department", choiceWith(1))},
		{name: "the maximum options", qs: one("department", choiceWith(MaxChoiceOptions))},
		{name: "the minimum levels", qs: one("urgency", levelsOf(MinScoreLevels))},
		{name: "the maximum levels", qs: one("urgency", levelsOf(MaxScoreLevels))},
		{name: "a bare noul", qs: one("angry", Question{Type: TypeNoul})},
		{name: "a boolean synonym", qs: one("angry", Question{Type: TypeBoolean})},
		{
			name: "a whole request",
			qs: mustQuestions(t, `{
				"department": {"type": "choice", "criteria": {"billing": "Charges", "technical": "Bugs"}},
				"urgency": {"type": "score", "criteria": ["Low", "High"]},
				"angry": {"type": "noul"}
			}`),
		},
		{
			name: "worked examples answering in each type's own terms",
			qs: mustQuestions(t, `{
				"department": {
					"type": "choice",
					"criteria": {"billing": "Charges", "technical": "Bugs"},
					"examples": [
						{"state": "charged twice", "answer": "billing"},
						{"state": "a 500 on every load", "answer": "technical"}
					]
				},
				"urgency": {
					"type": "score",
					"criteria": ["Low", "Medium", "High"],
					"examples": [
						{"state": "no rush", "answer": 0},
						{"state": "launch in an hour", "answer": 2}
					]
				},
				"angry": {
					"type": "boolean",
					"examples": [{"state": "thanks!", "answer": false}]
				}
			}`),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := Validate(tc.qs); err != nil {
				t.Fatalf("Validate returned %v, want no error", err)
			}
		})
	}
}

func TestValidateRejects(t *testing.T) {
	tooManyOptions := choiceWith(MaxChoiceOptions + 1)

	emptyOptionKey := NewOrderedMap[string]()
	emptyOptionKey.Set("billing", "Charges")
	emptyOptionKey.Set("   ", "Whitespace only")

	tests := []struct {
		name         string
		qs           Questions
		wantQuestion string
		wantField    string
		wantMessage  string
	}{
		{
			name:         "an empty question name",
			qs:           one("", Question{Type: TypeNoul}),
			wantQuestion: "",
			wantField:    "name",
			wantMessage:  "question name must not be empty",
		},
		{
			name:         "a whitespace question name",
			qs:           one("  \t ", Question{Type: TypeNoul}),
			wantQuestion: "  \t ",
			wantField:    "name",
			wantMessage:  "question name must not be empty",
		},
		{
			name:         "a choice with no options",
			qs:           one("department", choiceWith(0)),
			wantQuestion: "department",
			wantField:    "criteria",
			wantMessage:  "at least one option",
		},
		{
			name:         "a choice with too many options",
			qs:           one("department", tooManyOptions),
			wantQuestion: "department",
			wantField:    "criteria",
			wantMessage:  "at most 255 options, got 256",
		},
		{
			name:         "a choice with an empty option key",
			qs:           one("department", Question{Type: TypeChoice, Options: *emptyOptionKey}),
			wantQuestion: "department",
			wantField:    "criteria",
			wantMessage:  "option 1 has an empty key",
		},
		{
			name:         "a score with one level",
			qs:           one("urgency", levelsOf(1)),
			wantQuestion: "urgency",
			wantField:    "criteria",
			wantMessage:  "between 2 and 10 levels, got 1",
		},
		{
			name:         "a score with no levels",
			qs:           one("urgency", Question{Type: TypeScore}),
			wantQuestion: "urgency",
			wantField:    "criteria",
			wantMessage:  "between 2 and 10 levels, got 0",
		},
		{
			name:         "a score with too many levels",
			qs:           one("urgency", levelsOf(MaxScoreLevels+1)),
			wantQuestion: "urgency",
			wantField:    "criteria",
			wantMessage:  "between 2 and 10 levels, got 11",
		},
		{
			name:         "a score with an empty level label",
			qs:           one("urgency", Question{Type: TypeScore, Levels: []string{"Low", " "}}),
			wantQuestion: "urgency",
			wantField:    "criteria",
			wantMessage:  "level 1 has an empty label",
		},
		// A bad example is caught here rather than at prompt-building time.
		// It is shown to the model on every call of the wave, so an example
		// naming an answer the question could not give is wrong many times
		// over, and cheapest to reject before the first call.
		{
			name: "a choice example answering with an undeclared option key",
			qs: one("department", Question{
				Type:     TypeChoice,
				Options:  *optionsOf("billing", "technical"),
				Examples: []Example{{State: StringState("charged twice"), Choice: "sales"}},
			}),
			wantQuestion: "department",
			wantField:    "examples",
			wantMessage:  `example 0 answers "sales", which is not one of the declared option keys`,
		},
		{
			name: "a choice example answering with no option key at all",
			qs: one("department", Question{
				Type:    TypeChoice,
				Options: *optionsOf("billing"),
				Examples: []Example{
					{State: StringState("charged twice"), Choice: "billing"},
					{State: StringState("a 500 on every load")},
				},
			}),
			wantQuestion: "department",
			wantField:    "examples",
			wantMessage:  `example 1 answers "", which is not one of the declared option keys`,
		},
		{
			name: "a score example answering off the end of the scale",
			qs: one("urgency", Question{
				Type:     TypeScore,
				Levels:   []string{"Low", "Medium", "High"},
				Examples: []Example{{State: StringState("launch in an hour"), Level: 3}},
			}),
			wantQuestion: "urgency",
			wantField:    "examples",
			wantMessage:  "example 0 answers with level 3, outside the declared scale of 3 levels (0 to 2)",
		},
		{
			name: "a score example answering with a negative level",
			qs: one("urgency", Question{
				Type:     TypeScore,
				Levels:   []string{"Low", "High"},
				Examples: []Example{{State: StringState("no rush"), Level: -1}},
			}),
			wantQuestion: "urgency",
			wantField:    "examples",
			wantMessage:  "example 0 answers with level -1, outside the declared scale of 2 levels (0 to 1)",
		},
		{
			name: "an example with no state",
			qs: one("angry", Question{
				Type:     TypeNoul,
				Examples: []Example{{Noul: true}},
			}),
			wantQuestion: "angry",
			wantField:    "examples",
			wantMessage:  "example 0 has no state",
		},
		{
			name: "an example with no state on a choice question",
			qs: one("department", Question{
				Type:     TypeChoice,
				Options:  *optionsOf("billing"),
				Examples: []Example{{Choice: "billing"}},
			}),
			wantQuestion: "department",
			wantField:    "examples",
			wantMessage:  "example 0 has no state",
		},
		{
			name:         "an unknown question type",
			qs:           one("ranking", Question{Type: "ranking"}),
			wantQuestion: "ranking",
			wantField:    "type",
			wantMessage:  `unknown question type "ranking"`,
		},
		{
			name:         "a missing question type",
			qs:           one("mystery", Question{}),
			wantQuestion: "mystery",
			wantField:    "type",
			wantMessage:  "question type is missing",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := Validate(tc.qs)
			if err == nil {
				t.Fatal("Validate accepted an invalid request")
			}
			var verr *ValidationError
			if !errors.As(err, &verr) {
				t.Fatalf("Validate returned %T (%v), want a *ValidationError", err, err)
			}
			if verr.Question != tc.wantQuestion {
				t.Errorf("Question = %q, want %q", verr.Question, tc.wantQuestion)
			}
			if verr.Field != tc.wantField {
				t.Errorf("Field = %q, want %q", verr.Field, tc.wantField)
			}
			if !strings.Contains(verr.Message, tc.wantMessage) {
				t.Errorf("Message = %q, want it to contain %q", verr.Message, tc.wantMessage)
			}
			if !strings.Contains(err.Error(), verr.Message) {
				t.Errorf("Error() = %q, want it to contain the message %q", err.Error(), verr.Message)
			}
			if tc.wantField != "" && !strings.Contains(err.Error(), tc.wantField) {
				t.Errorf("Error() = %q, want it to name the field %q", err.Error(), tc.wantField)
			}
		})
	}
}

// TestValidateReportsTheFirstProblemInOrder pins that questions are checked in
// declaration order, so the reported name is stable.
func TestValidateReportsTheFirstProblemInOrder(t *testing.T) {
	qs := NewOrderedMap[Question]()
	qs.Set("angry", Question{Type: TypeNoul})
	qs.Set("urgency", levelsOf(1))
	qs.Set("department", choiceWith(0))

	var verr *ValidationError
	if err := Validate(*qs); !errors.As(err, &verr) {
		t.Fatalf("Validate returned %v, want a *ValidationError", err)
	}
	if verr.Question != "urgency" {
		t.Fatalf("reported question %q, want the first invalid one, %q", verr.Question, "urgency")
	}
}

// A question whose criteria and whose example are both wrong reports the
// criteria. An example answers in the criteria's terms, so there is nothing
// worth saying about it until the criteria themselves hold up.
func TestCriteriaAreReportedBeforeExamples(t *testing.T) {
	qs := one("urgency", Question{
		Type:     TypeScore,
		Levels:   []string{"Low"},
		Examples: []Example{{State: StringState("no rush"), Level: 9}},
	})

	var verr *ValidationError
	if err := Validate(qs); !errors.As(err, &verr) {
		t.Fatalf("Validate returned %v, want a *ValidationError", err)
	}
	if verr.Field != "criteria" {
		t.Errorf("Field = %q, want criteria; the message was %q", verr.Field, verr.Message)
	}
}

// A bad example costs nothing to find and is shown to the model on every call
// of the wave, so it is found before the wave starts rather than after a
// provider has been paid to read it.
func TestABadExampleIsRejectedBeforeAnyCallIsMade(t *testing.T) {
	var calls atomic.Int64
	e := New(ScorerFunc(func(context.Context, ScoreRequest) (ScoreResult, error) {
		calls.Add(1)
		return ScoreResult{Probability: 0.5}, nil
	}))

	qs := mustQuestions(t, `{"department": {
		"type": "choice",
		"criteria": {"billing": "Charges", "technical": "Bugs"},
		"examples": [{"state": "charged twice", "answer": "sales"}]
	}}`)

	_, err := e.Evaluate(context.Background(), Request{State: StringState("charged twice"), Questions: qs})
	var verr *ValidationError
	// Not a Fatal: the count below is the other half of the claim, and it is
	// worth reporting even once the error has gone wrong.
	if !errors.As(err, &verr) {
		t.Errorf("Evaluate returned %v, want a *ValidationError", err)
	} else if verr.Field != "examples" {
		t.Errorf("Field = %q, want examples", verr.Field)
	}
	if n := calls.Load(); n != 0 {
		t.Errorf("the scorer was called %d times; a request with a bad example must not reach a provider", n)
	}
}

func TestValidationErrorString(t *testing.T) {
	tests := []struct {
		name string
		err  *ValidationError
		want string
	}{
		{
			name: "fully populated",
			err:  &ValidationError{Question: "urgency", Field: "criteria", Message: "too few levels"},
			want: `classifier: invalid request: question "urgency": criteria: too few levels`,
		},
		{
			name: "without a question",
			err:  &ValidationError{Field: "questions", Message: "nothing to ask"},
			want: "classifier: invalid request: questions: nothing to ask",
		},
		{
			name: "message only",
			err:  &ValidationError{Message: "nothing to ask"},
			want: "classifier: invalid request: nothing to ask",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.err.Error(); got != tc.want {
				t.Fatalf("Error() = %q, want %q", got, tc.want)
			}
		})
	}
}

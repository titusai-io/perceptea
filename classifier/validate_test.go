package classifier

import (
	"errors"
	"fmt"
	"strings"
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

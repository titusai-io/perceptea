package classifier

import "testing"

// The expected strings below were written out by hand from the wording each
// statement is specified to have, rather than captured from the builders they
// check: a golden taken from the code under test cannot catch that code
// rephrasing a statement, and the phrasing is what the probabilities are
// calibrated against.

func TestChoiceStatement(t *testing.T) {
	tests := []struct {
		name         string
		question     string
		instructions string
		key          string
		description  string
		want         string
	}{
		{
			name:        "default instruction",
			question:    "department",
			key:         "billing",
			description: "Charges, refunds",
			want:        `The correct select the best label for "department" is "billing" (Charges, refunds).`,
		},
		{
			name:     "default instruction without a description",
			question: "department",
			key:      "billing",
			want:     `The correct select the best label for "department" is "billing".`,
		},
		{
			name:         "one trailing question mark is stripped",
			question:     "department",
			instructions: "Which team should handle this?",
			key:          "billing",
			description:  "Charges, refunds, invoices",
			want:         `The correct which team should handle this is "billing" (Charges, refunds, invoices).`,
		},
		{
			name:         "only the last question mark is stripped",
			question:     "department",
			instructions: "Which team??",
			key:          "billing",
			want:         `The correct which team? is "billing".`,
		},
		{
			name:         "a question mark inside the instruction survives",
			question:     "department",
			instructions: "Why? Which team",
			key:          "billing",
			want:         `The correct why? which team is "billing".`,
		},
		{
			name:         "the whole instruction is lowercased",
			question:     "department",
			instructions: "Which TEAM Should Handle This?",
			key:          "technical",
			description:  "Bugs",
			want:         `The correct which team should handle this is "technical" (Bugs).`,
		},
		{
			name:         "the option key and description keep their case",
			question:     "department",
			instructions: "Which team",
			key:          "Billing-Team",
			description:  "Charges AND Refunds",
			want:         `The correct which team is "Billing-Team" (Charges AND Refunds).`,
		},
		{
			name:         "an instruction ending in a period keeps it",
			question:     "department",
			instructions: "Pick a team.",
			key:          "billing",
			want:         `The correct pick a team. is "billing".`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ChoiceStatement(tc.question, tc.instructions, tc.key, tc.description)
			if got != tc.want {
				t.Fatalf("ChoiceStatement:\n got %q\nwant %q", got, tc.want)
			}
		})
	}
}

func TestScoreStatement(t *testing.T) {
	tests := []struct {
		name         string
		question     string
		instructions string
		index        int
		label        string
		want         string
	}{
		{
			name:     "default instruction",
			question: "urgency",
			index:    0,
			label:    "Low",
			want:     `On the scale for "Rate "urgency"", the most appropriate rating is level 0: "Low".`,
		},
		{
			name:         "declared instruction keeps its case and question mark",
			question:     "urgency",
			instructions: "How urgent is this?",
			index:        2,
			label:        "High",
			want:         `On the scale for "How urgent is this?", the most appropriate rating is level 2: "High".`,
		},
		{
			name:         "a two digit index",
			question:     "urgency",
			instructions: "How urgent is this?",
			index:        10,
			label:        "Extreme",
			want:         `On the scale for "How urgent is this?", the most appropriate rating is level 10: "Extreme".`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ScoreStatement(tc.question, tc.instructions, tc.index, tc.label)
			if got != tc.want {
				t.Fatalf("ScoreStatement:\n got %q\nwant %q", got, tc.want)
			}
		})
	}
}

func TestNoulStatement(t *testing.T) {
	tests := []struct {
		name         string
		question     string
		instructions string
		want         string
	}{
		{
			name:     "no instruction falls back to the question name",
			question: "angry",
			want:     `The proposition "angry" is true.`,
		},
		{
			name:         "a blank instruction falls back too",
			question:     "angry",
			instructions: "   ",
			want:         `The proposition "angry" is true.`,
		},
		{
			name:         "a question keeps its question mark",
			question:     "angry",
			instructions: "Is the customer angry?",
			want:         "Is the customer angry?",
		},
		{
			name:         "a sentence keeps its single period",
			question:     "angry",
			instructions: "The customer is angry.",
			want:         "The customer is angry.",
		},
		{
			name:         "an unpunctuated instruction gets a period, after trimming",
			question:     "angry",
			instructions: "  The customer is angry  ",
			want:         "The customer is angry.",
		},
		{
			name:         "any other punctuation still gets a period",
			question:     "angry",
			instructions: "The customer is angry!",
			want:         "The customer is angry!.",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := NoulStatement(tc.question, tc.instructions)
			if got != tc.want {
				t.Fatalf("NoulStatement:\n got %q\nwant %q", got, tc.want)
			}
		})
	}
}

func TestDefaultInstructions(t *testing.T) {
	if got, want := DefaultChoiceInstructions("department"), `Select the best label for "department"`; got != want {
		t.Errorf("DefaultChoiceInstructions = %q, want %q", got, want)
	}
	if got, want := DefaultScoreInstructions("urgency"), `Rate "urgency"`; got != want {
		t.Errorf("DefaultScoreInstructions = %q, want %q", got, want)
	}
	if got, want := DefaultNoulStatement("angry"), `The proposition "angry" is true.`; got != want {
		t.Errorf("DefaultNoulStatement = %q, want %q", got, want)
	}
}

// TestStatementsFollowDeclarationOrder checks that a question produces one
// statement per declared candidate, in the order they were declared — the
// order that later decides which key wins a tie.
func TestStatementsFollowDeclarationOrder(t *testing.T) {
	qs := mustQuestions(t, `{
		"department": {
			"type": "choice",
			"instructions": "Which team should handle this?",
			"criteria": {"zeta": "Last declared", "alpha": "First declared", "middle": ""}
		},
		"urgency": {"type": "score", "criteria": ["Low", "Medium", "High"]},
		"angry": {"type": "boolean", "instructions": "Is the customer angry?"}
	}`)

	tests := []struct {
		question string
		want     []string
	}{
		{
			question: "department",
			want: []string{
				`The correct which team should handle this is "zeta" (Last declared).`,
				`The correct which team should handle this is "alpha" (First declared).`,
				`The correct which team should handle this is "middle".`,
			},
		},
		{
			question: "urgency",
			want: []string{
				`On the scale for "Rate "urgency"", the most appropriate rating is level 0: "Low".`,
				`On the scale for "Rate "urgency"", the most appropriate rating is level 1: "Medium".`,
				`On the scale for "Rate "urgency"", the most appropriate rating is level 2: "High".`,
			},
		},
		{
			question: "angry",
			want:     []string{"Is the customer angry?"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.question, func(t *testing.T) {
			q, ok := qs.Get(tc.question)
			if !ok {
				t.Fatalf("question %q missing from the decoded request", tc.question)
			}
			got := statements(tc.question, q)
			if len(got) != len(tc.want) {
				t.Fatalf("statements(%q) produced %d statements, want %d: %q", tc.question, len(got), len(tc.want), got)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("statement %d:\n got %q\nwant %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

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

// wantExamplesHeader is the block's opening line, written out by hand for the
// same reason the statements above are.
const wantExamplesHeader = "EXAMPLES (worked answers for other states, as guidance; judge only the STATE below):"

// TestExamplesBlock pins the wording a question's worked examples are shown
// in. It is paid for on every call of the wave, so both the shape and the
// length of it are part of the contract.
func TestExamplesBlock(t *testing.T) {
	tests := []struct {
		name     string
		document string
		question string
		want     string
	}{
		{
			name:     "a choice answers with its option key",
			question: "department",
			document: `{"department": {
				"type": "choice",
				"instructions": "Which team should handle this?",
				"criteria": {"billing": "Charges, refunds, invoices", "technical": "Bugs"},
				"examples": [
					{"state": "I was charged twice for the Pro plan.", "answer": "billing"},
					{"state": "The dashboard 500s on every load.", "answer": "technical"}
				]
			}}`,
			want: wantExamplesHeader +
				"\n\nEXAMPLE 1 STATE:\nI was charged twice for the Pro plan." +
				"\nEXAMPLE 1 ANSWER: the correct option is \"billing\"." +
				"\n\nEXAMPLE 2 STATE:\nThe dashboard 500s on every load." +
				"\nEXAMPLE 2 ANSWER: the correct option is \"technical\".",
		},
		{
			name:     "a score answers with its level index and label",
			question: "urgency",
			document: `{"urgency": {
				"type": "score",
				"criteria": ["Low", "Medium", "High"],
				"examples": [
					{"state": "Whenever you get a chance.", "answer": 0},
					{"state": "Our launch is in an hour and nothing works.", "answer": 2}
				]
			}}`,
			want: wantExamplesHeader +
				"\n\nEXAMPLE 1 STATE:\nWhenever you get a chance." +
				"\nEXAMPLE 1 ANSWER: the correct rating is level 0: \"Low\"." +
				"\n\nEXAMPLE 2 STATE:\nOur launch is in an hour and nothing works." +
				"\nEXAMPLE 2 ANSWER: the correct rating is level 2: \"High\".",
		},
		{
			name:     "a noul answers with the proposition holding or not",
			question: "angry",
			document: `{"angry": {
				"type": "boolean",
				"instructions": "Is the customer angry?",
				"examples": [
					{"state": "Thanks for the quick fix!", "answer": false},
					{"state": "Third time this month. Fix it or I am cancelling.", "answer": true}
				]
			}}`,
			want: wantExamplesHeader +
				"\n\nEXAMPLE 1 STATE:\nThanks for the quick fix!" +
				"\nEXAMPLE 1 ANSWER: the proposition is false." +
				"\n\nEXAMPLE 2 STATE:\nThird time this month. Fix it or I am cancelling." +
				"\nEXAMPLE 2 ANSWER: the proposition is true.",
		},
		{
			name:     "a structured state is rendered the way a request's own state is",
			question: "department",
			document: `{"department": {
				"type": "choice",
				"criteria": {"billing": "Charges"},
				"examples": [{"state": {"plan": "pro", "charges": 2}, "answer": "billing"}]
			}}`,
			want: wantExamplesHeader +
				"\n\nEXAMPLE 1 STATE:\n{\n  \"plan\": \"pro\",\n  \"charges\": 2\n}" +
				"\nEXAMPLE 1 ANSWER: the correct option is \"billing\".",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			q, ok := mustQuestions(t, tc.document).Get(tc.question)
			if !ok {
				t.Fatalf("question %q missing from the decoded request", tc.question)
			}
			got, err := ExamplesBlock(q)
			if err != nil {
				t.Fatalf("ExamplesBlock: %v", err)
			}
			if got != tc.want {
				t.Fatalf("ExamplesBlock:\n got %q\nwant %q", got, tc.want)
			}
		})
	}
}

// A question that declares no examples renders nothing at all, so that the
// prompt it produces is the one it would have produced before examples
// existed. An empty block and a block of whitespace are different things to
// whatever assembles the prompt around it.
func TestExamplesBlockIsEmptyWithoutExamples(t *testing.T) {
	qs := mustQuestions(t, `{
		"department": {"type": "choice", "criteria": {"billing": "Charges"}},
		"urgency": {"type": "score", "criteria": ["Low", "High"]},
		"angry": {"type": "noul"}
	}`)
	for name, q := range qs.All() {
		got, err := ExamplesBlock(q)
		if err != nil {
			t.Fatalf("%s: ExamplesBlock: %v", name, err)
		}
		if got != "" {
			t.Errorf("%s: ExamplesBlock = %q, want the empty string", name, got)
		}
	}
}

// The block goes into the part of the prompt that must be byte-identical for
// every candidate of a question, so rendering it twice has to produce the same
// bytes twice — no map iteration, no clock, nothing from the candidate.
func TestExamplesBlockIsDeterministic(t *testing.T) {
	q, _ := mustQuestions(t, `{"department": {
		"type": "choice",
		"criteria": {"zeta": "Last", "alpha": "First", "middle": "Between"},
		"examples": [
			{"state": "one", "answer": "zeta"},
			{"state": "two", "answer": "alpha"},
			{"state": "three", "answer": "middle"}
		]
	}}`).Get("department")

	first, err := ExamplesBlock(q)
	if err != nil {
		t.Fatalf("ExamplesBlock: %v", err)
	}
	for i := range 16 {
		again, err := ExamplesBlock(q)
		if err != nil {
			t.Fatalf("ExamplesBlock %d: %v", i, err)
		}
		if again != first {
			t.Fatalf("ExamplesBlock is not stable\n run %d: %q\n run 0: %q", i+1, again, first)
		}
	}
}

// Examples keep the order they were declared in, the way every other declared
// list here does.
func TestExamplesBlockFollowsDeclarationOrder(t *testing.T) {
	q, _ := mustQuestions(t, `{"urgency": {
		"type": "score",
		"criteria": ["Low", "Medium", "High"],
		"examples": [
			{"state": "third", "answer": 2},
			{"state": "first", "answer": 0},
			{"state": "second", "answer": 1}
		]
	}}`).Get("urgency")

	got, err := ExamplesBlock(q)
	if err != nil {
		t.Fatalf("ExamplesBlock: %v", err)
	}
	want := wantExamplesHeader +
		"\n\nEXAMPLE 1 STATE:\nthird\nEXAMPLE 1 ANSWER: the correct rating is level 2: \"High\"." +
		"\n\nEXAMPLE 2 STATE:\nfirst\nEXAMPLE 2 ANSWER: the correct rating is level 0: \"Low\"." +
		"\n\nEXAMPLE 3 STATE:\nsecond\nEXAMPLE 3 ANSWER: the correct rating is level 1: \"Medium\"."
	if got != want {
		t.Fatalf("ExamplesBlock:\n got %q\nwant %q", got, want)
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

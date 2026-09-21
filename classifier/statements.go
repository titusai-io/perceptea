package classifier

import (
	"strconv"
	"strings"
)

// A statement is the one thing a [Scorer] is ever shown: a single proposition
// about the state, phrased so that "how likely is this true?" is a meaningful
// question. One is built per candidate answer — per option key, per score
// level, or one for a noul question.
//
// The wording below is fixed, quirks included, because the wording is what the
// probabilities are calibrated against: a tidier phrasing is a different
// question to the model, and the numbers move with it. How a statement is
// wrapped into a prompt is the provider's business, not this package's.

// DefaultChoiceInstructions is the instruction a choice question falls back to
// when it declares none.
func DefaultChoiceInstructions(name string) string {
	return `Select the best label for "` + name + `"`
}

// DefaultScoreInstructions is the instruction a score question falls back to
// when it declares none.
func DefaultScoreInstructions(name string) string {
	return `Rate "` + name + `"`
}

// DefaultNoulStatement is the statement a noul question falls back to when it
// declares no instruction.
func DefaultNoulStatement(name string) string {
	return `The proposition "` + name + `" is true.`
}

// ChoiceStatement renders the proposition that key is the right answer to a
// choice question, with the option's description in parentheses when it has
// one.
//
// The instruction is lowercased whole — not just its first letter — and one
// trailing question mark is dropped, so "Which team should handle this?"
// becomes "The correct which team should handle this is "billing" (…)." A
// question mark anywhere but the end survives, and a sentence-ending period is
// always appended.
//
// The lowercasing is [strings.ToLower], which is simple case mapping: it maps
// each rune on its own and so passes over the handful of characters whose
// lowercase form is context- or locale-dependent. A Greek capital sigma
// becomes σ even at the end of a word, where full case mapping would give ς,
// and a Turkish dotted capital İ becomes a bare i rather than the two runes
// i̇. Either changes the statement, and so the probability, for an instruction
// containing one. It is left that way on purpose: full case mapping means
// carrying Unicode's SpecialCasing table, which is a great deal of machinery
// for an instruction written in Greek or Turkish and lowercased by accident.
func ChoiceStatement(name, instructions, key, description string) string {
	if instructions == "" {
		instructions = DefaultChoiceInstructions(name)
	}
	phrase := strings.TrimSuffix(strings.ToLower(instructions), "?")

	var b strings.Builder
	b.WriteString("The correct ")
	b.WriteString(phrase)
	b.WriteString(` is "`)
	b.WriteString(key)
	b.WriteString(`"`)
	if description != "" {
		b.WriteString(" (")
		b.WriteString(description)
		b.WriteString(")")
	}
	b.WriteString(".")
	return b.String()
}

// ScoreStatement renders the proposition that level index is the right rating
// on a score question's scale. The instruction is used as written — no
// lowercasing, no question mark stripped.
func ScoreStatement(name, instructions string, index int, label string) string {
	if instructions == "" {
		instructions = DefaultScoreInstructions(name)
	}
	return `On the scale for "` + instructions +
		`", the most appropriate rating is level ` + strconv.Itoa(index) +
		`: "` + label + `".`
}

// NoulStatement renders a noul question's proposition: its instruction,
// trimmed, or a fallback naming the question. A statement that ends in neither
// a question mark nor a period gets a period; any other ending, "!" included,
// is left alone and the period is appended after it.
func NoulStatement(name, instructions string) string {
	statement := strings.TrimSpace(instructions)
	if statement == "" {
		return DefaultNoulStatement(name)
	}
	if !strings.HasSuffix(statement, "?") && !strings.HasSuffix(statement, ".") {
		statement += "."
	}
	return statement
}

// The labels of the worked-examples block. They are spelled the way the rest
// of the scoring prompt is — an upper-case label, a colon, the material — so
// that an example reads as the same kind of thing the model is about to be
// asked about.
const (
	examplesHeader = "EXAMPLES (worked answers for other states, as guidance; judge only the STATE below):"
	examplePrefix  = "EXAMPLE "
	exampleState   = " STATE:\n"
	exampleAnswer  = " ANSWER: "
)

// ExamplesBlock renders a question's worked examples as one labelled block: an
// example's state, then the answer that was correct for it, in the terms that
// question type answers in. A question that declares none renders the empty
// string, so nothing downstream mentions examples at all.
//
// The answer line is phrased to echo the statement the model will be scoring —
// "the correct option is", "the correct rating is level" — so that the example
// and the candidate are visibly the same claim, one already settled and one
// being asked about.
//
// The whole block is one string, and that is the point of it: a backend places
// it in the part of the prompt that is the same for every candidate of the
// question. Rendering an example as a prior exchange instead, ending in a
// {"p": 0.95} answer, would have to name a candidate statement in it, and the
// examples would then differ from call to call.
//
// An example's state is rendered the same way a request's own state is; the
// error is that rendering's, for a state whose bytes are not JSON.
func ExamplesBlock(q Question) (string, error) {
	if len(q.Examples) == 0 {
		return "", nil
	}

	var b strings.Builder
	b.WriteString(examplesHeader)
	for i, ex := range q.Examples {
		text, err := ex.State.Text()
		if err != nil {
			return "", err
		}
		n := strconv.Itoa(i + 1)
		b.WriteString("\n\n")
		b.WriteString(examplePrefix)
		b.WriteString(n)
		b.WriteString(exampleState)
		b.WriteString(text)
		b.WriteString("\n")
		b.WriteString(examplePrefix)
		b.WriteString(n)
		b.WriteString(exampleAnswer)
		b.WriteString(exampleAnswerText(q, ex))
	}
	return b.String(), nil
}

// exampleAnswerText renders one example's answer in its question type's terms.
//
// A level index outside the declared scale loses its label rather than
// panicking: [Validate] rejects one before any prompt is built, and a renderer
// is the wrong place to discover it a second time.
func exampleAnswerText(q Question, ex Example) string {
	switch q.Type {
	case TypeChoice:
		return `the correct option is "` + ex.Choice + `".`
	case TypeScore:
		out := "the correct rating is level " + strconv.Itoa(ex.Level)
		if ex.Level >= 0 && ex.Level < len(q.Levels) {
			out += `: "` + q.Levels[ex.Level] + `"`
		}
		return out + "."
	default:
		if ex.Noul {
			return "the proposition is true."
		}
		return "the proposition is false."
	}
}

// statements returns every statement a question needs scored, in the order the
// candidates were declared: one per option key for a choice, one per level for
// a score, and exactly one for a noul.
func statements(name string, q Question) []string {
	switch q.Type {
	case TypeChoice:
		keys := q.Options.Keys()
		out := make([]string, 0, len(keys))
		for _, key := range keys {
			description, _ := q.Options.Get(key)
			out = append(out, ChoiceStatement(name, q.Instructions, key, description))
		}
		return out
	case TypeScore:
		out := make([]string, 0, len(q.Levels))
		for i, label := range q.Levels {
			out = append(out, ScoreStatement(name, q.Instructions, i, label))
		}
		return out
	default:
		return []string{NoulStatement(name, q.Instructions)}
	}
}

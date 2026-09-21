package classifier

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// Oneshot mode asks for the whole answer sheet in one reply. It is the failure
// mode this package exists to avoid — the model has to invent a JSON document
// and keep every probability coherent at the same time — and it is kept as the
// comparison that makes the case for [ModeParallel]: the same questions and
// the same state, asked the obvious way, so the difference can be measured
// rather than asserted.

// oneshotSystemPrompt is the system message for [ModeOneshot]. The
// per-statement prompts of [ModeParallel] have no counterpart here: those
// belong to the provider, which owns how a [Scorer] call is phrased. A
// one-shot call has no statement to phrase, so its prompt is built from the
// questions and lives with them.
const oneshotSystemPrompt = "You are a precise decision engine. " +
	"Answer every question based only on the provided state. " +
	"Return probabilities that reflect genuine uncertainty. Do not invent information."

// errSnippetRunes bounds how much of an unparseable reply is quoted back.
const errSnippetRunes = 200

// oneshotQuestionBlock renders the "QUESTIONS:" section: one line per
// question, with an indented continuation listing a choice's options or a
// score's levels.
func oneshotQuestionBlock(qs Questions) string {
	var lines []string
	for name, q := range qs.All() {
		var b strings.Builder
		b.WriteString("- ")
		b.WriteString(name)
		b.WriteString(" (")
		// The declared word, not the normalised one: the model is shown
		// the question as the request wrote it, so one written as
		// "boolean" is described to the model as "boolean".
		b.WriteString(string(q.DeclaredType()))
		b.WriteString("): ")
		b.WriteString(q.Instructions)

		switch q.Type {
		case TypeChoice:
			parts := make([]string, 0, q.Options.Len())
			for key, description := range q.Options.All() {
				parts = append(parts, key+": "+description)
			}
			b.WriteString("\n  Options → ")
			b.WriteString(strings.Join(parts, ", "))
		case TypeScore:
			parts := make([]string, 0, len(q.Levels))
			for i, label := range q.Levels {
				parts = append(parts, strconv.Itoa(i)+"="+label)
			}
			b.WriteString("\n  Levels → ")
			b.WriteString(strings.Join(parts, " | "))
		}
		lines = append(lines, b.String())
	}
	return strings.Join(lines, "\n")
}

// oneshotUserPrompt renders the whole user message: the state, the questions
// and the shape the reply must take.
func oneshotUserPrompt(state string, qs Questions) string {
	return "STATE:\n" + state + "\n\nQUESTIONS:\n" + oneshotQuestionBlock(qs) + "\n\n" +
		"Respond with JSON matching this shape:\n" +
		`For each choice question: { "choice": "<key>", "confidence": 0-1, "probabilities": { "<key>": 0-1, ... } }` + "\n" +
		`For each score question: { "score": <float>, "confidence": 0-1 }` + "\n" +
		`For each noul question: { "noul": 0-1 }` + "\n" +
		"Top-level keys must be the question names."
}

// oneshotAnswer is one question's entry in the model's reply. Every field is a
// pointer so that "absent or null" is distinguishable from "zero": a field the
// model did not answer takes the default [oneshotAnswerFor] supplies, while a
// field it answered with 0 is an answer and is kept.
type oneshotAnswer struct {
	Choice        *string              `json:"choice"`
	Confidence    *float64             `json:"confidence"`
	Probabilities *OrderedMap[float64] `json:"probabilities"`
	Score         *float64             `json:"score"`
	Noul          *float64             `json:"noul"`
	Probability   *float64             `json:"probability"`
}

// evaluateOneshot asks for every answer in a single generation.
func (e *Evaluator) evaluateOneshot(ctx context.Context, req Request, state string) (evaluation, error) {
	generator, ok := e.scorer.(Generator)
	if !ok {
		return evaluation{}, fmt.Errorf("%w (%T)", ErrOneshotUnsupported, e.scorer)
	}

	result, err := generator.Generate(ctx, GenerateRequest{
		Model:       req.Model,
		System:      oneshotSystemPrompt,
		User:        oneshotUserPrompt(state, req.Questions),
		Temperature: req.Temperature,
		JSONObject:  true,
	})
	if err != nil {
		return evaluation{}, err
	}

	parsed, err := parseOneshotContent(result.Content)
	if err != nil {
		if truncatedFinish(result.FinishReason) {
			return evaluation{}, fmt.Errorf("%w; raise the provider's output token limit, "+
				"or use a model that does not spend it on thinking tokens: %s",
				ErrTruncatedReply, snippet(result.Content))
		}
		return evaluation{}, err
	}

	answers := NewOrderedMap[Answer]()
	for name, q := range req.Questions.All() {
		var data oneshotAnswer
		if raw, ok := parsed[name]; ok {
			// A reply that answers a question with something other than an
			// object is treated as no answer at all: the decode fails, data
			// is left zero, and every field falls back to its default.
			_ = json.Unmarshal(raw, &data)
		}
		answers.Set(name, oneshotAnswerFor(q, data))
	}

	return evaluation{
		answers:      *answers,
		inputTokens:  result.InputTokens,
		outputTokens: result.OutputTokens,
		calls:        1,
	}, nil
}

// parseOneshotContent decodes the reply into per-question raw values.
//
// Content that is not JSON at all is an error quoting what came back. Content
// that is valid JSON but not an object — an array, a bare string — parses to
// no answers, and every question then falls back to its defaults: well-formed
// JSON of the wrong shape is a reply that answered nothing, not a reply that
// could not be read. Empty content stands in for a reply the provider never
// filled in and is read as "{}".
func parseOneshotContent(content string) (map[string]json.RawMessage, error) {
	if strings.TrimSpace(content) == "" {
		content = "{}"
	}
	if !json.Valid([]byte(content)) {
		return nil, fmt.Errorf("classifier: one-shot model returned non-JSON: %s", snippet(content))
	}
	var parsed map[string]json.RawMessage
	if err := json.Unmarshal([]byte(content), &parsed); err != nil {
		return map[string]json.RawMessage{}, nil
	}
	return parsed, nil
}

// truncatedFinish reports whether a provider's finish reason says the reply
// ran out of output tokens rather than ending. The comparison is
// case-insensitive: the value is text a provider echoes, not a constant this
// package defines.
func truncatedFinish(reason string) bool {
	return strings.EqualFold(strings.TrimSpace(reason), "length")
}

// snippet trims a reply down to something quotable in an error. The cut is at
// 200 runes rather than 200 bytes, so the quote cannot end mid-character.
func snippet(s string) string {
	runes := []rune(s)
	if len(runes) <= errSnippetRunes {
		return s
	}
	return string(runes[:errSnippetRunes])
}

// oneshotAnswerFor turns one question's entry in the reply into its answer,
// filling in a default for everything the model left out.
func oneshotAnswerFor(q Question, data oneshotAnswer) Answer {
	switch q.Type {
	case TypeChoice:
		keys := q.Options.Keys()
		choice := ""
		if len(keys) > 0 {
			choice = keys[0]
		}
		if data.Choice != nil {
			choice = *data.Choice
		}
		confidence := 0.5
		if data.Confidence != nil {
			confidence = *data.Confidence
		}
		// With no distribution reported, the winner carries the whole
		// confidence and the rest are filled in at zero below.
		distribution := NewOrderedMap[float64]()
		if data.Probabilities != nil {
			// A Clone, not an assignment: the missing keys are filled in
			// below, and Set through an assignment copy writes into the
			// reply's own map while appending the key only to this one.
			d := data.Probabilities.Clone()
			distribution = &d
		} else {
			distribution.Set(choice, confidence)
		}
		for _, key := range keys {
			if _, ok := distribution.Get(key); !ok {
				distribution.Set(key, 0)
			}
		}
		return Answer{
			Type:          TypeChoice,
			Choice:        choice,
			Confidence:    confidence,
			Probabilities: *distribution,
		}

	case TypeScore:
		legend := NewOrderedMap[string]()
		for i, label := range q.Levels {
			legend.Set(strconv.Itoa(i), label)
		}
		score := 0.0
		if data.Score != nil {
			score = *data.Score
		}
		confidence := 0.5
		if data.Confidence != nil {
			confidence = *data.Confidence
		}
		// No per-level distribution is asked for in one-shot mode, so none is
		// reported: the legend is the only scale information available.
		return Answer{
			Type:       TypeScore,
			Score:      score,
			Confidence: confidence,
			Legend:     *legend,
		}

	default:
		noul := 0.5
		switch {
		case data.Noul != nil:
			noul = *data.Noul
		case data.Probability != nil:
			noul = *data.Probability
		}
		return Answer{Type: TypeNoul, Noul: noul}
	}
}

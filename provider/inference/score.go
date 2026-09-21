package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/titusai-io/perceptea/classifier"
)

// scoreInstruction is the calibrated-estimator instruction every scoring call
// is made under, and the first thing in the prompt. The probabilities a model
// returns are calibrated against this exact wording, so editing it moves the
// numbers.
const scoreInstruction = "You are a calibrated probability estimator. " +
	"Given a STATE and a STATEMENT, return only how likely the statement is true " +
	"based solely on the state. Do not invent facts. " +
	`Respond with JSON: {"p": <number 0 to 1>}.`

// scoreStateLabel and scoreStatementLabel introduce the two things the
// instruction names.
const (
	scoreStateLabel     = "STATE:\n"
	scoreStatementLabel = "STATEMENT:\n"
)

// scoreUserSuffix closes the user turn. The dash is an en dash (U+2013), not a
// hyphen; changing it changes the prompt, and so the probabilities.
const scoreUserSuffix = "\n\nHow likely is the statement true (0–1)?"

// The prompt is cut into two messages, and where the cut falls is the whole
// point of the shape.
//
// The first message is the prefix: the estimator instruction, the question's
// worked examples when it declares any, its candidate list when it has more
// than one candidate, and the state. Every candidate answer of one question is
// judged against exactly those, so the prefix is byte-identical for the whole
// wave — a choice of four, a score of four and a noul are nine calls and three
// prefixes, each sent several times over — and an endpoint that caches a
// matching prompt prefix serves the repeats from the cache. Only the suffix is
// then new work.
//
// The candidate list is the reason a question's own rivals can be shown at no
// cost. It is the same bytes for every candidate of the question, so it rides
// in the cacheable part: no extra call, and no extra uncached token after the
// first candidate.
//
// Nothing that varies by candidate may go in the prefix. One index, one count,
// one reordering and every call diverges at its first differing token; there
// is no error, no warning and no symptom except the bill, which is why a test
// asserts the bytes rather than the intent.
//
// The second message is the suffix, and holds what changes: the one statement
// being judged and the question to answer about it.
//
// No cache-control field is sent with any of it. Prefix caching on these
// endpoints happens automatically on a prefix match, and a field an endpoint
// has never heard of is one more thing for it to reject.

// promptPrefix renders the half of the prompt every candidate of one question
// shares: an instruction, the question's examples when it declares any, its
// candidate list when it has one, and the state. Both scorers build their
// prefix with it and differ only in the instruction they pass, so the layout —
// and the caching property that depends on it — cannot drift between them.
//
// The examples arrive already rendered as one labelled block. The tempting
// alternative is to stage each of them as a prior exchange — a user turn
// putting an example's statement, an assistant turn answering {"p": 0.95} —
// and it cannot be used here: a statement belongs to one candidate, so
// examples written that way differ on every call of the wave and there is no
// shared prefix left to cache. A labelled block teaches the same thing and
// stays identical across the wave.
//
// The candidate list arrives the same way and goes in the same place, after
// the examples and before the state. It names every candidate of the question
// including this one, which is exactly why it may be shared: it says what is
// being chosen among, and nothing about which candidate this call is judging.
// That is still in the suffix, and it is still the only thing the model is
// asked to put a number on.
//
// Either block may be empty — a question with no examples, a question with one
// candidate — and an empty one renders nothing whatever, separators included,
// so such a question produces the prompt it would have produced before either
// existed.
func promptPrefix(instruction, examples, candidates, state string) string {
	var b strings.Builder
	b.WriteString(instruction)
	b.WriteString("\n\n")
	if strings.TrimSpace(examples) != "" {
		b.WriteString(examples)
		b.WriteString("\n\n")
	}
	if strings.TrimSpace(candidates) != "" {
		b.WriteString(candidates)
		b.WriteString("\n\n")
	}
	b.WriteString(scoreStateLabel)
	b.WriteString(state)
	return b.String()
}

// scorePrefix renders the chat scorer's shared prefix.
func scorePrefix(examples, candidates, state string) string {
	return promptPrefix(scoreInstruction, examples, candidates, state)
}

// scoreSuffix renders the half that changes: this one candidate's statement
// and the question about it.
func scoreSuffix(statement string) string {
	return scoreStatementLabel + statement + scoreUserSuffix
}

// maxScoreTokens caps the reply. The answer is a handful of characters; the
// cap is what keeps a chatty model from turning one score into an essay.
const maxScoreTokens = 32

// fallbackProbability is the answer when nothing usable comes back. A
// candidate that cannot be read is maximally uninformative, not an error: one
// bad candidate should blunt one score, not fail the whole evaluation.
//
// The one exception is a reply the provider cut off at the output token
// limit; see [ErrTruncatedReply].
const fallbackProbability = 0.5

// ErrTruncatedReply reports a scoring reply that the provider stopped at the
// output token limit before it said anything this package could read.
//
// It is the one unreadable reply that is an error rather than a 0.5. Every
// other unreadable reply is a one-off: the model said something odd about one
// candidate, and blunting that one score is the proportionate answer. A
// truncated reply is not a one-off — the cap is the same on every call in the
// request, so every candidate comes back unreadable, every score is 0.5, and
// the softmax turns a uniform set of scores into a confident-looking answer
// that carries no information at all. That failure is silent, and silence is
// the worst property an answer can have, so this one fails loudly instead.
//
// Callers classify it with [errors.Is]; the wrapped error carries the detail.
var ErrTruncatedReply = errors.New("inference: the reply was cut off at the output token limit before a probability could be read")

// scoreSchemaName is the json_schema name sent at level one.
const scoreSchemaName = "prob"

// scoreSchema constrains the reply to {"p": <0..1>} and nothing else.
var scoreSchema = json.RawMessage(
	`{"type":"object","properties":{"p":{"type":"number","minimum":0,"maximum":1}},"required":["p"],"additionalProperties":false}`,
)

// probabilityPattern is the salvage regexp, matching the first decimal
// fraction or bare 0/1 in a reply that is not JSON.
var probabilityPattern = regexp.MustCompile(`0?\.\d+|[01](?:\.0+)?`)

// Score asks how likely one statement is, given one state, and returns the
// probability with the tokens it cost.
//
// Which of the two scorers answers is fixed when the client is built, by
// Config.Scorer: the chat scorer by default, and the logprob scorer when the
// client was configured for it. The package doc and Config.Scorer describe
// the difference between them; everything documented below is the first.
func (c *Client) Score(ctx context.Context, req classifier.ScoreRequest) (classifier.ScoreResult, error) {
	if c.scorer == scorerLogprob {
		return c.scoreByLogprob(ctx, req)
	}
	return c.scoreByChat(ctx, req)
}

// scoreByChat asks the model to write a probability and reads the number out
// of what it wrote.
//
// The prompt goes out as a shared prefix and a per-candidate suffix; see the
// note above [promptPrefix] for what may go in which, and why it matters.
//
// The call starts at the client's current structured-output level and steps
// down a level whenever the provider rejects the request shape, so a provider
// that cannot do json_schema still answers on the first Score call.
//
// The step is only remembered — for every later call, on this client — once a
// lower level has actually answered. That is the honest signal: a provider
// that genuinely cannot do json_schema succeeds at json_object, and the
// client should never ask again. A request that fails at all three levels was
// failing for some other reason, and leaves the level where it found it, so
// one bad model name or one over-long prompt cannot quietly strip structured
// output from every later request the client serves.
//
// Anything short of a transport or API failure yields a probability: an empty
// choices list, an unparseable reply and a reply with no number in it all
// degrade to 0.5 rather than returning an error. The exception is a reply that
// was cut off at the output token limit and carried no probability, which
// returns [ErrTruncatedReply]: see there for why that one is not forgiven.
func (c *Client) scoreByChat(ctx context.Context, req classifier.ScoreRequest) (classifier.ScoreResult, error) {
	model, err := c.resolveModel(req.Model)
	if err != nil {
		return classifier.ScoreResult{}, err
	}

	// Two messages, prefix then suffix; see the note above scorePrefix for
	// why the cut falls where it does. The prefix is the system turn and the
	// suffix the user turn, which keeps the exchange to the one shape every
	// endpoint accepts: a chat template that insists its roles alternate
	// would reject two user turns in a row.
	messages := []chatMessage{
		{Role: "system", Content: scorePrefix(req.Examples, req.Candidates, req.State)},
		{Role: "user", Content: scoreSuffix(req.Statement)},
	}

	started := c.outputLevel()
	// The level strictly increases on every step and the step is only taken
	// while a response_format was sent, which stops at levelPlain, so this
	// loop runs at most once per level.
	for level := started; ; level++ {
		body := chatRequest{
			Model:           model,
			Messages:        messages,
			Temperature:     req.Temperature,
			MaxTokens:       maxScoreTokens,
			ResponseFormat:  scoreResponseFormat(level),
			ReasoningEffort: c.reasoningEffort,
		}

		resp, err := c.complete(ctx, body)
		if err == nil {
			if level > started {
				c.downgrade(level)
			}
			return scoreResult(resp)
		}
		if body.ResponseFormat != nil && unsupportedShape(err) {
			continue
		}
		return classifier.ScoreResult{}, err
	}
}

// scoreResponseFormat renders the response_format for a level, or nil at the
// plain level.
func scoreResponseFormat(level outputLevel) *responseFormat {
	switch level {
	case levelJSONSchema:
		return &responseFormat{
			Type: "json_schema",
			JSONSchema: &jsonSchemaSpec{
				Name:   scoreSchemaName,
				Strict: true,
				Schema: scoreSchema,
			},
		}
	case levelJSONObject:
		return &responseFormat{Type: "json_object"}
	default:
		return nil
	}
}

// scoreResult turns a completion into a score, or into the error a reply the
// provider cut short has earned.
func scoreResult(resp *chatResponse) (classifier.ScoreResult, error) {
	out := classifier.ScoreResult{
		Probability:  fallbackProbability,
		InputTokens:  resp.Usage.PromptTokens,
		OutputTokens: resp.Usage.CompletionTokens,
	}
	if len(resp.Choices) == 0 {
		return out, nil
	}

	choice := resp.Choices[0]
	raw := string(choice.Message.Content)
	if strings.TrimSpace(raw) == "" {
		// Some providers put everything in a reasoning field and leave
		// content null.
		raw = choice.Message.reasoning()
	}

	// A truncated reply that still managed to say a number said it before it
	// ran out of room, and is an answer like any other. Only a truncated
	// reply with nothing readable in it is the failure worth reporting.
	if p, ok := probabilityFrom(raw); ok {
		out.Probability = p
		return out, nil
	}
	if choice.truncated() {
		return classifier.ScoreResult{}, truncatedScoreError(choice)
	}
	return out, nil
}

// truncatedScoreError explains one cut-off reply and what to do about it. The
// message is written for whoever has to fix it — an operator reading a 502,
// not a maintainer reading a stack — so it names the cap, the usual cause and
// both fixes.
func truncatedScoreError(choice chatChoice) error {
	detail := fmt.Sprintf("the %d token cap on a scoring call is spent on thinking tokens by a model that reasons, "+
		"so the reply ends before the JSON", maxScoreTokens)
	// The smoking gun, when the reply left it: nothing in content, and a
	// reasoning field with something in it. That is a thinking model under a
	// small cap and nothing else.
	if strings.TrimSpace(string(choice.Message.Content)) == "" {
		if reasoning := strings.TrimSpace(choice.Message.reasoning()); reasoning != "" {
			detail += fmt.Sprintf("; this reply's content was empty while its reasoning field held %d characters",
				len([]rune(reasoning)))
		}
	}
	return fmt.Errorf("%w: %s. Use a model that does not reason, or set %s to %q",
		ErrTruncatedReply, detail, envReasoningEffort, effortNone)
}

// The configuration this package's errors point at. They are spelled out
// rather than imported: a provider client is usable on its own, and must not
// depend on the server that happens to configure it. A test asserts that
// they still match the settings the server actually reads.
const (
	envReasoningEffort = "PERCEPTEA_REASONING_EFFORT"
	effortNone         = "none"

	// envScorer is the setting that picks between the two scorers, and
	// scorerChat and scorerLogprob are the two values it takes. They are
	// also the values [Config.Scorer] itself takes, so the same two
	// spellings serve the library caller and the operator reading an error.
	envScorer     = "PERCEPTEA_SCORER"
	scorerChat    = "chat"
	scorerLogprob = "logprob"
)

// probabilityFrom reads a probability out of a model reply, forgivingly and in
// this order: parse the reply as JSON and read "p" then "probability"; failing
// that, salvage the first number in the raw text. It reports whether it read
// one at all; a caller that has no better idea uses [fallbackProbability].
//
// A markdown fence around the JSON is stripped first.
//
// The salvage regexp only runs on a reply that is not JSON at all. A reply
// that is JSON is read as JSON or not at all: once a reply has a structure,
// the structure is what it meant, and scraping a number out of a document
// that put none under "p" would promote some other field to an answer.
func probabilityFrom(raw string) (float64, bool) {
	if candidate := strings.TrimSpace(stripFence(raw)); candidate != "" {
		var doc json.RawMessage
		if err := json.Unmarshal([]byte(candidate), &doc); err == nil {
			v, ok := probabilityFromJSON(doc)
			if !ok {
				return 0, false
			}
			return clamp01(v), true
		}
	}
	if match := probabilityPattern.FindString(raw); match != "" {
		if v, err := strconv.ParseFloat(match, 64); err == nil {
			return clamp01(v), true
		}
	}
	return 0, false
}

// probabilityFromJSON reads the probability out of a reply that did parse as
// JSON.
//
// Three rules govern how generously a reply is read. All three pull the same
// way: take the answer a model plainly meant, but never invent confidence the
// reply does not carry.
//
//   - "p" and "probability" are matched case-insensitively, because that is
//     how encoding/json resolves a field name. A model that shouts its key
//     still means the same thing, and no other key can collide with these two.
//   - A value that is present but not a number at all — {"p":""}, {"p":" "},
//     {"p":[]} — is read as no answer. Coercing an empty string to 0 would
//     report a confident "certainly false" on the strength of a field the
//     model left blank.
//   - A bare top-level number is accepted. A reply of exactly 0.28 is an
//     answer to the question that was asked, and throwing it away for want of
//     a wrapping object would discard a probability the model did report.
func probabilityFromJSON(doc json.RawMessage) (float64, bool) {
	var obj struct {
		P           json.RawMessage `json:"p"`
		Probability json.RawMessage `json:"probability"`
	}
	if err := json.Unmarshal(doc, &obj); err == nil {
		if v, ok := numberFrom(obj.P); ok {
			return v, true
		}
		if v, ok := numberFrom(obj.Probability); ok {
			return v, true
		}
		return 0, false
	}
	// Not an object. A bare number is unambiguous — but only while it is
	// already a probability. Clamping a bare 12 or 85 to 1 would turn a reply
	// that plainly is not a probability into maximum confidence, and so would
	// handing it to the salvage regexp, which reads the leading "1" out of 12
	// and out of 100. Out of range, the reply says nothing usable.
	if v, ok := numberFrom(doc); ok && v >= 0 && v <= 1 {
		return v, true
	}
	return 0, false
}

// numberFrom coerces a JSON value to a number: numbers pass through, numeric
// strings are parsed, and booleans become 1 and 0. Anything else — a blank
// field, an array, a non-finite value — reports that it is not a number at
// all, which is what keeps a blank field from being read as a zero.
func numberFrom(raw json.RawMessage) (float64, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return 0, false
	}

	var f float64
	if err := json.Unmarshal(trimmed, &f); err == nil {
		return finite(f)
	}

	var s string
	if err := json.Unmarshal(trimmed, &s); err == nil {
		v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err != nil {
			return 0, false
		}
		return finite(v)
	}

	var b bool
	if err := json.Unmarshal(trimmed, &b); err == nil {
		if b {
			return 1, true
		}
		return 0, true
	}

	return 0, false
}

// finite passes an ordinary number through and reports a NaN or an infinity
// as no number at all. Without the guard, a reply of "Infinity" reaches
// clamp01 and comes out as a perfectly confident 1.
func finite(v float64) (float64, bool) {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, false
	}
	return v, true
}

// clamp01 confines a probability to [0,1].
func clamp01(v float64) float64 {
	switch {
	case math.IsNaN(v):
		return fallbackProbability
	case v < 0:
		return 0
	case v > 1:
		return 1
	default:
		return v
	}
}

// stripFence removes a surrounding markdown code fence, with or without a
// language tag. Text that is not fenced is returned unchanged.
func stripFence(s string) string {
	trimmed := strings.TrimSpace(s)
	if !strings.HasPrefix(trimmed, "```") {
		return s
	}
	body := trimmed[3:]
	if end := strings.LastIndex(body, "```"); end >= 0 {
		body = body[:end]
	}
	if i := strings.IndexFunc(body, unicode.IsSpace); i > 0 && isLanguageTag(body[:i]) {
		body = body[i:]
	}
	return strings.TrimSpace(body)
}

// isLanguageTag reports whether a fence's first token looks like "json" rather
// than the start of the payload.
func isLanguageTag(s string) bool {
	if s == "" || len(s) > 16 {
		return false
	}
	for _, r := range s {
		if r != '-' && r != '_' && r != '+' && !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

// Package classifier turns a declared set of questions into typed answers with
// probability distributions, using a model that can only answer one thing:
// "given this state, how likely is this statement true?".
//
// Instead of asking a model to emit one large JSON blob describing every
// answer, each declared candidate answer is scored independently by a tiny
// constrained call, and the independent probabilities are normalised into a
// distribution.
//
// The package depends only on the standard library. Everything network-shaped
// sits behind the [Scorer] and [Generator] interfaces.
package classifier

import (
	"bytes"
	"encoding/json"
	"fmt"
	"iter"
)

// QuestionType names one of the three answer primitives.
type QuestionType string

const (
	// TypeChoice picks exactly one key from a declared set of options.
	TypeChoice QuestionType = "choice"
	// TypeScore rates the state against an ordered list of level labels and
	// returns the expected value across them, so a rating may land between
	// two declared levels.
	TypeScore QuestionType = "score"
	// TypeNoul answers a single proposition. Its value is the probability
	// that the proposition holds, so it carries no separate confidence.
	TypeNoul QuestionType = "noul"
	// TypeBoolean is accepted on input as a synonym for TypeNoul. Decoding
	// normalises it to TypeNoul, and answers are always reported as "noul".
	TypeBoolean QuestionType = "boolean"
)

// Mode selects how a request is evaluated.
type Mode string

const (
	// ModeParallel scores every candidate answer independently and then
	// normalises. It is the default and the reason this package exists.
	ModeParallel Mode = "parallel"
	// ModeOneshot asks the model for one JSON document covering every
	// question. Kept for comparison; it is markedly less reliable.
	ModeOneshot Mode = "oneshot"
)

// OrderedMap is a JSON object that remembers the order its keys appeared in.
//
// Order matters here. The order a choice question's criteria are declared in
// is part of what the request means: it fixes the order the statements are
// built and scored in, and it decides which option wins an exact tie. A Go map
// randomises iteration and would throw that away, so the document's own order
// is recorded alongside the values.
//
// The zero value is an empty map, ready to use. Reads are safe on a value;
// writes need a pointer.
type OrderedMap[V any] struct {
	keys []string
	vals map[string]V
}

// NewOrderedMap returns an empty map.
func NewOrderedMap[V any]() *OrderedMap[V] {
	return &OrderedMap[V]{vals: map[string]V{}}
}

// Clone returns an independent copy.
//
// Copying an OrderedMap by assignment shares its backing storage, so a Set
// through the copy is visible through the original. Take a Clone first
// whenever you intend to mutate a map you did not build.
func (m OrderedMap[V]) Clone() OrderedMap[V] {
	out := OrderedMap[V]{
		keys: make([]string, len(m.keys)),
		vals: make(map[string]V, len(m.vals)),
	}
	copy(out.keys, m.keys)
	for k, v := range m.vals {
		out.vals[k] = v
	}
	return out
}

// Set stores a value, appending the key if it is new and leaving the key's
// existing position alone if it is not.
//
// Writing through a map copied by assignment also writes through the original;
// see [OrderedMap.Clone].
func (m *OrderedMap[V]) Set(key string, v V) {
	if m.vals == nil {
		m.vals = map[string]V{}
	}
	if _, ok := m.vals[key]; !ok {
		m.keys = append(m.keys, key)
	}
	m.vals[key] = v
}

// Get reports the value stored under key.
func (m OrderedMap[V]) Get(key string) (V, bool) {
	v, ok := m.vals[key]
	return v, ok
}

// Len reports how many keys are stored.
func (m OrderedMap[V]) Len() int { return len(m.keys) }

// Keys returns the keys in their original order.
func (m OrderedMap[V]) Keys() []string {
	out := make([]string, len(m.keys))
	copy(out, m.keys)
	return out
}

// Values returns the values in their keys' original order.
func (m OrderedMap[V]) Values() []V {
	out := make([]V, 0, len(m.keys))
	for _, k := range m.keys {
		out = append(out, m.vals[k])
	}
	return out
}

// All iterates the entries in their original order.
func (m OrderedMap[V]) All() iter.Seq2[string, V] {
	return func(yield func(string, V) bool) {
		for _, k := range m.keys {
			if !yield(k, m.vals[k]) {
				return
			}
		}
	}
}

// MarshalJSON writes a JSON object with the keys in their original order.
func (m OrderedMap[V]) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, k := range m.keys {
		if i > 0 {
			buf.WriteByte(',')
		}
		key, err := json.Marshal(k)
		if err != nil {
			return nil, err
		}
		buf.Write(key)
		buf.WriteByte(':')
		val, err := json.Marshal(m.vals[k])
		if err != nil {
			return nil, fmt.Errorf("key %q: %w", k, err)
		}
		buf.Write(val)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// UnmarshalJSON reads a JSON object, recording the order of its keys. A JSON
// null decodes to an empty map. A duplicate key keeps the position of its
// first appearance and the value of its last, which is how the standard
// decoder resolves one and leaves the key order unambiguous.
func (m *OrderedMap[V]) UnmarshalJSON(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	if tok == nil {
		*m = OrderedMap[V]{}
		return nil
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '{' {
		return fmt.Errorf("expected a JSON object, got %s", firstToken(data))
	}
	out := OrderedMap[V]{vals: map[string]V{}}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return err
		}
		key, ok := keyTok.(string)
		if !ok {
			return fmt.Errorf("expected an object key, got %v", keyTok)
		}
		var v V
		if err := dec.Decode(&v); err != nil {
			return fmt.Errorf("key %q: %w", key, err)
		}
		out.Set(key, v)
	}
	if _, err := dec.Token(); err != nil {
		return err
	}
	*m = out
	return nil
}

// firstToken renders the leading token of a JSON document for error messages.
func firstToken(data []byte) string {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return "empty input"
	}
	if len(trimmed) > 32 {
		trimmed = append(trimmed[:32:32], "..."...)
	}
	return string(trimmed)
}

// NoulLabels optionally describes what "true" and "false" mean for a noul
// question. Nothing reads them — a noul statement is built from the question's
// instruction alone — but they are kept so a request can be round-tripped.
type NoulLabels struct {
	True  string `json:"true,omitempty"`
	False string `json:"false,omitempty"`
}

// Example is one worked answer, shown to the model so that a criterion can be
// taught by demonstration rather than only described. Which field carries the
// answer depends on the question's type.
//
// Examples are optional everywhere. A question without them produces exactly
// the prompt it would have produced before they existed.
type Example struct {
	// State is the material the example is about, in the same shapes a
	// request's own state may take.
	State State
	// Choice is the correct option key, for a choice question.
	Choice string
	// Level is the correct level index, for a score question.
	Level int
	// Noul is whether the proposition holds, for a noul question.
	Noul bool
}

// exampleWire is the JSON shape of an example: a state and an answer whose
// type follows the question's. A choice answers with its option key, a score
// with its level index, a noul with true or false.
type exampleWire struct {
	State  State           `json:"state"`
	Answer json.RawMessage `json:"answer"`
}

// Question is one declared question. Which fields carry meaning depends on
// Type: Options for a choice, Levels for a score, Labels for a noul.
type Question struct {
	Type         QuestionType
	Instructions string

	// Options holds a choice question's criteria: an ordered map of option
	// key to the description shown to the model.
	Options OrderedMap[string]
	// Levels holds a score question's criteria: ordered level labels, whose
	// indices are the scale.
	Levels []string
	// Labels holds a noul question's optional true/false descriptions.
	Labels NoulLabels
	// Examples are optional worked answers for this question. They are
	// rendered into the part of the prompt that is identical for every
	// candidate, so that adding them does not cost the shared prefix a
	// provider may be caching.
	Examples []Example

	// declared is the type exactly as the request document wrote it, kept
	// because Type normalises the TypeBoolean synonym away. The declared
	// word is what the one-shot prompt shows the model and what the question
	// is encoded back out as, so it has to survive that normalisation. It is
	// empty on a Question built in Go rather than decoded; see
	// [Question.DeclaredType].
	declared QuestionType
}

// DeclaredType reports the type the request document declared, which is the
// same as Type except for a question written as "boolean": Type normalises
// that to TypeNoul while DeclaredType still reports TypeBoolean.
//
// A Question built programmatically declares nothing, so it reports Type.
//
// This is the wording the model is shown in [ModeOneshot] and the wording a
// question is encoded back out as. It never reaches an [Answer]: a question
// written either way is answered identically, so an answer's type is always
// TypeNoul.
func (q Question) DeclaredType() QuestionType {
	if q.declared == "" {
		return q.Type
	}
	return q.declared
}

// questionWire is the JSON shape of a question. criteria is polymorphic — an
// object for a choice, an array for a score, an optional object for a noul —
// so it is decoded in a second pass once the type is known.
type questionWire struct {
	Type         QuestionType    `json:"type"`
	Instructions string          `json:"instructions,omitempty"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
	Examples     []exampleWire   `json:"examples,omitempty"`
}

// UnmarshalJSON decodes a question, normalising the "boolean" type synonym to
// TypeNoul and interpreting criteria according to the declared type.
func (q *Question) UnmarshalJSON(data []byte) error {
	var wire questionWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}

	out := Question{Type: wire.Type, Instructions: wire.Instructions, declared: wire.Type}
	switch wire.Type {
	case TypeChoice:
		if len(wire.Criteria) > 0 {
			if err := json.Unmarshal(wire.Criteria, &out.Options); err != nil {
				return fmt.Errorf("choice criteria must be an object of option keys to descriptions: %w", err)
			}
		}
	case TypeScore:
		if len(wire.Criteria) > 0 {
			if err := json.Unmarshal(wire.Criteria, &out.Levels); err != nil {
				return fmt.Errorf("score criteria must be an array of level labels: %w", err)
			}
		}
	case TypeNoul, TypeBoolean:
		out.Type = TypeNoul
		if len(wire.Criteria) > 0 && !bytes.Equal(bytes.TrimSpace(wire.Criteria), []byte("null")) {
			if err := json.Unmarshal(wire.Criteria, &out.Labels); err != nil {
				return fmt.Errorf("noul criteria must be an object with optional \"true\" and \"false\" descriptions: %w", err)
			}
		}
	case "":
		return fmt.Errorf("question is missing %q; expected one of %q, %q, %q", "type", TypeChoice, TypeScore, TypeNoul)
	default:
		return fmt.Errorf("unknown question type %q; expected one of %q, %q, %q", wire.Type, TypeChoice, TypeScore, TypeNoul)
	}

	examples, err := decodeExamples(out.Type, wire.Examples)
	if err != nil {
		return err
	}
	out.Examples = examples

	*q = out
	return nil
}

// decodeExamples reads each example's answer in the terms its question type
// uses. The answer is decoded here rather than left raw so that a wrong shape
// is a decode error naming the example, not a surprise at prompt-building
// time when there is no good way to report it.
func decodeExamples(t QuestionType, wire []exampleWire) ([]Example, error) {
	if len(wire) == 0 {
		return nil, nil
	}
	out := make([]Example, 0, len(wire))
	for i, w := range wire {
		ex := Example{State: w.State}
		if w.State.IsZero() {
			return nil, fmt.Errorf("example %d is missing %q", i, "state")
		}
		switch t {
		case TypeChoice:
			if err := json.Unmarshal(w.Answer, &ex.Choice); err != nil {
				return nil, fmt.Errorf("example %d: a choice question's example answers with an option key, as a string: %w", i, err)
			}
		case TypeScore:
			if err := json.Unmarshal(w.Answer, &ex.Level); err != nil {
				return nil, fmt.Errorf("example %d: a score question's example answers with a level index, as a whole number: %w", i, err)
			}
		case TypeNoul, TypeBoolean:
			if err := json.Unmarshal(w.Answer, &ex.Noul); err != nil {
				return nil, fmt.Errorf("example %d: a noul question's example answers with true or false: %w", i, err)
			}
		}
		out = append(out, ex)
	}
	return out, nil
}

// MarshalJSON writes the question back in its wire shape, with the type as it
// was declared: a question decoded from the "boolean" synonym is written back
// as "boolean", so a request round-trips to the document it arrived as. Only
// the answer normalises.
func (q Question) MarshalJSON() ([]byte, error) {
	wire := questionWire{Type: q.DeclaredType(), Instructions: q.Instructions}
	switch q.Type {
	case TypeChoice:
		raw, err := json.Marshal(q.Options)
		if err != nil {
			return nil, err
		}
		wire.Criteria = raw
	case TypeScore:
		raw, err := json.Marshal(q.Levels)
		if err != nil {
			return nil, err
		}
		wire.Criteria = raw
	case TypeNoul, TypeBoolean:
		if q.Labels != (NoulLabels{}) {
			raw, err := json.Marshal(q.Labels)
			if err != nil {
				return nil, err
			}
			wire.Criteria = raw
		}
	}
	return json.Marshal(wire)
}

// Questions is the set of questions in a request, in document order.
type Questions = OrderedMap[Question]

// State is the material the questions are asked about: a JSON string, object
// or array. The raw bytes are kept so that object key order survives into the
// prompt, which a decode into map[string]any would lose.
type State struct {
	raw json.RawMessage
}

// StringState wraps a plain text state.
func StringState(s string) State {
	raw, _ := json.Marshal(s)
	return State{raw: raw}
}

// JSONState wraps any value that can be marshalled to JSON.
func JSONState(v any) (State, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return State{}, err
	}
	return State{raw: raw}, nil
}

// RawState wraps already-encoded JSON without copying it through a value.
func RawState(raw json.RawMessage) State { return State{raw: raw} }

// IsZero reports whether no state was supplied at all. A state present but
// explicitly null is not zero.
func (s State) IsZero() bool { return len(bytes.TrimSpace(s.raw)) == 0 }

// IsNull reports whether the state was supplied as JSON null.
func (s State) IsNull() bool { return bytes.Equal(bytes.TrimSpace(s.raw), []byte("null")) }

// Raw returns the underlying JSON bytes.
func (s State) Raw() json.RawMessage { return s.raw }

// Text renders the state as it is put to the model: a JSON string is used
// verbatim, and anything else is re-indented with two spaces, preserving the
// original key order.
//
// Re-indenting rather than decoding and re-encoding is deliberate. These stay
// the caller's own bytes, so nothing is canonicalised on the way through: 1.50
// reaches the model as 1.50 rather than 1.5, 1e3 stays 1e3 rather than
// becoming 1000, and a duplicate key appears twice rather than collapsing to
// its last value. That is the cost. What it buys is that key order survives,
// which a decode into map[string]any would lose, and key order is what the
// prompt — and therefore the probability — depends on. Odd number literals and
// duplicate keys are rare in a state document and harmless in a prompt; a
// reordered state is neither.
func (s State) Text() (string, error) {
	trimmed := bytes.TrimSpace(s.raw)
	if len(trimmed) == 0 {
		return "", nil
	}
	if trimmed[0] == '"' {
		var str string
		if err := json.Unmarshal(trimmed, &str); err != nil {
			return "", err
		}
		return str, nil
	}
	var buf bytes.Buffer
	if err := json.Indent(&buf, trimmed, "", "  "); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// MarshalJSON writes the state's raw bytes.
func (s State) MarshalJSON() ([]byte, error) {
	if len(s.raw) == 0 {
		return []byte("null"), nil
	}
	return s.raw, nil
}

// UnmarshalJSON keeps a copy of the raw bytes.
func (s *State) UnmarshalJSON(data []byte) error {
	s.raw = append(json.RawMessage(nil), data...)
	return nil
}

// Request is one evaluation: a state, the questions to ask about it, and how
// to ask them.
type Request struct {
	State     State     `json:"state"`
	Questions Questions `json:"questions"`

	// Model is passed through to the scorer and echoed in the response. An
	// empty value leaves the choice to the scorer's own configuration.
	Model string `json:"model,omitempty"`
	// Temperature is the sampling temperature for the underlying calls. The
	// default, and the only value that makes the result reproducible, is 0.
	Temperature float64 `json:"temperature,omitempty"`
	// Mode defaults to ModeParallel.
	Mode Mode `json:"mode,omitempty"`
}

// Answer is the typed result for one question. Which fields carry meaning
// depends on Type, and JSON encoding emits only those fields:
//
//	choice → choice, confidence, probabilities
//	score  → score, confidence, legend, probabilities
//	noul   → noul
type Answer struct {
	Type QuestionType

	// Choice is the winning option key, for a choice answer.
	Choice string
	// Score is the expected value across level indices, for a score answer.
	Score float64
	// Noul is the probability the proposition holds, for a noul answer.
	Noul float64
	// Confidence combines the winning probability with its margin over the
	// runner-up. Reported for choice and score answers.
	Confidence float64
	// Legend maps a score answer's level indices to their labels.
	Legend OrderedMap[string]
	// Probabilities is the normalised distribution over candidates, keyed by
	// option key for a choice and by level index for a score.
	Probabilities OrderedMap[float64]
}

// MarshalJSON emits only the fields that carry meaning for the answer's type.
func (a Answer) MarshalJSON() ([]byte, error) {
	switch a.Type {
	case TypeChoice:
		return json.Marshal(struct {
			Type          QuestionType        `json:"type"`
			Choice        string              `json:"choice"`
			Confidence    float64             `json:"confidence"`
			Probabilities OrderedMap[float64] `json:"probabilities"`
		}{a.Type, a.Choice, a.Confidence, a.Probabilities})
	case TypeScore:
		payload := struct {
			Type          QuestionType         `json:"type"`
			Score         float64              `json:"score"`
			Confidence    float64              `json:"confidence"`
			Legend        OrderedMap[string]   `json:"legend"`
			Probabilities *OrderedMap[float64] `json:"probabilities,omitempty"`
		}{Type: a.Type, Score: a.Score, Confidence: a.Confidence, Legend: a.Legend}
		if a.Probabilities.Len() > 0 {
			p := a.Probabilities
			payload.Probabilities = &p
		}
		return json.Marshal(payload)
	case TypeNoul, TypeBoolean:
		return json.Marshal(struct {
			Type QuestionType `json:"type"`
			Noul float64      `json:"noul"`
		}{TypeNoul, a.Noul})
	default:
		return nil, fmt.Errorf("cannot encode answer with unknown type %q", a.Type)
	}
}

// answerWire is the union of every answer field, for decoding.
type answerWire struct {
	Type          QuestionType        `json:"type"`
	Choice        string              `json:"choice"`
	Score         float64             `json:"score"`
	Noul          float64             `json:"noul"`
	Confidence    float64             `json:"confidence"`
	Legend        OrderedMap[string]  `json:"legend"`
	Probabilities OrderedMap[float64] `json:"probabilities"`
}

// UnmarshalJSON decodes any of the three answer shapes.
func (a *Answer) UnmarshalJSON(data []byte) error {
	var wire answerWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	if wire.Type == TypeBoolean {
		wire.Type = TypeNoul
	}
	*a = Answer(wire)
	return nil
}

// Answers holds one answer per question, in the order the questions were
// declared.
type Answers = OrderedMap[Answer]

// Usage reports the tokens the evaluation consumed. A field is null when the
// provider reported nothing, which is a different claim from a count of zero.
type Usage struct {
	InputTokens  *int `json:"input_tokens"`
	OutputTokens *int `json:"output_tokens"`
}

// Meta describes how the evaluation ran.
type Meta struct {
	Mode          Mode  `json:"mode"`
	LatencyMS     int64 `json:"latency_ms"`
	ParallelCalls int   `json:"parallel_calls"`
}

// Response is the result of an evaluation.
type Response struct {
	Model   string  `json:"model"`
	Answers Answers `json:"answers"`
	Usage   Usage   `json:"usage"`
	Meta    *Meta   `json:"meta,omitempty"`
}

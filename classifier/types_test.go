package classifier

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestOrderedMapPreservesDocumentOrder(t *testing.T) {
	// Keys chosen so that alphabetical, reverse-alphabetical and insertion
	// order all differ: a plain Go map would pass an alphabetical assertion
	// by accident.
	const src = `{"zebra":"z","apple":"a","mango":"m"}`
	var m OrderedMap[string]
	if err := json.Unmarshal([]byte(src), &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	want := []string{"zebra", "apple", "mango"}
	if got := m.Keys(); !equalStrings(got, want) {
		t.Errorf("Keys() = %v, want %v", got, want)
	}
	out, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(out) != src {
		t.Errorf("round trip = %s, want %s", out, src)
	}
}

func TestOrderedMapDuplicateKeyKeepsFirstPositionAndLastValue(t *testing.T) {
	var m OrderedMap[string]
	if err := json.Unmarshal([]byte(`{"a":"1","b":"2","a":"3"}`), &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got := m.Keys(); !equalStrings(got, []string{"a", "b"}) {
		t.Errorf("Keys() = %v, want [a b]", got)
	}
	if v, _ := m.Get("a"); v != "3" {
		t.Errorf("Get(a) = %q, want %q", v, "3")
	}
}

func TestOrderedMapZeroValueAndNull(t *testing.T) {
	var zero OrderedMap[float64]
	out, err := json.Marshal(zero)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(out) != "{}" {
		t.Errorf("zero value marshals to %s, want {}", out)
	}
	if zero.Len() != 0 {
		t.Errorf("zero value has %d entries, want 0", zero.Len())
	}
	if _, ok := zero.Get("anything"); ok {
		t.Error("zero value returned a value")
	}

	var fromNull OrderedMap[float64]
	if err := json.Unmarshal([]byte(`null`), &fromNull); err != nil {
		t.Fatalf("unmarshal null: %v", err)
	}
	if fromNull.Len() != 0 {
		t.Errorf("null decoded to %d entries, want 0", fromNull.Len())
	}
}

func TestOrderedMapRejectsNonObject(t *testing.T) {
	for _, src := range []string{`[1,2]`, `"text"`, `7`, `true`} {
		var m OrderedMap[string]
		if err := json.Unmarshal([]byte(src), &m); err == nil {
			t.Errorf("unmarshal(%s) succeeded, want an error", src)
		}
	}
}

func TestOrderedMapSetIsAppendOnlyForNewKeys(t *testing.T) {
	m := NewOrderedMap[int]()
	m.Set("b", 1)
	m.Set("a", 2)
	m.Set("b", 3)
	if got := m.Keys(); !equalStrings(got, []string{"b", "a"}) {
		t.Errorf("Keys() = %v, want [b a]", got)
	}
	if got := m.Values(); len(got) != 2 || got[0] != 3 || got[1] != 2 {
		t.Errorf("Values() = %v, want [3 2]", got)
	}
}

func TestOrderedMapCloneIsIndependent(t *testing.T) {
	original := NewOrderedMap[int]()
	original.Set("a", 1)

	// An assignment copy aliases the original; this documents why Clone
	// exists rather than asserting that aliasing is desirable.
	alias := *original
	alias.Set("a", 99)
	if v, _ := original.Get("a"); v != 99 {
		t.Fatalf("assignment copy no longer aliases; Clone's rationale has changed")
	}

	clone := original.Clone()
	clone.Set("a", 7)
	clone.Set("b", 8)
	if v, _ := original.Get("a"); v != 99 {
		t.Errorf("writing through the clone changed the original: Get(a) = %d, want 99", v)
	}
	if _, ok := original.Get("b"); ok {
		t.Error("appending to the clone added a key to the original")
	}
	if v, _ := clone.Get("a"); v != 7 {
		t.Errorf("clone Get(a) = %d, want 7", v)
	}
}

func TestOrderedMapAllIteratesInOrderAndStops(t *testing.T) {
	var m OrderedMap[int]
	if err := json.Unmarshal([]byte(`{"x":1,"y":2,"z":3}`), &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	var seen []string
	for k, v := range m.All() {
		seen = append(seen, k)
		if v == 2 {
			break
		}
	}
	if !equalStrings(seen, []string{"x", "y"}) {
		t.Errorf("iteration visited %v, want [x y]", seen)
	}
}

func TestQuestionDecodesEachCriteriaShape(t *testing.T) {
	for _, tc := range []struct {
		name  string
		src   string
		check func(*testing.T, Question)
	}{
		{
			name: "choice keeps option order",
			src:  `{"type":"choice","instructions":"Which team?","criteria":{"billing":"Charges","technical":"Bugs"}}`,
			check: func(t *testing.T, q Question) {
				if q.Type != TypeChoice {
					t.Errorf("Type = %q, want choice", q.Type)
				}
				if got := q.Options.Keys(); !equalStrings(got, []string{"billing", "technical"}) {
					t.Errorf("option order = %v, want [billing technical]", got)
				}
				if d, _ := q.Options.Get("billing"); d != "Charges" {
					t.Errorf("billing description = %q", d)
				}
			},
		},
		{
			name: "score keeps level order",
			src:  `{"type":"score","criteria":["Low","Medium","High"]}`,
			check: func(t *testing.T, q Question) {
				if len(q.Levels) != 3 || q.Levels[0] != "Low" || q.Levels[2] != "High" {
					t.Errorf("Levels = %v", q.Levels)
				}
			},
		},
		{
			name: "noul without criteria",
			src:  `{"type":"noul","instructions":"The customer is angry"}`,
			check: func(t *testing.T, q Question) {
				if q.Type != TypeNoul || q.Instructions != "The customer is angry" {
					t.Errorf("got %+v", q)
				}
			},
		},
		{
			name: "noul with true/false labels",
			src:  `{"type":"noul","criteria":{"true":"yes it is","false":"no it is not"}}`,
			check: func(t *testing.T, q Question) {
				if q.Labels.True != "yes it is" || q.Labels.False != "no it is not" {
					t.Errorf("Labels = %+v", q.Labels)
				}
			},
		},
		{
			name: "boolean normalises to noul",
			src:  `{"type":"boolean","instructions":"Toxic?"}`,
			check: func(t *testing.T, q Question) {
				if q.Type != TypeNoul {
					t.Errorf("Type = %q, want noul", q.Type)
				}
			},
		},
		{
			name: "null criteria is tolerated",
			src:  `{"type":"noul","criteria":null}`,
			check: func(t *testing.T, q Question) {
				if q.Labels != (NoulLabels{}) {
					t.Errorf("Labels = %+v, want zero", q.Labels)
				}
			},
		},
		{
			name: "unknown fields are ignored",
			src:  `{"type":"score","criteria":["a","b"],"weight":3}`,
			check: func(t *testing.T, q Question) {
				if len(q.Levels) != 2 {
					t.Errorf("Levels = %v", q.Levels)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var q Question
			if err := json.Unmarshal([]byte(tc.src), &q); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			tc.check(t, q)
		})
	}
}

func TestQuestionRejectsBadInput(t *testing.T) {
	for _, tc := range []struct {
		name, src, wantSubstring string
	}{
		{"missing type", `{"instructions":"x"}`, "type"},
		{"unknown type", `{"type":"ranking"}`, "ranking"},
		{"choice criteria as an array", `{"type":"choice","criteria":["a"]}`, "object"},
		{"score criteria as an object", `{"type":"score","criteria":{"a":"b"}}`, "array"},
		{"choice description not a string", `{"type":"choice","criteria":{"a":3}}`, "a"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var q Question
			err := json.Unmarshal([]byte(tc.src), &q)
			if err == nil {
				t.Fatalf("decoded %s without error, got %+v", tc.src, q)
			}
			if !strings.Contains(err.Error(), tc.wantSubstring) {
				t.Errorf("error %q does not mention %q", err, tc.wantSubstring)
			}
		})
	}
}

// An example that gives no answer is not an example, and the three types have
// to agree about that.
//
// JSON null unmarshals into an int and into a bool without complaint, leaving
// the Go zero value behind: level 0 and false. Both are answers the question
// could have given, so nothing downstream objects, and the model is taught
// `the correct rating is level 0` for an example the request never answered —
// on every candidate of the question, in the part of the prompt a provider
// caches. A choice escapes this only by accident, because "" is not a declared
// option key; the accident is not a policy, and a missing key is the same
// omission as an explicit null.
func TestAnExampleThatAnswersNothingIsRejected(t *testing.T) {
	for _, tc := range []struct {
		name, src string
	}{
		{"a score answering null", `{"type":"score","criteria":["Low","High"],
			"examples":[{"state":"no rush","answer":null}]}`},
		{"a score with no answer key", `{"type":"score","criteria":["Low","High"],
			"examples":[{"state":"no rush"}]}`},
		{"a noul answering null", `{"type":"noul","instructions":"Angry?",
			"examples":[{"state":"thanks!","answer":null}]}`},
		{"a noul with no answer key", `{"type":"noul","instructions":"Angry?",
			"examples":[{"state":"thanks!"}]}`},
		{"a boolean answering null", `{"type":"boolean","instructions":"Angry?",
			"examples":[{"state":"thanks!","answer":null}]}`},
		{"a choice answering null", `{"type":"choice","criteria":{"billing":"money"},
			"examples":[{"state":"charged twice","answer":null}]}`},
		{"a choice with no answer key", `{"type":"choice","criteria":{"billing":"money"},
			"examples":[{"state":"charged twice"}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var q Question
			err := json.Unmarshal([]byte(tc.src), &q)
			if err == nil {
				t.Fatalf("decoded an example with no answer without error, got %+v", q.Examples)
			}
			// The exact sentence, not merely the word: a decode that
			// stumbled over the empty bytes produces a message mentioning
			// "answers" too, and it describes the wrong fault.
			if want := `example 0 is missing "answer"`; !strings.Contains(err.Error(), want) {
				t.Errorf("error %q, want it to say %q", err, want)
			}
		})
	}

	// The answers that are genuinely zero are answers, and still decode.
	for _, tc := range []struct {
		name, src string
		check     func(*testing.T, Example)
	}{
		{"level zero", `{"type":"score","criteria":["Low","High"],
			"examples":[{"state":"no rush","answer":0}]}`,
			func(t *testing.T, ex Example) {
				if ex.Level != 0 {
					t.Errorf("Level = %d, want 0", ex.Level)
				}
			}},
		{"false", `{"type":"noul","instructions":"Angry?",
			"examples":[{"state":"thanks!","answer":false}]}`,
			func(t *testing.T, ex Example) {
				if ex.Noul {
					t.Error("Noul = true, want false")
				}
			}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var q Question
			if err := json.Unmarshal([]byte(tc.src), &q); err != nil {
				t.Fatalf("a zero answer is an answer, but decoding failed: %v", err)
			}
			if len(q.Examples) != 1 {
				t.Fatalf("got %d examples, want 1", len(q.Examples))
			}
			tc.check(t, q.Examples[0])
		})
	}
}

// TestQuestionRoundTripsTheDeclaredType checks a decoded question encodes back
// to the document it arrived as, synonym included. Type normalises "boolean"
// away because evaluation has only three answer shapes; the declared word is
// kept beside it because the prompt and the wire format both need the word the
// caller wrote.
func TestQuestionRoundTripsTheDeclaredType(t *testing.T) {
	for _, src := range []string{
		`{"type":"boolean","instructions":"Toxic?"}`,
		`{"type":"noul","instructions":"Toxic?"}`,
		`{"type":"boolean","criteria":{"true":"yes","false":"no"}}`,
		`{"type":"choice","criteria":{"a":"A"}}`,
		`{"type":"score","criteria":["Low","High"]}`,
	} {
		var q Question
		if err := json.Unmarshal([]byte(src), &q); err != nil {
			t.Fatalf("unmarshal %s: %v", src, err)
		}
		out, err := json.Marshal(q)
		if err != nil {
			t.Fatalf("marshal %s: %v", src, err)
		}
		if string(out) != src {
			t.Errorf("round trip changed the question:\n got %s\nwant %s", out, src)
		}
	}
}

func TestQuestionDeclaredType(t *testing.T) {
	for _, tc := range []struct {
		src          string
		wantType     QuestionType
		wantDeclared QuestionType
	}{
		{src: `{"type":"boolean"}`, wantType: TypeNoul, wantDeclared: TypeBoolean},
		{src: `{"type":"noul"}`, wantType: TypeNoul, wantDeclared: TypeNoul},
		{src: `{"type":"score","criteria":["a","b"]}`, wantType: TypeScore, wantDeclared: TypeScore},
	} {
		var q Question
		if err := json.Unmarshal([]byte(tc.src), &q); err != nil {
			t.Fatalf("unmarshal %s: %v", tc.src, err)
		}
		if q.Type != tc.wantType {
			t.Errorf("%s: Type = %q, want %q", tc.src, q.Type, tc.wantType)
		}
		if got := q.DeclaredType(); got != tc.wantDeclared {
			t.Errorf("%s: DeclaredType() = %q, want %q", tc.src, got, tc.wantDeclared)
		}
	}

	// A question built in Go declares nothing, so it reports its own type
	// rather than an empty string.
	for _, typ := range []QuestionType{TypeNoul, TypeBoolean, TypeChoice, TypeScore} {
		if got := (Question{Type: typ}).DeclaredType(); got != typ {
			t.Errorf("a programmatic %q question reports DeclaredType() = %q", typ, got)
		}
	}
	if got, err := json.Marshal(Question{Type: TypeBoolean}); err != nil || string(got) != `{"type":"boolean"}` {
		t.Errorf("marshalling a programmatic boolean question gave %s, %v", got, err)
	}
}

func TestQuestionsKeepDocumentOrder(t *testing.T) {
	const src = `{"urgency":{"type":"score","criteria":["Low","High"]},` +
		`"department":{"type":"choice","criteria":{"billing":"b"}},` +
		`"angry":{"type":"noul"}}`
	var qs Questions
	if err := json.Unmarshal([]byte(src), &qs); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got := qs.Keys(); !equalStrings(got, []string{"urgency", "department", "angry"}) {
		t.Errorf("question order = %v", got)
	}
}

func TestAnswerMarshalsOnlyTheFieldsItsTypeCarries(t *testing.T) {
	legend := NewOrderedMap[string]()
	legend.Set("0", "Low")
	legend.Set("1", "High")
	probs := NewOrderedMap[float64]()
	probs.Set("0", 0.25)
	probs.Set("1", 0.75)

	for _, tc := range []struct {
		name   string
		answer Answer
		want   string
	}{
		{
			name:   "choice",
			answer: Answer{Type: TypeChoice, Choice: "billing", Confidence: 0.8, Probabilities: *probs, Score: 9, Noul: 9},
			want:   `{"type":"choice","choice":"billing","confidence":0.8,"probabilities":{"0":0.25,"1":0.75}}`,
		},
		{
			name:   "score with probabilities",
			answer: Answer{Type: TypeScore, Score: 1.5, Confidence: 0.6, Legend: *legend, Probabilities: *probs},
			want:   `{"type":"score","score":1.5,"confidence":0.6,"legend":{"0":"Low","1":"High"},"probabilities":{"0":0.25,"1":0.75}}`,
		},
		{
			name:   "score without probabilities omits the field",
			answer: Answer{Type: TypeScore, Score: 0, Confidence: 0.5, Legend: *legend},
			want:   `{"type":"score","score":0,"confidence":0.5,"legend":{"0":"Low","1":"High"}}`,
		},
		{
			name:   "noul keeps a zero value",
			answer: Answer{Type: TypeNoul, Noul: 0, Confidence: 0.9, Choice: "ignored"},
			want:   `{"type":"noul","noul":0}`,
		},
		{
			name:   "boolean is reported as noul",
			answer: Answer{Type: TypeBoolean, Noul: 0.4},
			want:   `{"type":"noul","noul":0.4}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := json.Marshal(tc.answer)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(got) != tc.want {
				t.Errorf("got  %s\nwant %s", got, tc.want)
			}
		})
	}
}

func TestAnswerUnknownTypeFailsToMarshal(t *testing.T) {
	if _, err := json.Marshal(Answer{Type: "ranking"}); err == nil {
		t.Error("marshalling an unknown answer type succeeded, want an error")
	}
}

func TestAnswerRoundTrip(t *testing.T) {
	const src = `{"type":"score","score":2.4,"confidence":0.75,"legend":{"0":"Low","1":"High"},"probabilities":{"0":0.3,"1":0.7}}`
	var a Answer
	if err := json.Unmarshal([]byte(src), &a); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if a.Type != TypeScore || a.Score != 2.4 || a.Confidence != 0.75 {
		t.Fatalf("decoded %+v", a)
	}
	if got := a.Legend.Keys(); !equalStrings(got, []string{"0", "1"}) {
		t.Errorf("legend order = %v", got)
	}
	out, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(out) != src {
		t.Errorf("round trip = %s\nwant        %s", out, src)
	}
}

func TestStateText(t *testing.T) {
	for _, tc := range []struct {
		name, src, want string
	}{
		{"plain string is used verbatim", `"Charged twice again!!"`, "Charged twice again!!"},
		{"string escapes are decoded", `"line one\nline two"`, "line one\nline two"},
		{
			name: "object keeps key order and two-space indent",
			src:  `{"subject":"Charged twice","message":"again"}`,
			want: "{\n  \"subject\": \"Charged twice\",\n  \"message\": \"again\"\n}",
		},
		{
			name: "array is indented",
			src:  `["a","b"]`,
			want: "[\n  \"a\",\n  \"b\"\n]",
		},
		{"number", `7`, "7"},
		{"null", `null`, "null"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var s State
			if err := json.Unmarshal([]byte(tc.src), &s); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			got, err := s.Text()
			if err != nil {
				t.Fatalf("Text: %v", err)
			}
			if got != tc.want {
				t.Errorf("Text() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestStateTextPinsTheExactRendering pins the two-space layout Text produces
// for an object state, nested array included, down to the last space and
// newline. The expected string was written out by hand rather than captured
// from Text, so it can catch Text changing.
func TestStateTextPinsTheExactRendering(t *testing.T) {
	var s State
	src := `{"subject":"Charged twice again!!","tags":["a","b"],"n":3}`
	if err := json.Unmarshal([]byte(src), &s); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	const want = "{\n  \"subject\": \"Charged twice again!!\",\n  \"tags\": [\n    \"a\",\n    \"b\"\n  ],\n  \"n\": 3\n}"
	got, err := s.Text()
	if err != nil {
		t.Fatalf("Text: %v", err)
	}
	if got != want {
		t.Errorf("Text() = %q\n  want = %q", got, want)
	}
}

func TestStateZeroAndNull(t *testing.T) {
	var absent State
	if !absent.IsZero() {
		t.Error("a State that was never set should report IsZero")
	}
	if absent.IsNull() {
		t.Error("an absent State is not a null State")
	}
	text, err := absent.Text()
	if err != nil || text != "" {
		t.Errorf("Text() on an absent state = %q, %v", text, err)
	}
	out, err := json.Marshal(absent)
	if err != nil || string(out) != "null" {
		t.Errorf("marshal of an absent state = %s, %v", out, err)
	}

	var explicitNull State
	if err := json.Unmarshal([]byte(`null`), &explicitNull); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if explicitNull.IsZero() {
		t.Error("an explicitly null state is present, not zero")
	}
	if !explicitNull.IsNull() {
		t.Error("an explicitly null state should report IsNull")
	}
}

func TestStateConstructors(t *testing.T) {
	if got, _ := StringState("hi").Text(); got != "hi" {
		t.Errorf("StringState.Text() = %q", got)
	}
	s, err := JSONState(map[string]int{"a": 1})
	if err != nil {
		t.Fatalf("JSONState: %v", err)
	}
	if got, _ := s.Text(); got != "{\n  \"a\": 1\n}" {
		t.Errorf("JSONState.Text() = %q", got)
	}
	if got, _ := RawState(json.RawMessage(`["x"]`)).Text(); got != "[\n  \"x\"\n]" {
		t.Errorf("RawState.Text() = %q", got)
	}
}

func TestUsageEncodesNullWhenUnreported(t *testing.T) {
	out, err := json.Marshal(Usage{})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(out) != `{"input_tokens":null,"output_tokens":null}` {
		t.Errorf("got %s", out)
	}
	in, o := 10, 20
	out, err = json.Marshal(Usage{InputTokens: &in, OutputTokens: &o})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(out) != `{"input_tokens":10,"output_tokens":20}` {
		t.Errorf("got %s", out)
	}
}

func TestRequestDecodesTheDocumentedShape(t *testing.T) {
	const src = `{
	  "state": {"subject":"Charged twice"},
	  "questions": {
	    "department": {"type":"choice","instructions":"Which team?","criteria":{"billing":"Charges","technical":"Bugs"}},
	    "angry": {"type":"boolean","instructions":"Is the customer angry?"}
	  },
	  "model": "z-ai/glm-5.3-flash",
	  "temperature": 0,
	  "mode": "parallel"
	}`
	var req Request
	if err := json.Unmarshal([]byte(src), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.Model != "z-ai/glm-5.3-flash" || req.Mode != ModeParallel {
		t.Errorf("got model %q mode %q", req.Model, req.Mode)
	}
	if got := req.Questions.Keys(); !equalStrings(got, []string{"department", "angry"}) {
		t.Errorf("question order = %v", got)
	}
	angry, _ := req.Questions.Get("angry")
	if angry.Type != TypeNoul {
		t.Errorf("boolean question decoded as %q", angry.Type)
	}
	text, err := req.State.Text()
	if err != nil {
		t.Fatalf("State.Text: %v", err)
	}
	if text != "{\n  \"subject\": \"Charged twice\"\n}" {
		t.Errorf("state text = %q", text)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

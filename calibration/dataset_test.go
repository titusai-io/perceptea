package calibration

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/titusai-io/perceptea/classifier"
)

// The three lines every parser test starts from: one of each question type,
// written exactly as a request body would declare them.
const (
	choiceLine = `{"id":"c1","name":"intent","state":"I want a refund","question":{"type":"choice","instructions":"Which intent is this?","criteria":{"refund":"wants money back","support":"wants help"}},"answer":"refund"}`
	scoreLine  = `{"id":"s1","name":"urgency","state":"the site is down","question":{"type":"score","instructions":"How urgent is this?","criteria":["low","medium","high"]},"answer":2}`
	noulLine   = `{"id":"n1","name":"pii","state":"my number is 07700 900123","question":{"type":"noul","instructions":"The message contains personal data."},"answer":true}`
)

// parseAll parses the given lines or fails the test.
func parseAll(t *testing.T, lines ...string) []Case {
	t.Helper()
	cases, err := ParseDataset(strings.NewReader(strings.Join(lines, "\n") + "\n"))
	if err != nil {
		t.Fatalf("ParseDataset: %v", err)
	}
	return cases
}

// parseError parses the given lines and returns the error, failing the test
// if there was none.
func parseError(t *testing.T, lines ...string) string {
	t.Helper()
	cases, err := ParseDataset(strings.NewReader(strings.Join(lines, "\n") + "\n"))
	if err == nil {
		t.Fatalf("ParseDataset accepted %d cases, want an error", len(cases))
	}
	return err.Error()
}

func TestParseDatasetReadsEachAnswerType(t *testing.T) {
	cases := parseAll(t, choiceLine, scoreLine, noulLine)

	if len(cases) != 3 {
		t.Fatalf("parsed %d cases, want 3", len(cases))
	}

	choice := cases[0]
	if choice.ID != "c1" || choice.Line != 1 || choice.Name != "intent" {
		t.Errorf("choice case = %+v, want id c1 on line 1 named intent", choice)
	}
	if choice.Question.Type != classifier.TypeChoice {
		t.Errorf("choice type = %q, want %q", choice.Question.Type, classifier.TypeChoice)
	}
	if got := choice.Question.Options.Keys(); len(got) != 2 || got[0] != "refund" || got[1] != "support" {
		t.Errorf("option order = %v, want [refund support]", got)
	}
	if choice.Choice != "refund" {
		t.Errorf("choice label = %q, want %q", choice.Choice, "refund")
	}

	score := cases[1]
	if score.Question.Type != classifier.TypeScore {
		t.Errorf("score type = %q, want %q", score.Question.Type, classifier.TypeScore)
	}
	if score.Level != 2 {
		t.Errorf("score label = %d, want 2", score.Level)
	}
	if score.Line != 2 {
		t.Errorf("score line = %d, want 2", score.Line)
	}

	noul := cases[2]
	if noul.Question.Type != classifier.TypeNoul {
		t.Errorf("noul type = %q, want %q", noul.Question.Type, classifier.TypeNoul)
	}
	if !noul.Noul {
		t.Error("noul label = false, want true")
	}

	// The state is kept as the caller's own bytes, which is what a request
	// body does, so the text put to the model is not a re-encoding.
	text, err := noul.State.Text()
	if err != nil {
		t.Fatalf("State.Text: %v", err)
	}
	if text != "my number is 07700 900123" {
		t.Errorf("state text = %q", text)
	}
}

func TestParseDatasetAcceptsTheBooleanSynonym(t *testing.T) {
	// A request body may write "boolean" for a noul question, so a dataset
	// lifted out of one may too.
	cases := parseAll(t, `{"state":"x","question":{"type":"boolean","instructions":"It holds."},"answer":false}`)

	if cases[0].Question.Type != classifier.TypeNoul {
		t.Errorf("type = %q, want %q", cases[0].Question.Type, classifier.TypeNoul)
	}
	if cases[0].Noul {
		t.Error("noul label = true, want false")
	}
}

func TestParseDatasetDefaultsTheIDAndTheQuestionName(t *testing.T) {
	cases := parseAll(t, `{"state":"x","question":{"type":"noul","instructions":"It holds."},"answer":true}`)

	if cases[0].ID != "line-1" {
		t.Errorf("ID = %q, want %q", cases[0].ID, "line-1")
	}
	if cases[0].Name != DefaultQuestionName {
		t.Errorf("Name = %q, want %q", cases[0].Name, DefaultQuestionName)
	}
}

func TestParseDatasetNamesTheLineThatWillNotParse(t *testing.T) {
	msg := parseError(t, noulLine, `{"state":`, noulLine)

	if !strings.Contains(msg, "line 2") {
		t.Errorf("error = %q, want it to name line 2", msg)
	}
}

func TestParseDatasetSkipsBlankLinesWithoutShiftingLineNumbers(t *testing.T) {
	cases := parseAll(t, noulLine, "", "   ", choiceLine)

	if len(cases) != 2 {
		t.Fatalf("parsed %d cases, want 2", len(cases))
	}
	// The second case is on line 4 of the file, and that is the number a
	// failure has to report for anyone to find it.
	if cases[1].Line != 4 {
		t.Errorf("second case line = %d, want 4", cases[1].Line)
	}
	if cases[1].ID != "c1" {
		t.Errorf("second case ID = %q, want c1", cases[1].ID)
	}
}

func TestParseDatasetRejectsALabelThatIsNotADeclaredOption(t *testing.T) {
	msg := parseError(t, `{"id":"c9","state":"x","question":{"type":"choice","criteria":{"refund":"a","support":"b"}},"answer":"upgrade"}`)

	for _, want := range []string{"line 1", `"c9"`, `"upgrade"`, `"refund"`, `"support"`} {
		if !strings.Contains(msg, want) {
			t.Errorf("error = %q, want it to mention %s", msg, want)
		}
	}
}

func TestParseDatasetRejectsALevelOutsideTheScale(t *testing.T) {
	msg := parseError(t, `{"id":"s9","state":"x","question":{"type":"score","criteria":["low","high"]},"answer":2}`)

	for _, want := range []string{"line 1", `"s9"`, "2", "0 to 1"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error = %q, want it to mention %s", msg, want)
		}
	}
}

func TestParseDatasetRejectsAnAnswerOfTheWrongShape(t *testing.T) {
	cases := []struct {
		name string
		line string
		want string
	}{
		{"choice answered with a number", `{"state":"x","question":{"type":"choice","criteria":{"a":"1"}},"answer":3}`, "option key"},
		{"score answered with a string", `{"state":"x","question":{"type":"score","criteria":["a","b"]},"answer":"b"}`, "level index"},
		{"noul answered with a string", `{"state":"x","question":{"type":"noul"},"answer":"yes"}`, "true or false"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if msg := parseError(t, c.line); !strings.Contains(msg, c.want) {
				t.Errorf("error = %q, want it to mention %q", msg, c.want)
			}
		})
	}
}

func TestParseDatasetRejectsAnIncompleteCase(t *testing.T) {
	cases := []struct {
		name string
		line string
		want string
	}{
		{"no state", `{"question":{"type":"noul","instructions":"It holds."},"answer":true}`, `missing "state"`},
		{"no answer", `{"state":"x","question":{"type":"noul","instructions":"It holds."}}`, `missing "answer"`},
		{"no question", `{"state":"x","answer":true}`, "question type is missing"},
		{"unaskable question", `{"state":"x","question":{"type":"choice"},"answer":"a"}`, "at least one option"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			msg := parseError(t, c.line)
			if !strings.Contains(msg, c.want) {
				t.Errorf("error = %q, want it to mention %q", msg, c.want)
			}
			if !strings.Contains(msg, "line 1") {
				t.Errorf("error = %q, want it to name line 1", msg)
			}
		})
	}
}

func TestParseDatasetRejectsAFileWithNoCases(t *testing.T) {
	msg := parseError(t, "", "   ")

	if !strings.Contains(msg, "no cases") {
		t.Errorf("error = %q, want it to say the dataset holds no cases", msg)
	}
}

func TestLoadDatasetNamesTheFileItCouldNotRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.jsonl")

	_, err := LoadDataset(path)
	if err == nil {
		t.Fatal("LoadDataset accepted a file that does not exist")
	}
	if !strings.Contains(err.Error(), path) {
		t.Errorf("error = %q, want it to name %q", err, path)
	}
}

func TestLoadDatasetNamesTheFileAndTheLineAtFault(t *testing.T) {
	path := filepath.Join(t.TempDir(), "broken.jsonl")
	if err := os.WriteFile(path, []byte(noulLine+"\nnot json\n"), 0o600); err != nil {
		t.Fatalf("writing the fixture: %v", err)
	}

	_, err := LoadDataset(path)
	if err == nil {
		t.Fatal("LoadDataset accepted a malformed file")
	}
	if !strings.Contains(err.Error(), path) || !strings.Contains(err.Error(), "line 2") {
		t.Errorf("error = %q, want it to name %q and line 2", err, path)
	}
}

func TestCaseRequestDeclaresTheQuestionUnderItsOwnName(t *testing.T) {
	c := parseAll(t, choiceLine)[0]

	req := c.Request("some-model", 0.25)

	if req.Model != "some-model" || req.Temperature != 0.25 {
		t.Errorf("request = %+v, want the run's model and temperature", req)
	}
	if req.Mode != classifier.ModeParallel {
		t.Errorf("mode = %q, want %q", req.Mode, classifier.ModeParallel)
	}
	if got := req.Questions.Keys(); len(got) != 1 || got[0] != "intent" {
		t.Errorf("question names = %v, want [intent]", got)
	}
	q, _ := req.Questions.Get("intent")
	if q.Type != classifier.TypeChoice {
		t.Errorf("question type = %q, want %q", q.Type, classifier.TypeChoice)
	}
	// Validate is what the evaluator will run; a request this package builds
	// must already pass it.
	if err := classifier.Validate(req.Questions); err != nil {
		t.Errorf("the built request does not validate: %v", err)
	}
}

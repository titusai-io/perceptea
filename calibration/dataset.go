package calibration

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/titusai-io/perceptea/classifier"
)

// DefaultQuestionName is the name a case's question is asked under when the
// dataset does not give one.
//
// The name is not decoration: a question that declares no instructions falls
// back to one built from its name, so it reaches the model. A dataset whose
// questions have instructions can leave it alone; one that relies on the
// fallback should set it.
const DefaultQuestionName = "answer"

// maxLineBytes bounds one dataset line. A state can be a whole document, so
// the scanner's default 64 KiB ceiling is too low; this is high enough that
// nothing realistic hits it and low enough that a corrupt file without
// newlines fails with a message instead of exhausting memory.
const maxLineBytes = 8 << 20

// Case is one labelled example: a state, the question to ask about it, and
// the answer that is correct.
//
// Which answer field carries the label follows the question's type, which is
// the convention [classifier.Example] already established for a worked
// example: a choice answers with an option key, a score with a level index, a
// noul with true or false.
type Case struct {
	// ID identifies the case in the report. It defaults to the dataset line
	// the case was read from.
	ID string
	// Line is the 1-based line of the dataset the case came from, so that a
	// failure can be traced back to the file rather than only to an ID the
	// dataset may not have set.
	Line int
	// Name is the name the question is declared under in the request.
	Name string
	// State is the material the question is asked about.
	State classifier.State
	// Question is the question, in exactly the shape the API takes, so the
	// dataset speaks the same language as a request body.
	Question classifier.Question

	// Choice is the correct option key, for a choice question.
	Choice string
	// Level is the correct level index, for a score question.
	Level int
	// Noul is whether the proposition holds, for a noul question.
	Noul bool
}

// caseWire is the JSON shape of one dataset line. The answer is polymorphic —
// a string, a whole number or a boolean, following the question's type — so it
// is decoded in a second pass once the type is known, exactly as a question's
// criteria and a worked example's answer are.
type caseWire struct {
	ID       string              `json:"id,omitempty"`
	Name     string              `json:"name,omitempty"`
	State    classifier.State    `json:"state"`
	Question classifier.Question `json:"question"`
	Answer   json.RawMessage     `json:"answer"`
}

// LoadDataset reads a JSON Lines dataset from path.
func LoadDataset(path string) ([]Case, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("calibration: opening the dataset: %w", err)
	}
	defer func() { _ = f.Close() }()

	cases, err := ParseDataset(f)
	if err != nil {
		return nil, fmt.Errorf("calibration: %s: %w", path, err)
	}
	return cases, nil
}

// ParseDataset reads one labelled case per line.
//
// A line that will not parse stops the read and is reported with its line
// number. Skipping it instead would be the same failure the coverage report
// exists to prevent, one file earlier: a dataset that quietly shrank to the
// cases that happened to be well-formed.
//
// A line that is empty or only whitespace is not a case and is passed over.
// That is the trailing newline every text file ends with, not a malformed
// record.
func ParseDataset(r io.Reader) ([]Case, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineBytes)

	var cases []Case
	line := 0
	for scanner.Scan() {
		line++
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}
		c, err := decodeCase(line, []byte(text))
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		cases = append(cases, c)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("line %d: %w", line+1, err)
	}
	if len(cases) == 0 {
		return nil, fmt.Errorf("the dataset holds no cases")
	}
	return cases, nil
}

// decodeCase reads one line into a [Case] and checks that the case can
// actually be run: that the question is askable, and that the label names an
// answer the question could produce. Both are checked here rather than at run
// time because here is the only place that still knows the line number.
func decodeCase(line int, data []byte) (Case, error) {
	var wire caseWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return Case{}, err
	}

	c := Case{ID: wire.ID, Line: line, Name: wire.Name, State: wire.State, Question: wire.Question}
	if c.ID == "" {
		c.ID = fmt.Sprintf("line-%d", line)
	}
	if strings.TrimSpace(c.Name) == "" {
		c.Name = DefaultQuestionName
	}
	if c.State.IsZero() {
		return Case{}, fmt.Errorf("case %q is missing %q", c.ID, "state")
	}

	questions := classifier.NewOrderedMap[classifier.Question]()
	questions.Set(c.Name, c.Question)
	if err := classifier.Validate(*questions); err != nil {
		return Case{}, fmt.Errorf("case %q: %w", c.ID, err)
	}

	if len(wire.Answer) == 0 {
		return Case{}, fmt.Errorf("case %q is missing %q", c.ID, "answer")
	}
	if err := c.decodeAnswer(wire.Answer); err != nil {
		return Case{}, fmt.Errorf("case %q: %w", c.ID, err)
	}
	return c, nil
}

// decodeAnswer reads the label in the terms its question type uses.
func (c *Case) decodeAnswer(raw json.RawMessage) error {
	switch c.Question.Type {
	case classifier.TypeChoice:
		if err := json.Unmarshal(raw, &c.Choice); err != nil {
			return fmt.Errorf("a choice question's answer is an option key, as a string: %w", err)
		}
		if _, ok := c.Question.Options.Get(c.Choice); !ok {
			return fmt.Errorf("answer %q is not one of the declared options %s",
				c.Choice, quoteList(c.Question.Options.Keys()))
		}
	case classifier.TypeScore:
		if err := json.Unmarshal(raw, &c.Level); err != nil {
			return fmt.Errorf("a score question's answer is a level index, as a whole number: %w", err)
		}
		if c.Level < 0 || c.Level >= len(c.Question.Levels) {
			return fmt.Errorf("answer %d is not a level index; the question declares %d levels, so the scale runs 0 to %d",
				c.Level, len(c.Question.Levels), len(c.Question.Levels)-1)
		}
	default:
		if err := json.Unmarshal(raw, &c.Noul); err != nil {
			return fmt.Errorf("a noul question's answer is true or false: %w", err)
		}
	}
	return nil
}

// quoteList renders keys as `"a", "b", "c"` for an error message.
func quoteList(keys []string) string {
	quoted := make([]string, 0, len(keys))
	for _, k := range keys {
		quoted = append(quoted, fmt.Sprintf("%q", k))
	}
	return strings.Join(quoted, ", ")
}

// Request builds the evaluation request for one case: the case's state, its
// one question under its own name, and the run's model and temperature.
func (c Case) Request(model string, temperature float64) classifier.Request {
	questions := classifier.NewOrderedMap[classifier.Question]()
	questions.Set(c.Name, c.Question)
	return classifier.Request{
		State:       c.State,
		Questions:   *questions,
		Model:       model,
		Temperature: temperature,
		Mode:        classifier.ModeParallel,
	}
}

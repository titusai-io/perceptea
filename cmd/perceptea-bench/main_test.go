package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/titusai-io/perceptea/calibration"
	"github.com/titusai-io/perceptea/classifier"
)

// No test here runs a benchmark: that needs a key and a provider. What is
// testable without either is everything that happens before the first call —
// the flags, the dataset, and the shape of the output — so that is what these
// cover.

// runErr runs the command with args and returns the error it reported,
// failing the test if it reported none.
func runErr(t *testing.T, args ...string) string {
	t.Helper()
	var stdout, stderr bytes.Buffer
	err := run(args, &stdout, &stderr)
	if err == nil {
		t.Fatalf("run(%v) succeeded; want an error. stdout:\n%s", args, stdout.String())
	}
	return err.Error()
}

func TestRunRequiresADataset(t *testing.T) {
	msg := runErr(t)

	if !strings.Contains(msg, "-dataset") {
		t.Errorf("error = %q, want it to name -dataset", msg)
	}
}

func TestRunRejectsAnImpossibleRepeatOrLimit(t *testing.T) {
	// A dataset that does not exist, deliberately: the flags are checked
	// before anything is opened, so a run that got past them fails with a
	// different message instead of reaching a provider from a unit test.
	absent := filepath.Join(t.TempDir(), "absent.jsonl")
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"repeat below one", []string{"-dataset", absent, "-repeat", "0"}, "-repeat"},
		{"negative limit", []string{"-dataset", absent, "-limit", "-3"}, "-limit"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if msg := runErr(t, c.args...); !strings.Contains(msg, c.want) {
				t.Errorf("error = %q, want it to name %s", msg, c.want)
			}
		})
	}
}

func TestRunNamesADatasetItCannotRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.jsonl")

	msg := runErr(t, "-dataset", path)

	if !strings.Contains(msg, path) {
		t.Errorf("error = %q, want it to name %q", msg, path)
	}
}

// TestExampleDatasetParses keeps the format the command documents and the
// format the loader accepts from drifting apart: the example beside main.go
// is the only thing most people will read before writing their own.
func TestExampleDatasetParses(t *testing.T) {
	cases, err := calibration.LoadDataset("example.jsonl")
	if err != nil {
		t.Fatalf("LoadDataset: %v", err)
	}

	if len(cases) != 7 {
		t.Fatalf("parsed %d cases, want 7", len(cases))
	}
	seen := map[classifier.QuestionType]int{}
	for _, c := range cases {
		seen[c.Question.Type]++
	}
	// All three question types, so the example shows how each is labelled.
	for _, want := range []classifier.QuestionType{classifier.TypeChoice, classifier.TypeScore, classifier.TypeNoul} {
		if seen[want] == 0 {
			t.Errorf("the example dataset has no %q case", want)
		}
	}
}

func TestWriteEmitsTextByDefaultAndJSONOnRequest(t *testing.T) {
	report := calibration.Report{
		Dataset:     "example.jsonl",
		Model:       "some-model",
		Coverage:    calibration.Coverage{Cases: 1, Repeat: 1, Attempted: 1, Evaluated: 1},
		Calibration: calibration.Calibrate([]calibration.Prediction{{P: 0.75, Holds: true}}),
	}

	var text bytes.Buffer
	if err := write(&text, report, false); err != nil {
		t.Fatalf("write text: %v", err)
	}
	if !strings.Contains(text.String(), "perceptea calibration benchmark") {
		t.Errorf("text output = %q", text.String())
	}

	var doc bytes.Buffer
	if err := write(&doc, report, true); err != nil {
		t.Fatalf("write json: %v", err)
	}
	var back calibration.Report
	if err := json.Unmarshal(doc.Bytes(), &back); err != nil {
		t.Fatalf("the -json output is not a JSON document: %v", err)
	}
	if back.Dataset != report.Dataset || back.Calibration.ECE != report.Calibration.ECE {
		t.Errorf("decoded report = %+v, want the same numbers as %+v", back, report)
	}
}

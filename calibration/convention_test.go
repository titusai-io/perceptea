package calibration

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/titusai-io/perceptea/classifier"
)

// A labelled answer in a dataset and a worked answer in a request are the same
// convention: a choice answers with its option key, a score with its level
// index, a noul with true or false. Two decoders implement it — this package's,
// which validates a file and reports a line, and the classifier's, which
// decodes a request. Their messages differ on purpose; what must never differ
// is which JSON shape each question type answers with.
//
// So the two are run over the same inputs and compared. Merging them would
// force one context's error messages onto the other; this pins the only part
// that has to agree.
//
// Two halves, because a decoder is half defined by what it refuses. Pinning
// only the answers both accept let the two disagree about which documents
// have an answer at all — see
// TestTheAnswerConventionRefusesTheSameDocuments.
func TestTheAnswerConventionMatchesTheClassifiers(t *testing.T) {
	for _, tc := range []struct {
		name     string
		question string
		answer   string
	}{
		{"choice answers with an option key", `{"type":"choice","criteria":{"billing":"money","tech":"bugs"}}`, `"tech"`},
		{"score answers with a level index", `{"type":"score","criteria":["Low","High","Critical"]}`, `2`},
		{"noul answers true", `{"type":"noul","instructions":"Is it urgent?"}`, `true`},
		{"noul answers false", `{"type":"noul","instructions":"Is it urgent?"}`, `false`},
		{"the boolean synonym answers the same way", `{"type":"boolean","instructions":"Is it urgent?"}`, `true`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The classifier's decoder, reached the only way it is exposed:
			// a question carrying one worked example.
			var viaClassifier classifier.Question
			doc := tc.question[:len(tc.question)-1] +
				`,"examples":[{"state":"some state","answer":` + tc.answer + `}]}`
			if err := json.Unmarshal([]byte(doc), &viaClassifier); err != nil {
				t.Fatalf("the classifier rejected %s: %v", doc, err)
			}
			if len(viaClassifier.Examples) != 1 {
				t.Fatalf("the classifier decoded %d examples, want 1", len(viaClassifier.Examples))
			}
			want := viaClassifier.Examples[0]

			// This package's decoder, over the same question and answer.
			line := `{"state":"some state","question":` + tc.question + `,"answer":` + tc.answer + `}`
			cases, err := ParseDataset(strings.NewReader(line))
			if err != nil {
				t.Fatalf("this package rejected %s: %v", line, err)
			}
			got := cases[0]

			if got.Choice != want.Choice {
				t.Errorf("choice: this package read %q, the classifier read %q", got.Choice, want.Choice)
			}
			if got.Level != want.Level {
				t.Errorf("level: this package read %d, the classifier read %d", got.Level, want.Level)
			}
			if got.Noul != want.Noul {
				t.Errorf("noul: this package read %v, the classifier read %v", got.Noul, want.Noul)
			}
		})
	}
}

// TestTheAnswerConventionRefusesTheSameDocuments is the other half: a label
// that was never given must be refused, by both decoders, for every question
// type.
//
// JSON null is the case that is easy to miss. json.Unmarshal reads it into an
// int and into a bool without complaint and leaves the Go zero value, so a
// score labelled null becomes level 0 and a noul becomes false — a label the
// dataset never gave. A choice escapes only by accident, because "" is not a
// declared option key, which is what makes the three types disagree about one
// document.
//
// It matters more here than in a request. A worked example teaches the model
// something wrong; a dataset label is the ground truth every number in the
// report is scored against, so a fabricated one makes the Brier score, the
// calibration error, the accuracy and the mean absolute error all wrong — a
// silent wrong answer from the one tool whose job is saying whether the
// answers can be trusted.
func TestTheAnswerConventionRefusesTheSameDocuments(t *testing.T) {
	questions := []struct{ name, question string }{
		{"choice", `{"type":"choice","criteria":{"billing":"money","tech":"bugs"}}`},
		{"score", `{"type":"score","criteria":["Low","High","Critical"]}`},
		{"noul", `{"type":"noul","instructions":"Is it urgent?"}`},
		{"the boolean synonym", `{"type":"boolean","instructions":"Is it urgent?"}`},
	}
	answers := []struct{ name, field string }{
		{"an explicit null answer", `,"answer":null`},
		{"no answer at all", ``},
	}

	for _, q := range questions {
		for _, a := range answers {
			t.Run(q.name+" with "+a.name, func(t *testing.T) {
				// The classifier's decoder, reached the only way it is
				// exposed: a question carrying one worked example.
				doc := q.question[:len(q.question)-1] +
					`,"examples":[{"state":"some state"` + a.field + `}]}`
				var viaClassifier classifier.Question
				if err := json.Unmarshal([]byte(doc), &viaClassifier); err == nil {
					t.Errorf("the classifier accepted %s", doc)
				}

				// This package's decoder, over the same question and answer.
				line := `{"id":"c1","state":"some state","question":` + q.question + a.field + "}\n"
				_, err := ParseDataset(strings.NewReader(line))
				if err == nil {
					t.Fatalf("this package accepted %s", line)
				}
				// The dataset's own error still says where in the file to
				// look, and which field is missing.
				for _, want := range []string{"line 1", `"answer"`} {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error = %q, want it to mention %q", err, want)
					}
				}
			})
		}
	}
}

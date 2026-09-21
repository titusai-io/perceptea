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

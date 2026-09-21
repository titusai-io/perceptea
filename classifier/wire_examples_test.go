package classifier

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

// The block a question declares has to arrive at the [Scorer], identically on
// every candidate of that question and on no candidate of any other. Until
// this was wired the examples validated and rendered and then went nowhere,
// and nothing else in the package noticed.
func TestExamplesReachTheScorerOnEveryCandidate(t *testing.T) {
	const doc = `{"state":"a ticket","questions":{"dept":{"type":"choice",
	  "instructions":"Which team?","criteria":{"billing":"money","tech":"bugs"},
	  "examples":[{"state":"charged twice","answer":"billing"}]},
	  "plain":{"type":"noul","instructions":"Is it urgent?"}}}`
	var req Request
	if err := json.Unmarshal([]byte(doc), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Calls are made from separate goroutines and their order is not
	// promised, so they are collected under a lock and keyed by the
	// statement rather than by arrival.
	var mu sync.Mutex
	byStatement := map[string]string{}
	ev := New(ScorerFunc(func(_ context.Context, r ScoreRequest) (ScoreResult, error) {
		mu.Lock()
		byStatement[r.Statement] = r.Examples
		mu.Unlock()
		return ScoreResult{Probability: 0.5}, nil
	}))
	if _, err := ev.Evaluate(context.Background(), req); err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if len(byStatement) != 3 {
		t.Fatalf("scored %d distinct statements, want 3", len(byStatement))
	}

	var blocks []string
	for statement, examples := range byStatement {
		isChoice := strings.Contains(statement, "The correct which team")
		switch {
		case isChoice && examples == "":
			t.Errorf("a choice candidate carried no examples: %q", statement)
		case isChoice:
			blocks = append(blocks, examples)
			if !strings.Contains(examples, "charged twice") || !strings.Contains(examples, "billing") {
				t.Errorf("the block does not carry the example: %q", examples)
			}
		case examples != "":
			t.Errorf("the noul declares no examples but was given %q", examples)
		}
	}
	if len(blocks) != 2 {
		t.Fatalf("%d choice candidates carried examples, want 2", len(blocks))
	}
	// The whole value of the block is being the same bytes every time: a
	// backend places it in the part of the prompt it expects to be identical
	// across the wave, and a per-candidate difference defeats that silently.
	if blocks[0] != blocks[1] {
		t.Errorf("the two candidates of one question were given different blocks:\n%q\n%q", blocks[0], blocks[1])
	}
}

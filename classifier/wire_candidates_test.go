package classifier

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

// The request every test below scores. It carries the four shapes that matter
// to a candidate list: a choice with several options, a score with several
// levels, a choice with exactly one option, and a noul. The two multi-option
// questions phrase their candidates differently on purpose — a pair of
// questions whose lists happened to match could not show a question being
// handed another's.
const candidateWiring = `{"state":"a ticket","questions":{
	"dept":{"type":"choice","instructions":"Which team?",
	  "criteria":{"billing":"money","tech":"bugs","sales":"plans"}},
	"urgency":{"type":"score","instructions":"How urgent?","criteria":["Low","Medium","High"]},
	"solo":{"type":"choice","instructions":"Is it billing?","criteria":{"billing":"money"}},
	"plain":{"type":"noul","instructions":"Is it urgent?"}}}`

// The candidates of the two multi-option questions, spelled the way the
// statement builders spell them, so a test can name the statements it expects
// in a list without re-deriving them from the thing under test.
var (
	wiredDept = []string{
		ChoiceStatement("dept", "Which team?", "billing", "money"),
		ChoiceStatement("dept", "Which team?", "tech", "bugs"),
		ChoiceStatement("dept", "Which team?", "sales", "plans"),
	}
	wiredUrgency = []string{
		ScoreStatement("urgency", "How urgent?", 0, "Low"),
		ScoreStatement("urgency", "How urgent?", 1, "Medium"),
		ScoreStatement("urgency", "How urgent?", 2, "High"),
	}
)

// candidatesByStatement evaluates candidateWiring and reports what candidate
// list each scored statement was handed.
//
// Calls are made from separate goroutines and their order is not promised, so
// they are collected under a lock and keyed by the statement rather than by
// arrival.
func candidatesByStatement(t *testing.T) map[string]string {
	t.Helper()
	var req Request
	if err := json.Unmarshal([]byte(candidateWiring), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	var mu sync.Mutex
	seen := map[string]string{}
	ev := New(ScorerFunc(func(_ context.Context, r ScoreRequest) (ScoreResult, error) {
		mu.Lock()
		seen[r.Statement] = r.Candidates
		mu.Unlock()
		return ScoreResult{Probability: 0.5}, nil
	}))
	if _, err := ev.Evaluate(context.Background(), req); err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	// 3 options, 3 levels, 1 option and 1 proposition.
	if len(seen) != 8 {
		t.Fatalf("scored %d distinct statements, want 8", len(seen))
	}
	return seen
}

// wantBlock is the list the block builder is specified to produce for these
// statements, written out here rather than taken from CandidatesBlock: this
// file is checking that a list reaches the scorer and which one, and the
// wording itself is pinned in TestCandidatesBlock.
func wantBlock(statements []string) string {
	return wantCandidatesHeader + "\n- " + strings.Join(statements, "\n- ")
}

// Every candidate of a question is handed its question's whole list, and the
// same bytes of it. That is the property the block is worth anything for: a
// per-candidate difference would cost the wave its shared prefix without
// changing a single answer.
func TestTheCandidateListReachesEveryCandidateOfItsQuestion(t *testing.T) {
	// Two candidates rendering the same statement would collapse into one
	// key above and make what follows true for the wrong reason.
	for _, group := range [][]string{wiredDept, wiredUrgency} {
		for i := range group {
			for j := i + 1; j < len(group); j++ {
				if group[i] == group[j] {
					t.Fatalf("candidates %d and %d render the same statement %q; "+
						"this fixture cannot see a list that lost one", i, j, group[i])
				}
			}
		}
	}

	seen := candidatesByStatement(t)
	for _, group := range [][]string{wiredDept, wiredUrgency} {
		want := wantBlock(group)
		for i, statement := range group {
			got, ok := seen[statement]
			if !ok {
				t.Fatalf("candidate %d was never scored: %q", i, statement)
			}
			if got != want {
				t.Errorf("candidate %d was handed a different list\n got: %q\nwant: %q", i, got, want)
			}
		}
	}
}

// A question with one candidate has no rivals to name, so it is scored with
// exactly the request it was scored with before any of this existed.
func TestASingleCandidateQuestionCarriesNoCandidateList(t *testing.T) {
	seen := candidatesByStatement(t)
	alone := map[string]string{
		"the one option of a one-option choice": ChoiceStatement("solo", "Is it billing?", "billing", "money"),
		"a noul":                                NoulStatement("plain", "Is it urgent?"),
	}
	for name, statement := range alone {
		got, ok := seen[statement]
		if !ok {
			t.Fatalf("%s was never scored: %q", name, statement)
		}
		if got != "" {
			t.Errorf("%s was handed a candidate list: %q", name, got)
		}
	}
}

// One request, two questions, two lists. A question's candidates see their
// own rivals and never another question's, which would be a list of answers
// to a question nobody asked.
func TestEachQuestionCarriesItsOwnCandidateListAndNotAnothers(t *testing.T) {
	if wantBlock(wiredDept) == wantBlock(wiredUrgency) {
		t.Fatal("the two questions render the same list; this fixture cannot see one handed the other's")
	}

	seen := candidatesByStatement(t)
	for _, tc := range []struct {
		name  string
		own   []string
		other []string
	}{
		{name: "dept", own: wiredDept, other: wiredUrgency},
		{name: "urgency", own: wiredUrgency, other: wiredDept},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for i, statement := range tc.own {
				got := seen[statement]
				for j, foreign := range tc.other {
					if strings.Contains(got, foreign) {
						t.Errorf("candidate %d was shown the other question's candidate %d: %q", i, j, got)
					}
				}
			}
		})
	}
}

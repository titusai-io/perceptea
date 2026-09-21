package calibration

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/titusai-io/perceptea/classifier"
)

// distribution builds an answer's probability map from alternating key and
// value pairs, in the order given.
func distribution(t *testing.T, pairs ...any) classifier.OrderedMap[float64] {
	t.Helper()
	if len(pairs)%2 != 0 {
		t.Fatalf("distribution needs key/value pairs, got %d arguments", len(pairs))
	}
	out := classifier.NewOrderedMap[float64]()
	for i := 0; i < len(pairs); i += 2 {
		key, ok := pairs[i].(string)
		if !ok {
			t.Fatalf("distribution key %d is %T, want a string", i, pairs[i])
		}
		value, ok := pairs[i+1].(float64)
		if !ok {
			t.Fatalf("distribution value for %q is %T, want a float64", key, pairs[i+1])
		}
		out.Set(key, value)
	}
	return *out
}

// mixedRun is the fixture every summary test below is derived from: three
// choice cases, two score cases, two noul cases and one that failed.
//
// The numbers are chosen so that no bin holds a single repeated value and no
// label is constant: a dataset where every case has the same answer, or where
// every prediction is the same number, produces a plausible-looking report
// whatever the code does.
//
// Three choice cases rather than two, for the same reason. Two cases of which
// one is right cannot tell "count the right answers" from "count the wrong
// ones" — both come to one — and a fixture that cannot tell those apart
// cannot check an accuracy. Two of three can.
func mixedRun(t *testing.T) Run {
	t.Helper()

	const (
		choiceQuestion = `{"type":"choice","instructions":"Which intent?","criteria":{"a":"A","b":"B","c":"C"}}`
		scoreQuestion  = `{"type":"score","instructions":"How urgent?","criteria":["l0","l1","l2"]}`
		noulQuestion   = `{"type":"noul","instructions":"It holds."}`
	)
	lines := []string{
		`{"id":"c1","name":"intent","state":"x","question":` + choiceQuestion + `,"answer":"a"}`,
		`{"id":"c2","name":"intent","state":"x","question":` + choiceQuestion + `,"answer":"b"}`,
		`{"id":"c3","name":"intent","state":"x","question":` + choiceQuestion + `,"answer":"c"}`,
		`{"id":"s1","name":"urgency","state":"x","question":` + scoreQuestion + `,"answer":2}`,
		`{"id":"s2","name":"urgency","state":"x","question":` + scoreQuestion + `,"answer":2}`,
		`{"id":"n1","name":"holds","state":"x","question":` + noulQuestion + `,"answer":true}`,
		`{"id":"n2","name":"holds","state":"x","question":` + noulQuestion + `,"answer":false}`,
		`{"id":"f1","name":"holds","state":"x","question":` + noulQuestion + `,"answer":true}`,
	}
	cases, err := ParseDataset(strings.NewReader(strings.Join(lines, "\n") + "\n"))
	if err != nil {
		t.Fatalf("building the fixture: %v", err)
	}

	answers := []classifier.Answer{
		{Type: classifier.TypeChoice, Choice: "a", Probabilities: distribution(t, "a", 0.7, "b", 0.2, "c", 0.1)},
		{Type: classifier.TypeChoice, Choice: "a", Probabilities: distribution(t, "a", 0.5, "b", 0.4, "c", 0.1)},
		{Type: classifier.TypeChoice, Choice: "c", Probabilities: distribution(t, "a", 0.1, "b", 0.2, "c", 0.7)},
		{Type: classifier.TypeScore, Score: 1.1, Probabilities: distribution(t, "0", 0.2, "1", 0.5, "2", 0.3)},
		{Type: classifier.TypeScore, Score: 1.7, Probabilities: distribution(t, "0", 0.1, "1", 0.1, "2", 0.8)},
		{Type: classifier.TypeNoul, Noul: 0.9},
		{Type: classifier.TypeNoul, Noul: 0.3},
		{},
	}

	outcomes := make([]Outcome, 0, len(cases))
	for i, c := range cases {
		o := Outcome{Case: c, Attempt: 1, Answer: answers[i]}
		if c.ID == "f1" {
			o.Answer = classifier.Answer{}
			o.Err = errors.New("upstream said no")
		}
		outcomes = append(outcomes, o)
	}

	return Run{
		Dataset:  "fixture.jsonl",
		Model:    "some-model",
		Cases:    len(cases),
		Repeat:   1,
		Outcomes: outcomes,
	}
}

// TestSummariseCoverage checks the part of the report that says how much of
// the dataset the rest of it was computed from.
func TestSummariseCoverage(t *testing.T) {
	got := Summarise(mixedRun(t)).Coverage

	want := Coverage{Cases: 8, Repeat: 1, Attempted: 8, Evaluated: 7, Failed: 1}
	if got.Cases != want.Cases || got.Repeat != want.Repeat || got.Attempted != want.Attempted ||
		got.Evaluated != want.Evaluated || got.Failed != want.Failed {
		t.Errorf("coverage = %+v, want %+v", got, want)
	}
	if len(got.Failures) != 1 {
		t.Fatalf("got %d failures, want 1", len(got.Failures))
	}
	// A count without the reason sends whoever reads it to the provider's
	// dashboard to guess, so the reason is part of the report.
	f := got.Failures[0]
	if f.ID != "f1" || f.Line != 8 || f.Attempt != 1 || f.Reason != "upstream said no" {
		t.Errorf("failure = %+v, want f1 on line 8, attempt 1, upstream said no", f)
	}
}

// TestSummariseCoverageUnderRepeats is the fixture mixedRun cannot be.
//
// Every case there runs once and every attempt is dispatched, so Cases,
// Repeat and Attempted all read as "the same eight things counted three
// ways" and Failure.Attempt is 1 for every failure. Four of the numbers
// Summarise reports are therefore indistinguishable from constants: the
// repeat could be hard-coded to 1, the attempts could be counted as the
// cases, the cases as the attempts, and a failure could claim to be attempt 1
// whichever pass it came from, and the suite would not notice.
//
// So: three cases, four repeats, and one case cut short — a run that was
// interrupted, which is a shape the command now reports rather than throws
// away. Twelve attempts were possible and ten happened, nine of them
// answered, and the one that failed was the third pass over the case on line
// two. No two of those numbers are equal, and none of them equals 1.
func TestSummariseCoverageUnderRepeats(t *testing.T) {
	cases := noulDataset(t, "clouds", "sun", "fog")
	answer := classifier.Answer{Type: classifier.TypeNoul, Noul: 0.6}

	var outcomes []Outcome
	add := func(index, attempt int, err error) {
		o := Outcome{Case: cases[index], Attempt: attempt, Answer: answer}
		if err != nil {
			o.Answer = classifier.Answer{}
			o.Err = err
		}
		outcomes = append(outcomes, o)
	}
	for attempt := 1; attempt <= 4; attempt++ {
		add(0, attempt, nil)
	}
	for attempt := 1; attempt <= 4; attempt++ {
		var err error
		if attempt == 3 {
			err = errors.New("upstream said no")
		}
		add(1, attempt, err)
	}
	// The third case was interrupted after two of its four passes.
	add(2, 1, nil)
	add(2, 2, nil)

	got := Summarise(Run{Cases: len(cases), Repeat: 4, Outcomes: outcomes}).Coverage

	want := Coverage{Cases: 3, Repeat: 4, Attempted: 10, Evaluated: 9, Failed: 1}
	if got.Cases != want.Cases || got.Repeat != want.Repeat || got.Attempted != want.Attempted ||
		got.Evaluated != want.Evaluated || got.Failed != want.Failed {
		t.Errorf("coverage = %+v, want %+v", got, want)
	}
	if len(got.Failures) != 1 {
		t.Fatalf("got %d failures, want 1", len(got.Failures))
	}
	// Which pass a failure came from is the whole reason a run is repeated:
	// a case that fails on one pass of four is a different finding from one
	// that fails on all of them.
	if f := got.Failures[0]; f.ID != "sun" || f.Line != 2 || f.Attempt != 3 {
		t.Errorf("failure = %+v, want sun on line 2, attempt 3", f)
	}
}

// The repeat a report carries is the one the run declared, and a run that
// declared none ran each case once.
func TestSummariseReportsOneRepeatForARunThatDeclaredNone(t *testing.T) {
	for _, declared := range []int{0, -1, 1} {
		got := Summarise(Run{Cases: 1, Repeat: declared}).Coverage.Repeat
		if got != 1 {
			t.Errorf("a run with Repeat %d reported %d, want 1", declared, got)
		}
	}
}

// TestSummariseCalibration works the whole reliability table out by hand from
// the seventeen probabilities the fixture's seven answered cases produce.
//
// Each case contributes one prediction per candidate, with Holds true for the
// candidate the label names:
//
//	c1 label a: (0.7 yes) (0.2 no)  (0.1 no)
//	c2 label b: (0.5 no)  (0.4 yes) (0.1 no)
//	c3 label c: (0.1 no)  (0.2 no)  (0.7 yes)
//	s1 label 2: (0.2 no)  (0.5 no)  (0.3 yes)
//	s2 label 2: (0.1 no)  (0.1 no)  (0.8 yes)
//	n1 label true:  (0.9 yes)
//	n2 label false: (0.3 no)
//
// Binned, that is:
//
//	bin 1 [0.1,0.2): 0.1 no x5                 n=5 mean 0.1 obs 0/5=0    gap 0.1
//	bin 2 [0.2,0.3): 0.2 no x3                 n=3 mean 0.2 obs 0/3=0    gap 0.2
//	bin 3 [0.3,0.4): 0.3 yes, 0.3 no           n=2 mean 0.3 obs 1/2=0.5  gap 0.2
//	bin 4 [0.4,0.5): 0.4 yes                   n=1 mean 0.4 obs 1/1=1    gap 0.6
//	bin 5 [0.5,0.6): 0.5 no, 0.5 no            n=2 mean 0.5 obs 0/2=0    gap 0.5
//	bin 7 [0.7,0.8): 0.7 yes, 0.7 yes          n=2 mean 0.7 obs 2/2=1    gap 0.3
//	bin 8 [0.8,0.9): 0.8 yes                   n=1 mean 0.8 obs 1/1=1    gap 0.2
//	bin 9 [0.9,1.0]: 0.9 yes                   n=1 mean 0.9 obs 1/1=1    gap 0.1
//
// Bins 0 and 6 are empty. The counts sum to 17, and
//
//	ECE = (1/17) * (5(0.1) + 3(0.2) + 2(0.2) + 1(0.6) + 2(0.5) + 2(0.3) + 1(0.2) + 1(0.1))
//	    = (1/17) * (0.5 + 0.6 + 0.4 + 0.6 + 1.0 + 0.6 + 0.2 + 0.1)
//	    = 4.0 / 17
//	    = 0.2352941176...
//
// which a report stores to six places as 0.235294.
func TestSummariseCalibration(t *testing.T) {
	got := Summarise(mixedRun(t)).Calibration

	if got.Predictions != 17 {
		t.Errorf("Predictions = %d, want 17", got.Predictions)
	}
	if got.ECE != 0.235294 {
		t.Errorf("ECE = %v, want 0.235294", got.ECE)
	}

	want := []Bin{
		{Low: 0.0, High: 0.1, Count: 0},
		{Low: 0.1, High: 0.2, Count: 5, MeanPredicted: 0.1, ObservedFrequency: 0},
		{Low: 0.2, High: 0.3, Count: 3, MeanPredicted: 0.2, ObservedFrequency: 0},
		{Low: 0.3, High: 0.4, Count: 2, MeanPredicted: 0.3, ObservedFrequency: 0.5},
		{Low: 0.4, High: 0.5, Count: 1, MeanPredicted: 0.4, ObservedFrequency: 1},
		{Low: 0.5, High: 0.6, Count: 2, MeanPredicted: 0.5, ObservedFrequency: 0},
		{Low: 0.6, High: 0.7, Count: 0},
		{Low: 0.7, High: 0.8, Count: 2, MeanPredicted: 0.7, ObservedFrequency: 1},
		{Low: 0.8, High: 0.9, Count: 1, MeanPredicted: 0.8, ObservedFrequency: 1},
		{Low: 0.9, High: 1.0, Count: 1, MeanPredicted: 0.9, ObservedFrequency: 1},
	}
	if len(got.Bins) != len(want) {
		t.Fatalf("got %d bins, want %d", len(got.Bins), len(want))
	}
	for i, w := range want {
		if got.Bins[i] != w {
			t.Errorf("bin %d = %+v, want %+v", i, got.Bins[i], w)
		}
	}
}

// TestSummariseAnswerMetrics derives each headline by hand from the same
// fixture.
//
// Choice, using the summed multi-class form:
//
//	c1 label a: (0.7-1)^2 + (0.2)^2 + (0.1)^2 = 0.09 + 0.04 + 0.01 = 0.14
//	c2 label b: (0.5)^2 + (0.4-1)^2 + (0.1)^2 = 0.25 + 0.36 + 0.01 = 0.62
//	c3 label c: (0.1)^2 + (0.2)^2 + (0.7-1)^2 = 0.01 + 0.04 + 0.09 = 0.14
//	mean Brier = (0.14 + 0.62 + 0.14)/3 = 0.9/3 = 0.3
//	c1 answered "a" and is right, c2 answered "a" and is wrong, c3 answered
//	"c" and is right, so accuracy = 2/3 = 0.666667 to six places.
//
// Score, mean absolute error against the labelled level:
//
//	s1 answered 1.1, label 2: |1.1 - 2| = 0.9
//	s2 answered 1.7, label 2: |1.7 - 2| = 0.3
//	MAE = (0.9 + 0.3)/2 = 0.6
//	s1 Brier = (0.2)^2 + (0.5)^2 + (0.3-1)^2 = 0.04 + 0.25 + 0.49 = 0.78
//	s2 Brier = (0.1)^2 + (0.1)^2 + (0.8-1)^2 = 0.01 + 0.01 + 0.04 = 0.06
//	mean Brier = (0.78 + 0.06)/2 = 0.42
//
// Noul, (p - y)^2:
//
//	n1 answered 0.9, label true:  (0.9 - 1)^2 = 0.01
//	n2 answered 0.3, label false: (0.3 - 0)^2 = 0.09
//	mean Brier = (0.01 + 0.09)/2 = 0.05
func TestSummariseAnswerMetrics(t *testing.T) {
	got := Summarise(mixedRun(t))

	if got.Choice == nil || got.Score == nil || got.Noul == nil {
		t.Fatalf("report = %+v, want all three sections", got)
	}
	wantChoice := ChoiceMetrics{Cases: 3, Correct: 2, Accuracy: 0.666667, Brier: 0.3}
	if *got.Choice != wantChoice {
		t.Errorf("choice = %+v, want %+v", *got.Choice, wantChoice)
	}
	wantScore := ScoreMetrics{Cases: 2, MAE: 0.6, Brier: 0.42}
	if *got.Score != wantScore {
		t.Errorf("score = %+v, want %+v", *got.Score, wantScore)
	}
	wantNoul := NoulMetrics{Cases: 2, Brier: 0.05}
	if *got.Noul != wantNoul {
		t.Errorf("noul = %+v, want %+v", *got.Noul, wantNoul)
	}
}

func TestSummariseCarriesWhatWasMeasured(t *testing.T) {
	got := Summarise(mixedRun(t))

	if got.Dataset != "fixture.jsonl" || got.Model != "some-model" {
		t.Errorf("report = dataset %q model %q, want fixture.jsonl and some-model", got.Dataset, got.Model)
	}
}

func TestSummariseOmitsATypeThatWasNotRun(t *testing.T) {
	run := mixedRun(t)
	// Keep only the noul cases.
	var kept []Outcome
	for _, o := range run.Outcomes {
		if o.Case.Question.Type == classifier.TypeNoul && o.Err == nil {
			kept = append(kept, o)
		}
	}
	run.Outcomes = kept
	run.Cases = len(kept)

	got := Summarise(run)

	// An absent section means "none were run", which is a different claim
	// from a section full of zeroes.
	if got.Choice != nil {
		t.Errorf("choice = %+v, want nil", *got.Choice)
	}
	if got.Score != nil {
		t.Errorf("score = %+v, want nil", *got.Score)
	}
	if got.Noul == nil || got.Noul.Cases != 2 {
		t.Errorf("noul = %+v, want 2 cases", got.Noul)
	}
}

func TestSummariseWhenEveryCaseFailed(t *testing.T) {
	run := mixedRun(t)
	for i := range run.Outcomes {
		run.Outcomes[i].Err = errors.New("upstream said no")
	}

	got := Summarise(run)

	if got.Coverage.Evaluated != 0 || got.Coverage.Failed != 8 {
		t.Errorf("coverage = %+v, want 0 evaluated and 8 failed", got.Coverage)
	}
	if got.Calibration.Predictions != 0 || got.Calibration.ECE != 0 {
		t.Errorf("calibration = %+v, want nothing measured", got.Calibration)
	}
	if got.Choice != nil || got.Score != nil || got.Noul != nil {
		t.Error("a run that measured nothing reported an answer section")
	}
}

func TestChoiceDistributionFollowsTheQuestionsOrder(t *testing.T) {
	cases, err := ParseDataset(strings.NewReader(
		`{"id":"c1","name":"intent","state":"x","question":{"type":"choice","criteria":{"a":"A","b":"B","c":"C"}},"answer":"b"}` + "\n"))
	if err != nil {
		t.Fatalf("building the fixture: %v", err)
	}
	// Deliberately in a different order from the question, and missing
	// nothing, so only the ordering can be wrong.
	answer := classifier.Answer{
		Type:          classifier.TypeChoice,
		Choice:        "a",
		Probabilities: distribution(t, "c", 0.1, "b", 0.2, "a", 0.7),
	}

	probs, correct := choiceDistribution(cases[0], answer)

	want := []float64{0.7, 0.2, 0.1}
	if len(probs) != len(want) {
		t.Fatalf("got %d probabilities, want %d", len(probs), len(want))
	}
	for i := range want {
		if probs[i] != want[i] {
			t.Fatalf("probabilities = %v, want %v in the question's own order", probs, want)
		}
	}
	if correct != 1 {
		t.Errorf("correct index = %d, want 1 (option b)", correct)
	}
}

func TestChoiceDistributionScoresAMissingCandidateAsZero(t *testing.T) {
	cases, err := ParseDataset(strings.NewReader(
		`{"id":"c1","name":"intent","state":"x","question":{"type":"choice","criteria":{"a":"A","b":"B","c":"C"}},"answer":"a"}` + "\n"))
	if err != nil {
		t.Fatalf("building the fixture: %v", err)
	}
	answer := classifier.Answer{
		Type:          classifier.TypeChoice,
		Choice:        "a",
		Probabilities: distribution(t, "a", 0.7, "c", 0.1),
	}

	probs, correct := choiceDistribution(cases[0], answer)

	// A reply missing a key must leave a hole rather than shift every later
	// candidate one place along.
	want := []float64{0.7, 0, 0.1}
	for i := range want {
		if probs[i] != want[i] {
			t.Fatalf("probabilities = %v, want %v", probs, want)
		}
	}
	if correct != 0 {
		t.Errorf("correct index = %d, want 0", correct)
	}
}

func TestReportTextShowsCoverageAndTheHeadlines(t *testing.T) {
	text := Summarise(mixedRun(t)).Text()

	want := []string{
		"fixture.jsonl",
		"some-model",
		"8 cases x 1 repeat = 8 attempts",
		"evaluated 7, failed 1",
		"line 8 (f1) attempt 1: upstream said no",
		"ECE 0.2353 over 17 predicted probabilities",
		"[0.4,0.5)",
		"[0.9,1.0]",
		"accuracy 0.667 (2/3)",
		"MAE 0.600 levels [headline]",
		"Brier 0.0500",
	}
	for _, w := range want {
		if !strings.Contains(text, w) {
			t.Errorf("report text does not contain %q\n---\n%s", w, text)
		}
	}
}

func TestReportJSONIsTheSameNumbers(t *testing.T) {
	report := Summarise(mixedRun(t))

	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("encoding the report: %v", err)
	}
	var back Report
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("decoding the report: %v", err)
	}

	if back.Calibration.ECE != report.Calibration.ECE {
		t.Errorf("ECE survived encoding as %v, want %v", back.Calibration.ECE, report.Calibration.ECE)
	}
	if back.Choice == nil || *back.Choice != *report.Choice {
		t.Errorf("choice metrics = %+v, want %+v", back.Choice, report.Choice)
	}
	if len(back.Calibration.Bins) != BinCount {
		t.Errorf("got %d bins, want %d: every bin is written, empty ones included",
			len(back.Calibration.Bins), BinCount)
	}

	// A report that changed on every run could not be diffed against a
	// stored one, so nothing in it may carry the clock.
	for _, forbidden := range []string{"time", "timestamp", "date", "duration"} {
		if strings.Contains(string(raw), `"`+forbidden+`"`) {
			t.Errorf("the report document carries a %q field", forbidden)
		}
	}
}

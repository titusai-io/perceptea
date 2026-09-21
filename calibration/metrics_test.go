package calibration

import (
	"math"
	"testing"
)

// tolerance is how far a computed statistic may sit from the value derived by
// hand. Every expected number in this file is worked out on paper from
// decimal arithmetic; the code does the same arithmetic in binary floating
// point, where 0.9 is not 0.9. A few ulps of drift is the difference; a wrong
// formula is many orders of magnitude more.
const tolerance = 1e-9

// closeTo fails unless got is within tolerance of want.
func closeTo(t *testing.T, what string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Errorf("%s = %v, want %v", what, got, want)
	}
}

func TestBinIndexPutsBoundariesInTheBinTheyOpen(t *testing.T) {
	// A probability on a boundary belongs to the bin it opens, and 1.0 has to
	// land somewhere rather than index off the end of a ten-element slice.
	cases := []struct {
		p    float64
		want int
	}{
		{0.0, 0},
		{0.05, 0},
		{0.0999, 0},
		{0.1, 1},
		{0.15, 1},
		{0.2, 2},
		{0.3, 3},
		{0.7, 7},
		{0.8999, 8},
		{0.9, 9},
		{0.999, 9},
		{1.0, 9},
		{-0.5, 0},
		{1.5, 9},
		{math.NaN(), 0},
	}
	for _, c := range cases {
		if got := binIndex(c.p); got != c.want {
			t.Errorf("binIndex(%v) = %d, want %d", c.p, got, c.want)
		}
	}
}

// TestCalibrateNoticesABadlyCalibratedRun is the fixture that matters most.
//
// A perfectly calibrated fixture cannot tell a working expected calibration
// error from one that returns zero, so this one is deliberately bad: ten
// predictions all claiming 0.9, of which one held.
//
//	bin 9 [0.9,1.0]: n = 10, mean p = 0.9, observed = 1/10 = 0.1
//	                 gap = |0.9 - 0.1| = 0.8
//	ECE = (10/10) * 0.8 = 0.8
//
// Every other bin is empty and contributes nothing.
func TestCalibrateNoticesABadlyCalibratedRun(t *testing.T) {
	preds := make([]Prediction, 0, 10)
	preds = append(preds, Prediction{P: 0.9, Holds: true})
	for range 9 {
		preds = append(preds, Prediction{P: 0.9, Holds: false})
	}

	got := Calibrate(preds)

	if got.Predictions != 10 {
		t.Errorf("Predictions = %d, want 10", got.Predictions)
	}
	closeTo(t, "ECE", got.ECE, 0.8)

	top := got.Bins[9]
	if top.Count != 10 {
		t.Errorf("bin 9 count = %d, want 10", top.Count)
	}
	closeTo(t, "bin 9 mean predicted", top.MeanPredicted, 0.9)
	closeTo(t, "bin 9 observed frequency", top.ObservedFrequency, 0.1)

	for i, b := range got.Bins[:9] {
		if b.Count != 0 {
			t.Errorf("bin %d count = %d, want 0", i, b.Count)
		}
	}
}

// TestCalibratePerfectlyCalibratedRunScoresZero is the control for the test
// above: the same code, a fixture whose predictions match reality, and an
// error of zero.
//
//	bin 2 [0.2,0.3): n = 10, mean p = 0.2, observed = 2/10 = 0.2, gap = 0
//	bin 8 [0.8,0.9): n = 10, mean p = 0.8, observed = 8/10 = 0.8, gap = 0
//	ECE = (10/20)*0 + (10/20)*0 = 0
//
// On its own this proves nothing — a function returning a constant zero would
// pass it — which is exactly why it is paired with the fixture above.
func TestCalibratePerfectlyCalibratedRunScoresZero(t *testing.T) {
	var preds []Prediction
	for i := range 10 {
		preds = append(preds, Prediction{P: 0.2, Holds: i < 2})
	}
	for i := range 10 {
		preds = append(preds, Prediction{P: 0.8, Holds: i < 8})
	}

	got := Calibrate(preds)

	closeTo(t, "ECE", got.ECE, 0)
	closeTo(t, "bin 2 mean predicted", got.Bins[2].MeanPredicted, 0.2)
	closeTo(t, "bin 2 observed frequency", got.Bins[2].ObservedFrequency, 0.2)
	closeTo(t, "bin 8 mean predicted", got.Bins[8].MeanPredicted, 0.8)
	closeTo(t, "bin 8 observed frequency", got.Bins[8].ObservedFrequency, 0.8)
}

// TestCalibrateBinTable works the whole table out on paper, with three
// unequally sized bins so that a weighting by anything other than bin count
// gives a different answer.
//
// The ten predictions, and the bin each falls in:
//
//	bin 0 [0.0,0.1): 0.00 no, 0.00 no, 0.05 no, 0.05 yes
//	bin 5 [0.5,0.6): 0.50 yes, 0.50 no
//	bin 9 [0.9,1.0]: 0.90 yes, 0.90 yes, 1.00 yes, 1.00 no
//
// Bin by bin:
//
//	bin 0: n = 4, mean p = (0 + 0 + 0.05 + 0.05)/4 = 0.1/4 = 0.025
//	              observed = 1/4 = 0.25,  gap = |0.025 - 0.25| = 0.225
//	bin 5: n = 2, mean p = (0.5 + 0.5)/2 = 0.5
//	              observed = 1/2 = 0.5,   gap = 0
//	bin 9: n = 4, mean p = (0.9 + 0.9 + 1.0 + 1.0)/4 = 3.8/4 = 0.95
//	              observed = 3/4 = 0.75,  gap = |0.95 - 0.75| = 0.2
//
//	ECE = (4/10)(0.225) + (2/10)(0) + (4/10)(0.2)
//	    = 0.09 + 0 + 0.08
//	    = 0.17
//
// The two outer bins err in opposite directions — bin 0 predicted less than
// happened, bin 9 more — so a sum that forgot the absolute value would score
// (4/10)(0.025 - 0.25) + 0 + (4/10)(0.95 - 0.75) = -0.09 + 0.08 = -0.01
// instead of 0.17. That is what makes this fixture able to see.
func TestCalibrateBinTable(t *testing.T) {
	preds := []Prediction{
		{P: 0.0, Holds: false},
		{P: 0.0, Holds: false},
		{P: 0.05, Holds: false},
		{P: 0.05, Holds: true},
		{P: 0.5, Holds: true},
		{P: 0.5, Holds: false},
		{P: 0.9, Holds: true},
		{P: 0.9, Holds: true},
		{P: 1.0, Holds: true},
		{P: 1.0, Holds: false},
	}

	got := Calibrate(preds)

	if got.Predictions != 10 {
		t.Errorf("Predictions = %d, want 10", got.Predictions)
	}
	closeTo(t, "ECE", got.ECE, 0.17)

	if len(got.Bins) != BinCount {
		t.Fatalf("len(Bins) = %d, want %d", len(got.Bins), BinCount)
	}

	want := map[int]Bin{
		0: {Low: 0.0, High: 0.1, Count: 4, MeanPredicted: 0.025, ObservedFrequency: 0.25},
		5: {Low: 0.5, High: 0.6, Count: 2, MeanPredicted: 0.5, ObservedFrequency: 0.5},
		9: {Low: 0.9, High: 1.0, Count: 4, MeanPredicted: 0.95, ObservedFrequency: 0.75},
	}
	for i, b := range got.Bins {
		w, populated := want[i]
		if !populated {
			if b.Count != 0 {
				t.Errorf("bin %d count = %d, want 0", i, b.Count)
			}
			continue
		}
		if b.Count != w.Count {
			t.Errorf("bin %d count = %d, want %d", i, b.Count, w.Count)
		}
		closeTo(t, "bin low", b.Low, w.Low)
		closeTo(t, "bin high", b.High, w.High)
		closeTo(t, "bin mean predicted", b.MeanPredicted, w.MeanPredicted)
		closeTo(t, "bin observed frequency", b.ObservedFrequency, w.ObservedFrequency)
	}
}

func TestCalibrateWithNoPredictions(t *testing.T) {
	got := Calibrate(nil)

	if got.Predictions != 0 {
		t.Errorf("Predictions = %d, want 0", got.Predictions)
	}
	if got.ECE != 0 {
		t.Errorf("ECE = %v, want 0", got.ECE)
	}
	if len(got.Bins) != BinCount {
		t.Fatalf("len(Bins) = %d, want %d", len(got.Bins), BinCount)
	}
	// The table keeps its shape when there is nothing in it, so two reports
	// still line up row for row when one of them measured nothing.
	closeTo(t, "bin 9 high", got.Bins[9].High, 1.0)
}

func TestBrierNoul(t *testing.T) {
	cases := []struct {
		p     float64
		holds bool
		want  float64
	}{
		// (0.8 - 1)^2 = (-0.2)^2 = 0.04
		{0.8, true, 0.04},
		// (0.3 - 1)^2 = (-0.7)^2 = 0.49
		{0.3, true, 0.49},
		// (0.25 - 0)^2 = 0.0625
		{0.25, false, 0.0625},
		// A shrug always costs (0.5 - y)^2 = 0.25, whichever way it went.
		{0.5, true, 0.25},
		{0.5, false, 0.25},
		// The extremes: certain and right, certain and wrong.
		{1, true, 0},
		{1, false, 1},
	}
	for _, c := range cases {
		closeTo(t, "BrierNoul", BrierNoul(c.p, c.holds), c.want)
	}
}

func TestBrierMulticlass(t *testing.T) {
	cases := []struct {
		name    string
		probs   []float64
		correct int
		want    float64
	}{
		{
			// (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09 + 0.04 + 0.01
			name: "confident and right", probs: []float64{0.7, 0.2, 0.1}, correct: 0, want: 0.14,
		},
		{
			// (0.7-0)^2 + (0.2-1)^2 + (0.1-0)^2 = 0.49 + 0.64 + 0.01
			name: "confident and wrong", probs: []float64{0.7, 0.2, 0.1}, correct: 1, want: 1.14,
		},
		{
			// (0.2-0)^2 + (0.5-0)^2 + (0.3-1)^2 = 0.04 + 0.25 + 0.49
			name: "spread over three levels", probs: []float64{0.2, 0.5, 0.3}, correct: 2, want: 0.78,
		},
		{
			// (1-0)^2 + (0-1)^2 = 2, the worst score the summed form allows.
			name: "certain and wrong", probs: []float64{1, 0}, correct: 1, want: 2,
		},
		{
			// (0.5-0.5)^2 twice = 0.5, the price of a two-way shrug.
			name: "two-way shrug", probs: []float64{0.5, 0.5}, correct: 0, want: 0.5,
		},
		{
			// No candidate is the right one, so every term is (p-0)^2:
			// 0.49 + 0.04 + 0.01 = 0.54.
			name: "label outside the distribution", probs: []float64{0.7, 0.2, 0.1}, correct: -1, want: 0.54,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			closeTo(t, "BrierMulticlass", BrierMulticlass(c.probs, c.correct), c.want)
		})
	}
}

func TestMean(t *testing.T) {
	closeTo(t, "mean", mean([]float64{0.04, 0.49, 0.0625}), 0.1975)
	closeTo(t, "mean of nothing", mean(nil), 0)
}

func TestRoundReport(t *testing.T) {
	// 3.4/14 = 0.242857142857..., which is what a report stores as 0.242857.
	if got := roundReport(3.4 / 14); got != 0.242857 {
		t.Errorf("roundReport(3.4/14) = %v, want 0.242857", got)
	}
	// The wobble a report exists to hide: both of these are 0.17.
	if got := roundReport(0.17000000000000004); got != 0.17 {
		t.Errorf("roundReport(0.17000000000000004) = %v, want 0.17", got)
	}
}

package classifier

import (
	"math"
	"testing"
)

// Every expected value in this file was worked out from the formulas these
// helpers implement — the softmax, the logit transform, the confidence
// expression, the roundings — and pasted in as a literal. None of it was
// recorded from this package's own output, which matters: a golden copied from
// the code under test agrees with that code whatever the code does, and so
// cannot tell a correct implementation from a broken one.

// equalFloats compares exactly. These expressions are evaluated in float64 in
// a pinned order, so the result is reproducible to the bit and a difference of
// even one ulp means something moved.
func equalFloats(got, want []float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name        string
		logits      []float64
		temperature float64
		want        []float64
	}{
		{
			name:        "temperature one",
			logits:      []float64{1, 2, 3},
			temperature: 1,
			want:        []float64{0.09003057317038046, 0.24472847105479764, 0.6652409557748218},
		},
		{
			name:        "temperature two flattens",
			logits:      []float64{1, 2, 3},
			temperature: 2,
			want:        []float64{0.1863237232258476, 0.3071958857184984, 0.506480391055654},
		},
		{
			name:        "at the temperature floor",
			logits:      []float64{1, 2, 3},
			temperature: 0.05,
			want:        []float64{4.248354246535078e-18, 2.0611536181902033e-9, 0.9999999979388463},
		},
		{
			name:        "below the floor is clamped to it",
			logits:      []float64{1, 2, 3},
			temperature: 0.01,
			want:        []float64{4.248354246535078e-18, 2.0611536181902033e-9, 0.9999999979388463},
		},
		{
			name:        "zero temperature is clamped to the floor",
			logits:      []float64{1, 2, 3},
			temperature: 0,
			want:        []float64{4.248354246535078e-18, 2.0611536181902033e-9, 0.9999999979388463},
		},
		{
			name:        "negative temperature is clamped to the floor",
			logits:      []float64{1, 2, 3},
			temperature: -7,
			want:        []float64{4.248354246535078e-18, 2.0611536181902033e-9, 0.9999999979388463},
		},
		{
			name:        "large logits do not overflow",
			logits:      []float64{1000, 1001, 1002},
			temperature: 1,
			want:        []float64{0.09003057317038046, 0.24472847105479764, 0.6652409557748218},
		},
		{
			name:        "large negative logits do not underflow to nothing",
			logits:      []float64{-1000, -1001},
			temperature: 1,
			want:        []float64{0.7310585786300049, 0.2689414213699951},
		},
		{
			name:        "equal logits are uniform",
			logits:      []float64{0, 0, 0, 0},
			temperature: 1,
			want:        []float64{0.25, 0.25, 0.25, 0.25},
		},
		{
			name:        "single logit takes everything",
			logits:      []float64{5},
			temperature: 1,
			want:        []float64{1},
		},
		{
			name:        "empty input gives an empty distribution",
			logits:      []float64{},
			temperature: 1,
			want:        []float64{},
		},
		{
			name:        "nil input gives an empty distribution",
			logits:      nil,
			temperature: 1,
			want:        []float64{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := softmax(tc.logits, tc.temperature)
			if !equalFloats(got, tc.want) {
				t.Fatalf("softmax(%v, %v) = %v, want %v", tc.logits, tc.temperature, got, tc.want)
			}
			if got == nil {
				t.Fatal("softmax returned a nil slice; it should always return a slice")
			}
		})
	}
}

// TestSoftmaxWouldOverflowWithoutMaxSubtraction shows what the max-subtraction
// is for: the same logits exponentiated directly are +Inf, which would make
// every probability NaN.
func TestSoftmaxWouldOverflowWithoutMaxSubtraction(t *testing.T) {
	if !math.IsInf(math.Exp(1002), 1) {
		t.Fatal("precondition failed: exp(1002) is expected to overflow to +Inf")
	}
	got := softmax([]float64{1000, 1001, 1002}, 1)
	sum := 0.0
	for _, p := range got {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			t.Fatalf("softmax produced a non-finite probability: %v", got)
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("softmax over large logits summed to %v, want 1", sum)
	}
}

func TestScoresToDistribution(t *testing.T) {
	tests := []struct {
		name   string
		scores []float64
		want   []float64
	}{
		{
			name:   "independent scores are normalised",
			scores: []float64{0.9, 0.2, 0.05},
			want:   []float64{0.9674681753889676, 0.02687411598302688, 0.005657708628005656},
		},
		{
			name:   "the extremes are clamped to the logit bounds",
			scores: []float64{0, 1},
			want:   []float64{0.000001002001999995993, 0.9999989979980001},
		},
		{
			name:   "scores already at the bounds are unchanged",
			scores: []float64{0.001, 0.999},
			want:   []float64{0.000001002001999995993, 0.9999989979980001},
		},
		{
			name:   "out-of-range scores clamp to the same bounds",
			scores: []float64{-3, 7},
			want:   []float64{0.000001002001999995993, 0.9999989979980001},
		},
		{
			name:   "equal scores stay equal",
			scores: []float64{0.5, 0.5},
			want:   []float64{0.5, 0.5},
		},
		{
			name:   "empty input gives an empty distribution",
			scores: []float64{},
			want:   []float64{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := scoresToDistribution(tc.scores)
			if !equalFloats(got, tc.want) {
				t.Fatalf("scoresToDistribution(%v) = %v, want %v", tc.scores, got, tc.want)
			}
			if len(got) == 0 {
				return
			}
			sum := 0.0
			for _, p := range got {
				sum += p
			}
			if math.Abs(sum-1) > 1e-12 {
				t.Fatalf("distribution %v summed to %v, want 1", got, sum)
			}
		})
	}
}

// TestScoresToDistributionClampBounds pins the clamped logits themselves, so a
// change to either bound is caught at the source.
func TestScoresToDistributionClampBounds(t *testing.T) {
	const wantLow = -6.906754778648554 // Math.log(0.001 / 0.999)
	const wantHigh = 6.906754778648554 // Math.log(0.999 / 0.001)
	if got := math.Log(logitClampLow / (1 - logitClampLow)); got != wantLow {
		t.Fatalf("logit of the low clamp = %v, want %v", got, wantLow)
	}
	if got := math.Log(logitClampHigh / (1 - logitClampHigh)); got != wantHigh {
		t.Fatalf("logit of the high clamp = %v, want %v", got, wantHigh)
	}
	// Anything at or below the low bound has to land on the same probability.
	below := scoresToDistribution([]float64{-1, 0.5})
	atBound := scoresToDistribution([]float64{0.001, 0.5})
	if !equalFloats(below, atBound) {
		t.Fatalf("scores below the clamp gave %v, at the clamp %v; they should agree", below, atBound)
	}
}

func TestRound3(t *testing.T) {
	tests := []struct {
		in, want float64
	}{
		{0.7123, 0.712},
		{0.0005, 0.001},
		{0.9995, 1},
		{1.0005, 1.001},
		{0.12349, 0.123},
		{0.12351, 0.124},
		{0, 0},
		{1, 1},
		{0.0004999, 0},
	}
	for _, tc := range tests {
		if got := round3(tc.in); got != tc.want {
			t.Errorf("round3(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestRound2(t *testing.T) {
	tests := []struct {
		in, want float64
	}{
		{2.675, 2.68},
		{2.665, 2.67},
		{1.005, 1},
		{0.125, 0.13},
		{1.235, 1.24},
		{1.8139534883720931, 1.81},
		{0, 0},
		{3, 3},
	}
	for _, tc := range tests {
		if got := round2(tc.in); got != tc.want {
			t.Errorf("round2(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestConfidence(t *testing.T) {
	tests := []struct {
		name  string
		probs []float64
		want  float64
	}{
		{name: "no candidates", probs: []float64{}, want: 0.25},
		{name: "nil candidates", probs: nil, want: 0.25},
		{name: "single candidate clamps to one", probs: []float64{1}, want: 1},
		{name: "flat over four", probs: []float64{0.25, 0.25, 0.25, 0.25}, want: 0.375},
		{name: "two-way coin flip", probs: []float64{0.5, 0.5}, want: 0.5},
		{name: "clear winner", probs: []float64{0.6, 0.3, 0.1}, want: 0.7},
		{name: "decisive winner clamps to one", probs: []float64{0.05, 0.9, 0.05}, want: 1},
		{name: "near tie", probs: []float64{0.46153846153846156, 0.46153846153846156, 0.07692307692307696}, want: 0.481},
		{name: "order does not matter", probs: []float64{0.1, 0.6, 0.3}, want: 0.7},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			input := make([]float64, len(tc.probs))
			copy(input, tc.probs)
			got := confidence(input)
			if got != tc.want {
				t.Fatalf("confidence(%v) = %v, want %v", tc.probs, got, tc.want)
			}
			if !equalFloats(input, tc.probs) {
				t.Fatalf("confidence reordered its input: %v, was %v", input, tc.probs)
			}
		})
	}
}

func TestClamp01(t *testing.T) {
	tests := []struct{ in, want float64 }{
		{-0.5, 0}, {0, 0}, {0.5, 0.5}, {1, 1}, {1.25, 1},
	}
	for _, tc := range tests {
		if got := clamp01(tc.in); got != tc.want {
			t.Errorf("clamp01(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

// TestSanitiseProbability pins the boundary rule on its own. clamp01 alone
// would pass a NaN straight through: math.Max(0, NaN) is NaN and so is
// math.Min(1, NaN).
func TestSanitiseProbability(t *testing.T) {
	for _, tc := range []struct{ in, want float64 }{
		{0, 0}, {0.5, 0.5}, {1, 1},
		{-0.0001, 0}, {-3, 0}, {1.0001, 1}, {12, 1},
		{math.Inf(1), 1}, {math.Inf(-1), 0},
	} {
		if got := sanitiseProbability(tc.in); got != tc.want {
			t.Errorf("sanitiseProbability(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
	if got := sanitiseProbability(math.NaN()); got != 0.5 {
		t.Errorf("sanitiseProbability(NaN) = %v, want 0.5", got)
	}
	if got := clamp01(math.NaN()); !math.IsNaN(got) {
		t.Errorf("clamp01(NaN) = %v; if clamp01 handles NaN itself, sanitiseProbability's own branch is untested", got)
	}
}

func TestArgmax(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   int
	}{
		{name: "first is largest", values: []float64{0.5, 0.3, 0.2}, want: 0},
		{name: "last is largest", values: []float64{0.2, 0.3, 0.5}, want: 2},
		{name: "middle is largest", values: []float64{0.2, 0.5, 0.3}, want: 1},
		{name: "an exact tie goes to the first", values: []float64{0.4, 0.4, 0.2}, want: 0},
		{name: "a tie at the back goes to the first of them", values: []float64{0.1, 0.45, 0.45}, want: 1},
		{name: "all equal picks the first", values: []float64{0.25, 0.25, 0.25, 0.25}, want: 0},
		{name: "single candidate", values: []float64{0.3}, want: 0},
		{name: "empty", values: nil, want: 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := argmax(tc.values); got != tc.want {
				t.Fatalf("argmax(%v) = %d, want %d", tc.values, got, tc.want)
			}
		})
	}
}

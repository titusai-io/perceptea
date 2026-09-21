package classifier

import (
	"math"
	"sort"
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

// TestScoresToDistribution pins the transform at [DefaultSoftmaxTemperature].
//
// Every want below is the literal this table carried before the temperature
// became configurable, unchanged: dividing a logit by 1 is the identity, so
// the default has to reproduce the old distributions to the bit, and this
// table is where that is checked. A default that had quietly become anything
// else would move every one of them.
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
			got := scoresToDistribution(tc.scores, DefaultSoftmaxTemperature)
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
	below := scoresToDistribution([]float64{-1, 0.5}, DefaultSoftmaxTemperature)
	atBound := scoresToDistribution([]float64{0.001, 0.5}, DefaultSoftmaxTemperature)
	if !equalFloats(below, atBound) {
		t.Fatalf("scores below the clamp gave %v, at the clamp %v; they should agree", below, atBound)
	}
}

func TestRound6(t *testing.T) {
	tests := []struct {
		in, want float64
	}{
		{0.7123, 0.7123},
		{0.0000005, 0.000001},
		{0.0000004999, 0},
		{1.0000005, 1.000001},
		{0.1234564, 0.123456},
		{0.1234566, 0.123457},
		{0, 0},
		{1, 1},
		// The three that used to round to a certainty and no longer do.
		// 0.9995 and 0.0005 were 1 and 0.001 at three places; the third is
		// the most extreme probability this package can actually reach.
		{0.9995, 0.9995},
		{0.0005, 0.0005},
		{0.9999989979980001, 0.999999},
	}
	for _, tc := range tests {
		if got := round6(tc.in); got != tc.want {
			t.Errorf("round6(%v) = %v, want %v", tc.in, got, tc.want)
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
		{name: "near tie", probs: []float64{0.46153846153846156, 0.46153846153846156, 0.07692307692307696}, want: 0.480769},
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

// TestSoftmaxTemperatureCannotChangeTheOrder is the property that makes the
// temperature a safe knob: dividing every logit of one question by the same
// positive number is monotone, so it can flatten or sharpen a distribution but
// never reorder it.
//
// The scores deliberately differ from one another. Equal scores would give a
// uniform distribution at every temperature, which would satisfy "the order
// never changes" without the temperature having done anything at all — so the
// second half of the test insists the distributions really are different, and
// the control below shows what a fixture that could not see this looks like.
func TestSoftmaxTemperatureCannotChangeTheOrder(t *testing.T) {
	scores := []float64{0.9, 0.6, 0.2, 0.75, 0.05}
	temperatures := []float64{0.2, 0.5, DefaultSoftmaxTemperature, 2, 5.04, 20}

	// The ranking at the default, worked out from the scores: the logit is
	// increasing in the score, and the softmax is increasing in the logit.
	wantOrder := []int{0, 3, 1, 2, 4}

	var previous []float64
	for _, temperature := range temperatures {
		got := scoresToDistribution(scores, temperature)
		if order := ranking(got); !equalInts(order, wantOrder) {
			t.Fatalf("at temperature %v the candidates rank %v, want %v", temperature, order, wantOrder)
		}
		if argmax(got) != 0 {
			t.Fatalf("at temperature %v the argmax is %d, want 0", temperature, argmax(got))
		}
		sum := 0.0
		for _, p := range got {
			sum += p
		}
		if math.Abs(sum-1) > 1e-12 {
			t.Fatalf("at temperature %v the distribution summed to %v, want 1", temperature, sum)
		}
		if previous != nil && equalFloats(got, previous) {
			t.Fatalf("temperature %v produced the same distribution as the one before it: %v",
				temperature, got)
		}
		previous = got
	}
}

// TestSoftmaxTemperatureIsInvisibleToEqualScores is the control for the test
// above. Candidates that scored the same are uniform at every temperature, so
// a fixture built out of them would pass whatever the temperature did — or did
// not — reach the softmax.
func TestSoftmaxTemperatureIsInvisibleToEqualScores(t *testing.T) {
	equal := []float64{0.4, 0.4, 0.4}
	first := scoresToDistribution(equal, DefaultSoftmaxTemperature)
	for _, temperature := range []float64{0.2, 2, 5.04, 20} {
		if got := scoresToDistribution(equal, temperature); !equalFloats(got, first) {
			t.Fatalf("equal scores gave %v at temperature %v and %v at the default; "+
				"they are expected to be identical, which is why no temperature test may use them",
				got, temperature, first)
		}
	}
}

// ranking lists the candidate indices from most to least probable, ties broken
// by the earlier index, which is [argmax]'s rule.
func ranking(probs []float64) []int {
	order := make([]int, len(probs))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return probs[order[a]] > probs[order[b]] })
	return order
}

func equalInts(got, want []int) bool {
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

// mostExtremeReachable is the largest probability [scoresToDistribution] can
// return at [DefaultSoftmaxTemperature]: two candidates, one pinned at each
// clamp, so their logits are 2·ln(0.999/0.001) apart and the winner takes
// 999²/(999²+1) of the mass.
const mostExtremeReachable = 0.9999989979980001

// TestSixPlacesIsTheSmallestPrecisionThatKeepsAProbabilityBelowOne is why
// [round6] is round6 and not round3, round4 or round5. Each of the coarser
// roundings reports the most extreme probability this package can compute as
// an exact certainty; six is the first that does not.
func TestSixPlacesIsTheSmallestPrecisionThatKeepsAProbabilityBelowOne(t *testing.T) {
	// The constant is the arithmetic, not a recording. The closed form is
	// 999²/(999²+1); the softmax reaches it through exp and log and lands one
	// ulp away, which is the only slack allowed here.
	if want := 998001.0 / 998002.0; math.Abs(mostExtremeReachable-want) > 2e-16 {
		t.Fatalf("the extreme constant is %v, want 999²/(999²+1) = %v", mostExtremeReachable, want)
	}
	// And it is what the transform actually reaches, from scores outside the
	// clamp so nothing narrower is being measured.
	extreme := scoresToDistribution([]float64{1, 0}, DefaultSoftmaxTemperature)
	if extreme[0] != mostExtremeReachable {
		t.Fatalf("the most extreme reachable probability is %v, want %v", extreme[0], mostExtremeReachable)
	}

	for places := 1; places <= 5; places++ {
		scale := math.Pow(10, float64(places))
		if got := math.Round(mostExtremeReachable*scale) / scale; got != 1 {
			t.Errorf("rounding %v to %d places gives %v; six places is supposed to be the first "+
				"that does not report it as a certainty", mostExtremeReachable, places, got)
		}
	}
	if got := round6(mostExtremeReachable); got != 0.999999 {
		t.Fatalf("round6(%v) = %v, want 0.999999", mostExtremeReachable, got)
	}
	if round6(extreme[1]) != 0.000001 {
		t.Fatalf("round6(%v) = %v, want 0.000001", extreme[1], round6(extreme[1]))
	}
}

// TestNoReportedProbabilityIsACertainty sweeps what a question can actually
// produce and insists none of it rounds to an exact 0 or an exact 1. The
// estimator clamps its inputs to (0.001, 0.999), so it can never mean either,
// and a caller who wants to re-normalise or re-temper a published distribution
// needs log(p) to exist.
//
// Two candidates and three are swept in full, at the saturated extremes and
// across a grid in between. Four and up have a bottom end six places cannot
// hold; that case has a test of its own below, rather than being left out of
// this one quietly.
func TestNoReportedProbabilityIsACertainty(t *testing.T) {
	grid := []float64{-5, 0, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 0.9999, 1, 12}

	check := func(scores []float64) {
		t.Helper()
		probs := scoresToDistribution(scores, DefaultSoftmaxTemperature)
		for i, p := range probs {
			reported := round6(p)
			if reported == 0 || reported == 1 {
				t.Fatalf("scores %v report candidate %d as %v, which is a certainty the "+
					"estimator cannot claim (unrounded %v)", scores, i, reported, p)
			}
		}
		if c := confidence(probs); c == 0 {
			t.Fatalf("scores %v give confidence 0", scores)
		}
	}

	for _, a := range grid {
		for _, b := range grid {
			check([]float64{a, b})
			for _, c := range grid {
				check([]float64{a, b, c})
			}
		}
	}
}

// TestTheReportedBottomEndIsARoundingFloor pins the one place the property
// above gives out, so that it is a known limit rather than something a caller
// discovers.
//
// From four candidates upward, a question whose model scored three or more of
// them at the top clamp and another at the bottom drives that other below half
// of the last reported place, and six decimals report it as 0. The value
// itself is not zero — the softmax over finite logits never is — so this is
// the rounding's floor and not the estimator claiming an impossibility.
func TestTheReportedBottomEndIsARoundingFloor(t *testing.T) {
	// Three candidates: still reported, and only just — 5.01e-7 against a
	// floor of 5e-7.
	three := scoresToDistribution([]float64{0, 1, 1}, DefaultSoftmaxTemperature)
	if got := round6(three[0]); got != 0.000001 {
		t.Fatalf("with three candidates the smallest reachable probability reports as %v, want 0.000001", got)
	}

	// Four: the same pattern falls under the floor.
	four := scoresToDistribution([]float64{0, 1, 1, 1}, DefaultSoftmaxTemperature)
	if four[0] <= 0 {
		t.Fatalf("the smallest probability is %v; the softmax is expected to stay strictly positive", four[0])
	}
	if four[0] >= 5e-7 {
		t.Fatalf("the smallest probability is %v, which six places can still report; "+
			"this test is supposed to be standing on a value below the rounding floor", four[0])
	}
	if got := round6(four[0]); got != 0 {
		t.Fatalf("round6(%v) = %v; the documented limit says four saturated candidates report 0", four[0], got)
	}

	// The one that is not a limit: the same four candidates with a single
	// winner — the ordinary confident answer — stay well inside the range.
	ordinary := scoresToDistribution([]float64{1, 0, 0, 0}, DefaultSoftmaxTemperature)
	for i, p := range ordinary {
		if r := round6(p); r == 0 || r == 1 {
			t.Fatalf("an ordinary confident four-way answer reports candidate %d as %v", i, r)
		}
	}
}

// TestConfidenceReachesOneOnlyByItsOwnClamp separates the two ways a reported
// confidence can be exactly 1. The expression overshoots 1 outright for a
// decisive winner and the clamp brings it back, which is a value this package
// computed; what must not happen is a confidence below 1 being rounded up into
// one.
func TestConfidenceReachesOneOnlyByItsOwnClamp(t *testing.T) {
	// Decisive: the unclamped expression is 1.125, so the 1 is the clamp's.
	decisive := []float64{0.05, 0.9, 0.05}
	if unclamped := 0.5*0.9 + 0.5*((0.9-0.05)+0.5); unclamped <= 1 {
		t.Fatalf("the expression over %v is %v; this fixture is meant to overshoot 1", decisive, unclamped)
	}
	if got := confidence(decisive); got != 1 {
		t.Fatalf("confidence(%v) = %v, want 1", decisive, got)
	}

	// Just short: at three places this rounded up to 1 and claimed a
	// certainty; at six it reports what it is.
	//
	//	0.5·0.8331 + 0.5·((0.8331 − 0.1669) + 0.5) = 0.99965
	almost := []float64{0.8331, 0.1669}
	if got := confidence(almost); got != 0.99965 {
		t.Fatalf("confidence(%v) = %v, want 0.99965", almost, got)
	}
	if got := math.Round(0.99965*1000) / 1000; got != 1 {
		t.Fatalf("at three places the same confidence is %v; this fixture is meant to be one "+
			"the old precision reported as a certainty", got)
	}
}

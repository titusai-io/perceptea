package calibration

import "math"

// BinCount is the number of equal-width bins the reliability table is built
// from: ten, each 0.1 of probability wide.
const BinCount = 10

// Bin is one row of the reliability table: every prediction whose probability
// fell in [Low, High), and what actually happened to them.
//
// The table is the part a human reads. A single expected calibration error can
// hide a model that is wildly overconfident at the top of the range and
// compensating for it at the bottom; the rows cannot.
type Bin struct {
	// Low and High are the bin's bounds. The interval is half-open, [Low,
	// High), except for the last bin, which includes 1.0.
	Low  float64 `json:"low"`
	High float64 `json:"high"`
	// Count is how many predictions landed in the bin.
	Count int `json:"count"`
	// MeanPredicted is the mean probability of those predictions: what the
	// service claimed would happen.
	MeanPredicted float64 `json:"mean_predicted"`
	// ObservedFrequency is the fraction of them that actually held: what did
	// happen. A well-calibrated bin has the two within noise of each other.
	ObservedFrequency float64 `json:"observed_frequency"`
}

// Calibration is the reliability table and the one number computed from it.
type Calibration struct {
	// Predictions is how many (probability, outcome) pairs went into the
	// table. It is not the number of cases: a choice with four options
	// contributes four.
	Predictions int `json:"predictions"`
	// ECE is the expected calibration error: the bin-count-weighted mean gap
	// between MeanPredicted and ObservedFrequency. Zero is perfect; the worst
	// attainable value is 1, which is what a run that predicted 1.0 for
	// events that never happened would score.
	ECE float64 `json:"ece"`
	// Bins always has [BinCount] entries, empty ones included, so that two
	// reports line up row for row when they are diffed.
	Bins []Bin `json:"bins"`
}

// binIndex reports which bin p belongs to.
//
// The bins are half-open so that a probability on a boundary belongs to the
// bin it opens — 0.1 is the first member of [0.1,0.2), not the last of
// [0.0,0.1) — and the top bin is closed at 1.0 so that a certain prediction
// has somewhere to go instead of indexing off the end. A probability outside
// [0,1] is clamped rather than rejected: the classifier already guarantees
// the range, and a benchmark that panicked on a stray value would be a worse
// instrument than one that recorded it in the nearest bin.
func binIndex(p float64) int {
	if math.IsNaN(p) || p <= 0 {
		return 0
	}
	if p >= 1 {
		return BinCount - 1
	}
	return int(p * BinCount)
}

// Calibrate groups preds into [BinCount] equal-width bins and computes the
// expected calibration error over them.
//
// The error is sum over bins of (count/total) * |mean predicted - observed
// frequency|: each bin contributes its disagreement, weighted by how much of
// the run it accounts for. An empty bin contributes nothing and is still
// reported, so the table always has the same shape.
func Calibrate(preds []Prediction) Calibration {
	bins := make([]Bin, BinCount)
	sums := make([]float64, BinCount)
	held := make([]int, BinCount)
	for i := range bins {
		bins[i].Low = float64(i) / BinCount
		bins[i].High = float64(i+1) / BinCount
	}

	for _, pred := range preds {
		i := binIndex(pred.P)
		bins[i].Count++
		sums[i] += pred.P
		if pred.Holds {
			held[i]++
		}
	}

	ece := 0.0
	for i := range bins {
		if bins[i].Count == 0 {
			continue
		}
		n := float64(bins[i].Count)
		bins[i].MeanPredicted = sums[i] / n
		bins[i].ObservedFrequency = float64(held[i]) / n
		gap := math.Abs(bins[i].MeanPredicted - bins[i].ObservedFrequency)
		ece += n / float64(len(preds)) * gap
	}

	return Calibration{Predictions: len(preds), ECE: ece, Bins: bins}
}

// round rounds every reported statistic in c, for a report that will be
// diffed against another. See [reportPlaces].
func (c Calibration) round() Calibration {
	out := c
	out.ECE = roundReport(c.ECE)
	out.Bins = make([]Bin, len(c.Bins))
	for i, b := range c.Bins {
		b.MeanPredicted = roundReport(b.MeanPredicted)
		b.ObservedFrequency = roundReport(b.ObservedFrequency)
		out.Bins[i] = b
	}
	return out
}

// BrierNoul is the Brier score of one noul answer: (p - y)^2, where y is 1
// when the proposition held and 0 when it did not.
//
// It runs from 0 for a confident correct answer to 1 for a confident wrong
// one, and a shrug at 0.5 always costs 0.25. Lower is better.
func BrierNoul(p float64, holds bool) float64 {
	y := 0.0
	if holds {
		y = 1
	}
	d := p - y
	return d * d
}

// BrierMulticlass is the Brier score of one distribution over candidates: the
// sum over candidates of (p_k - y_k)^2, where y_k is 1 for the candidate the
// label names and 0 for every other.
//
// This is the summed form, so it runs from 0 to 2 rather than 0 to 1 — a
// confidently wrong two-way answer scores (1-0)^2 + (0-1)^2 = 2. The summed
// form is used because it is the one that reduces to [BrierNoul] doubled for
// a two-candidate question, which keeps the two numbers comparable once you
// know the factor; halving it here would make the relationship a surprise
// instead of a stated convention.
//
// A correct index outside the distribution contributes no y term, which
// scores the distribution as though nothing was supposed to happen. The
// dataset loader rejects a label that names no declared candidate, so that
// case is a guard rather than a path.
func BrierMulticlass(probs []float64, correct int) float64 {
	sum := 0.0
	for k, p := range probs {
		y := 0.0
		if k == correct {
			y = 1
		}
		d := p - y
		sum += d * d
	}
	return sum
}

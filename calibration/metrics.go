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
	// events that never happened would score. The bound holds for any input,
	// a probability outside [0,1] included; see [sanitise].
	ECE float64 `json:"ece"`
	// Bins always has [BinCount] entries, empty ones included, so that two
	// reports line up row for row when they are diffed.
	Bins []Bin `json:"bins"`
}

// sanitise maps a number that is not a probability onto the nearest one that
// is: anything below zero, and a NaN, to 0; anything above one to 1.
//
// A stray value is clamped rather than rejected because the classifier
// already guarantees the range, and a benchmark that panicked on one would be
// a worse instrument than one that recorded it at the edge. [Calibrate] is
// exported, though, so "already guaranteed" is a statement about one caller
// and not about every caller — and an unclamped value does not merely land in
// the wrong bin, it is *averaged*: a prediction of 5 gave a bin whose mean
// predicted probability was 5 and an expected calibration error of 3.5,
// against a documented maximum of 1. A NaN was worse than wrong; it made the
// whole report NaN, and a NaN cannot be written as JSON at all, so -json
// failed with an encoding error instead of a number.
//
// The bin and the value are both taken from here so that the two cannot
// disagree about what a stray value meant.
func sanitise(p float64) float64 {
	if math.IsNaN(p) || p < 0 {
		return 0
	}
	if p > 1 {
		return 1
	}
	return p
}

// binLow reports the lower bound of bin i. It is the one place a bin boundary
// is computed, so that the table's rows and [binIndex]'s arithmetic cannot
// describe different bins.
func binLow(i int) float64 { return float64(i) / BinCount }

// binIndex reports which bin p belongs to.
//
// The bins are half-open so that a probability on a boundary belongs to the
// bin it opens — 0.1 is the first member of [0.1,0.2), not the last of
// [0.0,0.1) — and the top bin is closed at 1.0 so that a certain prediction
// has somewhere to go instead of indexing off the end.
//
// p*BinCount and [binLow] are two different roundings of the same boundary
// and they do not always agree: for exactly one float64 in [0,1] — the one
// just below 0.9 — the multiplication rounds up to 9.0 and puts the value in
// a bin whose reported Low is above it. It changes no arithmetic, because
// the bin's mean is computed from its members either way, but it makes one
// row of a table people read to check a claim say something untrue about its
// own contents. So the index is walked back to the bin that actually
// contains p.
func binIndex(p float64) int {
	p = sanitise(p)
	if p >= 1 {
		return BinCount - 1
	}
	i := int(p * BinCount)
	if i >= BinCount {
		i = BinCount - 1
	}
	if i > 0 && p < binLow(i) {
		i--
	}
	return i
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
		bins[i].Low = binLow(i)
		bins[i].High = binLow(i + 1)
	}

	for _, pred := range preds {
		i := binIndex(pred.P)
		bins[i].Count++
		// The sanitised value, not the one that arrived: a bin must not
		// report a mean its own bounds exclude. See [sanitise].
		sums[i] += sanitise(pred.P)
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

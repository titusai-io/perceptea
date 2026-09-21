package classifier

import (
	"math"
	"sort"
)

// The numerics here are the whole of the package's arithmetic: a softmax, a
// logit transform, two roundings and a confidence expression. Everything is
// done in float64, and each expression is written in the order it is meant to
// be evaluated, so that the same inputs always give the same bits.
//
// One property is worth stating because the rest of the file leans on it:
// [math.Round] rounds a half away from zero, which is only distinguishable
// from rounding a half upwards when the value is negative. Every value rounded
// in this package — a probability, an expected value over non-negative
// indices, a confidence — is non-negative, so the distinction never arises.
// That premise is what [sanitiseProbability] exists to guarantee: it is the
// only door a probability enters this package through.

// minTemperature is the floor applied to a temperature before dividing by it,
// so that a zero or negative temperature cannot produce a division by zero or
// a sign flip.
const minTemperature = 0.05

// softmax normalises logits into a distribution summing to one.
//
// The temperature is floored at [minTemperature]. The maximum is subtracted
// before exponentiating, which leaves the result unchanged mathematically and
// keeps large-magnitude logits from overflowing to +Inf. An empty input gives
// an empty output rather than a panic.
//
// The final division needs no guard against a zero sum: subtracting the
// maximum makes one term exp(0) == 1 exactly, so a non-empty input always sums
// to at least 1, and an empty one has already returned.
//
// scoresToDistribution is the only caller and always passes a temperature of 1;
// the parameter is kept so the transform stays the general one rather than
// having that single value baked into it.
func softmax(logits []float64, temperature float64) []float64 {
	out := make([]float64, len(logits))
	if len(logits) == 0 {
		return out
	}

	t := math.Max(minTemperature, temperature)

	maxScaled := math.Inf(-1)
	for i, x := range logits {
		out[i] = x / t
		if out[i] > maxScaled {
			maxScaled = out[i]
		}
	}

	sum := 0.0
	for i, scaled := range out {
		out[i] = math.Exp(scaled - maxScaled)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// logitClampLow and logitClampHigh bound a probability away from 0 and 1 so its
// logit stays finite.
const (
	logitClampLow  = 0.001
	logitClampHigh = 0.999
)

// scoresToDistribution turns independently estimated probabilities into one
// distribution: each score is clamped into (0,1), mapped to its logit, and the
// logits are softmaxed at temperature 1.
//
// The scores come from separate calls and do not sum to anything in
// particular; this is what makes them comparable.
func scoresToDistribution(scores []float64) []float64 {
	logits := make([]float64, len(scores))
	for i, p := range scores {
		clamped := math.Min(logitClampHigh, math.Max(logitClampLow, p))
		logits[i] = math.Log(clamped / (1 - clamped))
	}
	return softmax(logits, 1)
}

// round3 rounds to three decimal places, as every reported probability is.
func round3(x float64) float64 { return math.Round(x*1000) / 1000 }

// round2 rounds to two decimal places, as a score is.
func round2(x float64) float64 { return math.Round(x*100) / 100 }

// clamp01 bounds x to [0,1].
func clamp01(x float64) float64 { return math.Min(1, math.Max(0, x)) }

// unusableProbability is what a probability that cannot be believed becomes:
// maximally uninformative, which is what an unreadable answer is worth
// everywhere else in the service too.
const unusableProbability = 0.5

// sanitiseProbability brings a [Scorer]'s answer into [0,1].
//
// [Scorer] is an interface, so a probability arriving here was produced by an
// implementation this package does not control, and its range cannot be
// assumed. The clamp belongs at this one boundary because that is what lets
// everything downstream treat a probability as a probability.
//
// A NaN is not clamped but replaced outright by [unusableProbability]: left
// alone it spreads through the softmax into every probability and confidence
// of the question it belongs to, and json.Marshal refuses to encode one, so a
// single unreadable candidate would fail the whole response instead of
// blunting one score.
func sanitiseProbability(p float64) float64 {
	if math.IsNaN(p) {
		return unusableProbability
	}
	return clamp01(p)
}

// confidence scores how decisive a distribution is: half the winning
// probability plus half of its margin over the runner-up, offset so that a
// two-way coin flip lands at 0.5. The result is rounded to three places and
// clamped to [0,1] — a near-certain winner overshoots 1 before clamping.
//
// A missing top or runner-up counts as 0, so an empty distribution scores 0.25
// and a single candidate scores 1. The input is not modified.
func confidence(probs []float64) float64 {
	sorted := make([]float64, len(probs))
	copy(sorted, probs)
	sort.Sort(sort.Reverse(sort.Float64Slice(sorted)))

	var top1, top2 float64
	if len(sorted) > 0 {
		top1 = sorted[0]
	}
	if len(sorted) > 1 {
		top2 = sorted[1]
	}
	gap := top1 - top2
	// No FMA barrier is needed here: multiplying by 0.5 only changes the
	// exponent, so both halves are exact and a fused multiply-add cannot
	// round differently from a separate multiply and add.
	return clamp01(round3(0.5*top1 + 0.5*(gap+0.5)))
}

// argmax reports the index of the largest value, the earliest one on an exact
// tie: the comparison is a strict ">", so a later candidate has to beat the
// incumbent outright to displace it. That decides which option key a choice
// question returns when two are equally likely, and the rule is the earliest
// declared candidate rather than the last one scanned because the same request
// has to give the same answer every time.
func argmax(values []float64) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}

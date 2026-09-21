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
// The temperature here is the **softmax** temperature: how sharply this
// package normalises one question's candidate scores against each other. It
// has nothing to do with the sampling temperature a request or
// PERCEPTEA_TEMPERATURE sends to the provider, which decides how the model
// draws its tokens and never reaches this file. One is how the model answers;
// this one is how the answers are turned into a distribution.
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
// A single temperature divides every logit by the same positive number, which
// is monotone: it cannot reorder the logits and so cannot move an argmax. That
// is the property that makes [WithSoftmaxTemperature] a safe knob — it changes
// how confident an answer reads, never which answer it is.
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
// logits are softmaxed at the given softmax temperature — [DefaultSoftmaxTemperature]
// unless the operator configured another, and never the sampling temperature
// the provider was called with.
//
// The scores come from separate calls and do not sum to anything in
// particular; this is what makes them comparable.
func scoresToDistribution(scores []float64, temperature float64) []float64 {
	logits := make([]float64, len(scores))
	for i, p := range scores {
		clamped := math.Min(logitClampHigh, math.Max(logitClampLow, p))
		logits[i] = math.Log(clamped / (1 - clamped))
	}
	return softmax(logits, temperature)
}

// round6 rounds to six decimal places, as every reported probability and
// confidence is.
//
// Six, and not three, because a rounding must never turn a number this package
// can produce into one it cannot. The scores entering [scoresToDistribution]
// are clamped to (0.001, 0.999), so the widest logit gap two candidates can
// open is 2·ln(0.999/0.001) = 13.8135…, and the most extreme probability
// reachable — one candidate at the top clamp, one at the bottom, at
// [DefaultSoftmaxTemperature] — is 999²/(999²+1) = 0.9999989979980001. Three
// decimals report that as exactly 1 and five still do; six report 0.999999.
// A reported 1 would be a certainty the estimator is built never to claim, and
// it is also a dead end for a caller: nobody can re-normalise or re-temper a
// published distribution through log(0).
//
// The bottom end cannot be pinned the same way, and is not papered over. The
// smallest probability a question of n candidates can reach at temperature 1
// is 1/(1+(n-1)·999²): 1.002e-6 for two candidates, 3.34e-7 for four,
// 3.94e-9 for the 255 a choice may declare. Six decimals report the first two
// as 0.000001 and everything from four saturated candidates upward as 0. That
// zero is the rounding's, not the value's — the value is small, never zero —
// and it is accepted rather than chased: reaching it needs three or more
// candidates the model scored at 0.999 while scoring another at 0.001, which
// is a model contradicting itself, and no fixed number of decimals survives
// the case anyway. A configured softmax temperature below 1 sharpens the
// distribution without bound and moves both ends: under about 0.95 even the
// top end rounds to 1 again. Temperatures at or above 1 — the default, and
// every fitted value measured so far — keep the guarantee.
func round6(x float64) float64 { return math.Round(x*1e6) / 1e6 }

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
// two-way coin flip lands at 0.5. The result is rounded to six places by
// [round6], for the reason given there, and clamped to [0,1].
//
// A confidence of exactly 1 is still reachable, and is not the rounding's
// doing: the expression overshoots 1 outright for a decisive winner — the most
// extreme two-candidate distribution puts it at 1.2499984969969999 — and the
// clamp brings it back. That 1 is a value this package computed, so reporting
// it claims nothing it cannot support.
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
	return clamp01(round6(0.5*top1 + 0.5*(gap+0.5)))
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

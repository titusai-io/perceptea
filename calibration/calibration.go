// Package calibration measures whether a Perceptea answer's probabilities
// mean what they say.
//
// The package exists because every other accuracy claim this project might
// make — a different scorer, a different prompt, a different model — is
// unfalsifiable without it. It reads a file of labelled cases, evaluates each
// one down the ordinary classifier path, and reports four things: how well the
// probabilities are calibrated (an expected calibration error, with the
// reliability table it was computed from), how sharp they are (Brier), how
// often the answer is simply right (accuracy for a choice, mean absolute error
// for a score), and how much of the dataset actually ran.
//
// That last one is not a footnote. A benchmark that silently evaluated 40 of
// 100 cases and reported a lovely calibration error is worse than no
// benchmark, so a failed case is counted and named rather than dropped, and
// coverage is printed next to every number.
//
// Nothing here reaches the network. The run drives an [Evaluator]; what sits
// behind that is the caller's business, which is also what makes every part of
// this package testable without a key.
package calibration

import "math"

// Prediction is one probability the service produced, paired with whether the
// thing it was a probability of turned out to hold.
//
// Reducing every answer to these is what lets one calibration error cover a
// mixed dataset. A noul answer contributes its single probability; a choice
// contributes one per declared option; a score contributes one per declared
// level. In each case Holds is true for exactly the candidate the label names,
// so a confident wrong answer contributes both a high probability that did not
// hold and a low one that did.
type Prediction struct {
	// P is the predicted probability, as the service reported it.
	P float64
	// Holds is whether the predicted event actually occurred.
	Holds bool
}

// reportPlaces is how many decimal places a reported statistic keeps.
//
// The numbers are rounded on the way into a [Report] rather than on the way
// out, because the JSON document exists to be diffed against a stored run and
// a diff between 0.17000000000000004 and 0.16999999999999998 is a diff that
// means nothing. Six places is far past the precision any of these statistics
// actually carries.
const reportPlaces = 6

// roundReport rounds x to [reportPlaces] decimal places.
func roundReport(x float64) float64 {
	scale := math.Pow(10, reportPlaces)
	return math.Round(x*scale) / scale
}

// mean returns the arithmetic mean of xs, or 0 for an empty slice. A zero for
// "nothing to average" is safe here only because every caller reports the
// count alongside the mean, so an empty set is never mistaken for a perfect
// score.
func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sum := 0.0
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

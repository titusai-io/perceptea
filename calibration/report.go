package calibration

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/titusai-io/perceptea/classifier"
)

// Report is everything one run measured. It is the text a human reads and the
// document a script diffs against a stored run, which is why it carries no
// timestamp: a field that changes on every run makes every diff non-empty and
// so makes the diff useless.
type Report struct {
	// Dataset and Model say what was measured and with what. Neither is a
	// number, and both are the first thing anyone comparing two reports needs
	// to check.
	Dataset string `json:"dataset,omitempty"`
	Model   string `json:"model,omitempty"`

	Coverage    Coverage    `json:"coverage"`
	Calibration Calibration `json:"calibration"`

	// Choice, Score and Noul are present only when the dataset held cases of
	// that type, so an absent section means "none were run" rather than "all
	// of them scored zero".
	Choice *ChoiceMetrics `json:"choice,omitempty"`
	Score  *ScoreMetrics  `json:"score,omitempty"`
	Noul   *NoulMetrics   `json:"noul,omitempty"`
}

// Coverage says how much of the dataset the numbers above it were computed
// from.
type Coverage struct {
	// Cases is how many labelled cases the run was given, and Repeat how many
	// times each was evaluated, so Attempted is the product.
	Cases     int `json:"cases"`
	Repeat    int `json:"repeat"`
	Attempted int `json:"attempted"`
	// Evaluated is how many attempts produced an answer, and Failed how many
	// did not. Every other number in the report is computed from Evaluated
	// attempts alone.
	Evaluated int `json:"evaluated"`
	Failed    int `json:"failed"`
	// Failures names each one. A count without the reasons sends whoever
	// reads it back to the provider's dashboard to guess.
	Failures []Failure `json:"failures,omitempty"`
}

// Failure is one attempt that produced no answer.
type Failure struct {
	ID      string `json:"id"`
	Line    int    `json:"line"`
	Attempt int    `json:"attempt"`
	Reason  string `json:"reason"`
}

// ChoiceMetrics summarises the choice cases.
type ChoiceMetrics struct {
	// Cases is how many choice attempts were scored.
	Cases int `json:"cases"`
	// Correct is how many of them named the labelled option, and Accuracy is
	// that as a fraction.
	Correct  int     `json:"correct"`
	Accuracy float64 `json:"accuracy"`
	// Brier is the mean [BrierMulticlass] over the returned distributions,
	// so it runs 0 to 2 and lower is better.
	Brier float64 `json:"brier"`
}

// ScoreMetrics summarises the score cases.
//
// MAE is the headline. A score answer is an expected value over level
// indices, so the honest question about it is how far off the labelled level
// it landed, in levels — a number with a unit somebody can argue with. Brier
// over the level distribution is reported next to it because the distribution
// is what the calibration error is computed from, and a mean absolute error
// says nothing about whether the spread around the answer was believable.
type ScoreMetrics struct {
	// Cases is how many score attempts were scored.
	Cases int `json:"cases"`
	// MAE is the mean absolute error of the reported score against the
	// labelled level index, in levels.
	MAE float64 `json:"mae"`
	// Brier is the mean [BrierMulticlass] over the level distributions, so it
	// runs 0 to 2 and lower is better.
	Brier float64 `json:"brier"`
}

// NoulMetrics summarises the noul cases.
type NoulMetrics struct {
	// Cases is how many noul attempts were scored.
	Cases int `json:"cases"`
	// Brier is the mean (p - y)^2, so it runs 0 to 1 and lower is better. A
	// run that answered 0.5 to everything scores exactly 0.25, which is the
	// number to beat before any other claim is worth making.
	Brier float64 `json:"brier"`
}

// Summarise turns a run's outcomes into the report.
//
// Only attempts that produced an answer contribute to a metric; the rest
// contribute to the coverage section, which is reported whether or not
// anything failed.
func Summarise(run Run) Report {
	rep := Report{
		Dataset: run.Dataset,
		Model:   run.Model,
		Coverage: Coverage{
			Cases:     run.Cases,
			Repeat:    max(run.Repeat, 1),
			Attempted: len(run.Outcomes),
		},
	}

	var (
		preds        []Prediction
		choiceBriers []float64
		scoreBriers  []float64
		scoreErrors  []float64
		noulBriers   []float64
		choiceCases  int
		choiceRight  int
	)

	for _, o := range run.Outcomes {
		if o.Err != nil {
			rep.Coverage.Failed++
			rep.Coverage.Failures = append(rep.Coverage.Failures, Failure{
				ID:      o.Case.ID,
				Line:    o.Case.Line,
				Attempt: o.Attempt,
				Reason:  o.Err.Error(),
			})
			continue
		}
		rep.Coverage.Evaluated++

		switch o.Case.Question.Type {
		case classifier.TypeChoice:
			probs, correct := choiceDistribution(o.Case, o.Answer)
			preds = append(preds, predictions(probs, correct)...)
			choiceBriers = append(choiceBriers, BrierMulticlass(probs, correct))
			choiceCases++
			if o.Answer.Choice == o.Case.Choice {
				choiceRight++
			}

		case classifier.TypeScore:
			probs := levelDistribution(o.Case, o.Answer)
			preds = append(preds, predictions(probs, o.Case.Level)...)
			scoreBriers = append(scoreBriers, BrierMulticlass(probs, o.Case.Level))
			scoreErrors = append(scoreErrors, math.Abs(o.Answer.Score-float64(o.Case.Level)))

		default:
			preds = append(preds, Prediction{P: o.Answer.Noul, Holds: o.Case.Noul})
			noulBriers = append(noulBriers, BrierNoul(o.Answer.Noul, o.Case.Noul))
		}
	}

	rep.Calibration = Calibrate(preds).round()

	if choiceCases > 0 {
		rep.Choice = &ChoiceMetrics{
			Cases:    choiceCases,
			Correct:  choiceRight,
			Accuracy: roundReport(float64(choiceRight) / float64(choiceCases)),
			Brier:    roundReport(mean(choiceBriers)),
		}
	}
	if len(scoreErrors) > 0 {
		rep.Score = &ScoreMetrics{
			Cases: len(scoreErrors),
			MAE:   roundReport(mean(scoreErrors)),
			Brier: roundReport(mean(scoreBriers)),
		}
	}
	if len(noulBriers) > 0 {
		rep.Noul = &NoulMetrics{
			Cases: len(noulBriers),
			Brier: roundReport(mean(noulBriers)),
		}
	}
	return rep
}

// choiceDistribution reads the answer's probabilities in the order the
// question declared its options, and reports which of them the label names.
//
// The question's order is used rather than the answer's so that a provider
// reply which somehow lost or reordered a key still produces a distribution
// the same length as the question, with a missing candidate scored as zero
// rather than shifting every index along by one.
func choiceDistribution(c Case, a classifier.Answer) (probs []float64, correct int) {
	keys := c.Question.Options.Keys()
	probs = make([]float64, len(keys))
	correct = -1
	for i, key := range keys {
		p, _ := a.Probabilities.Get(key)
		probs[i] = p
		if key == c.Choice {
			correct = i
		}
	}
	return probs, correct
}

// levelDistribution reads the answer's probabilities by level index, which is
// how a score answer keys them.
func levelDistribution(c Case, a classifier.Answer) []float64 {
	probs := make([]float64, len(c.Question.Levels))
	for i := range probs {
		p, _ := a.Probabilities.Get(strconv.Itoa(i))
		probs[i] = p
	}
	return probs
}

// predictions pairs each candidate's probability with whether that candidate
// was the labelled one.
func predictions(probs []float64, correct int) []Prediction {
	out := make([]Prediction, len(probs))
	for i, p := range probs {
		out[i] = Prediction{P: p, Holds: i == correct}
	}
	return out
}

// Text renders the report for a human.
func (r Report) Text() string {
	var b strings.Builder

	b.WriteString("perceptea calibration benchmark\n\n")
	if r.Dataset != "" {
		fmt.Fprintf(&b, "dataset  %s\n", r.Dataset)
	}
	if r.Model != "" {
		fmt.Fprintf(&b, "model    %s\n", r.Model)
	}

	cov := r.Coverage
	fmt.Fprintf(&b, "\nCoverage\n")
	fmt.Fprintf(&b, "  %d cases x %d repeat = %d attempts\n", cov.Cases, cov.Repeat, cov.Attempted)
	fmt.Fprintf(&b, "  evaluated %d, failed %d\n", cov.Evaluated, cov.Failed)
	for _, f := range cov.Failures {
		fmt.Fprintf(&b, "    line %d (%s) attempt %d: %s\n", f.Line, f.ID, f.Attempt, f.Reason)
	}

	fmt.Fprintf(&b, "\nCalibration\n")
	fmt.Fprintf(&b, "  ECE %.4f over %d predicted probabilities, in %d equal-width bins\n",
		r.Calibration.ECE, r.Calibration.Predictions, BinCount)
	fmt.Fprintf(&b, "  a bin whose observed frequency is below its mean p is overconfident\n\n")

	tw := tabwriter.NewWriter(&b, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  bin\tcount\tmean p\tobserved\tgap")
	for i, bin := range r.Calibration.Bins {
		closing := ")"
		if i == len(r.Calibration.Bins)-1 {
			closing = "]"
		}
		span := fmt.Sprintf("[%.1f,%.1f%s", bin.Low, bin.High, closing)
		if bin.Count == 0 {
			fmt.Fprintf(tw, "  %s\t%d\t-\t-\t-\n", span, bin.Count)
			continue
		}
		fmt.Fprintf(tw, "  %s\t%d\t%.3f\t%.3f\t%.3f\n",
			span, bin.Count, bin.MeanPredicted, bin.ObservedFrequency,
			math.Abs(bin.MeanPredicted-bin.ObservedFrequency))
	}
	_ = tw.Flush()

	fmt.Fprintf(&b, "\nAnswers\n")
	if r.Choice == nil && r.Score == nil && r.Noul == nil {
		b.WriteString("  no case produced an answer\n")
		return b.String()
	}
	if c := r.Choice; c != nil {
		fmt.Fprintf(&b, "  choice  %d cases  accuracy %.3f (%d/%d)  Brier %.4f (multi-class over the options, summed, 0-2)\n",
			c.Cases, c.Accuracy, c.Correct, c.Cases, c.Brier)
	}
	if s := r.Score; s != nil {
		fmt.Fprintf(&b, "  score   %d cases  MAE %.3f levels [headline]  Brier %.4f (multi-class over the levels, summed, 0-2)\n",
			s.Cases, s.MAE, s.Brier)
	}
	if n := r.Noul; n != nil {
		fmt.Fprintf(&b, "  noul    %d cases  Brier %.4f ((p-y)^2, 0-1; answering 0.5 to everything scores 0.2500)\n",
			n.Cases, n.Brier)
	}
	return b.String()
}

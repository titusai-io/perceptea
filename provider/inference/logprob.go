package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/titusai-io/perceptea/classifier"
)

// The logprob scorer asks a different question from the chat scorer and reads
// the answer somewhere else.
//
// The chat scorer asks the model to write a probability and parses the number
// out of what it wrote. That number is quantised by writing habits: a model
// types 0.8, 0.9 and 0.95, and hardly ever 0.87, so the shape of the
// distribution owes as much to how a model likes to phrase a number as to what
// it believes.
//
// This scorer asks for a decision instead — is the statement true, Yes or No —
// and never reads the word that came back. It reads the distribution the model
// was about to sample that word from: P = exp(l_yes) / (exp(l_yes) + exp(l_no))
// over the two branches' log probabilities. That is continuous, it is the
// estimate rather than a description of one, and it costs one token of decode
// instead of ten.
//
// What it needs in return is a provider that reports logprobs. When one does
// not, the call fails; see [ErrNoLogprobs] for why that is not negotiable.

// logprobInstruction is the decision instruction this scorer works under, and
// the first thing in its prompt. As with the chat scorer's instruction, the
// numbers are calibrated against this exact wording and editing it moves them.
//
// It asks for one word rather than for a probability. Under a one-token cap
// the model never finishes a sentence anyway, and the word it would have
// written is not what is read: the wording exists to concentrate the first
// token's distribution on the two branches, so that the mass outside them is
// small enough to ignore.
const logprobInstruction = "You are a calibrated judge. " +
	"Given a STATE and a STATEMENT, decide whether the statement is true based solely on the state. " +
	"Do not invent facts. " +
	"Answer with one word and nothing else: Yes or No."

// logprobUserSuffix closes the user turn, repeating the two words the answer
// must be one of. The repetition is deliberate: the instruction is far away in
// the prompt by the time the model answers, and the last line is what the
// first token is conditioned on most strongly.
const logprobUserSuffix = "\n\nIs the statement true? Answer Yes or No."

// logprobPrefix renders the half of the prompt every candidate of one question
// shares, and logprobSuffix the half that changes. The cut falls exactly where
// the chat scorer's does, and for the same reason: every candidate of one
// question is judged against one instruction, one examples block and one
// state, so those three bytes-for-bytes repeat across the wave and an endpoint
// that caches a matching prefix serves the repeats from its cache. See the
// note above [promptPrefix].
//
// Nothing that varies by candidate may go in the prefix, here no less than
// there. The scorer changes what is asked; it does not change what may be
// shared.
func logprobPrefix(examples, state string) string {
	return promptPrefix(logprobInstruction, examples, state)
}

// logprobSuffix renders this one candidate's statement and the question about
// it.
func logprobSuffix(statement string) string {
	return scoreStatementLabel + statement + logprobUserSuffix
}

const (
	// maxLogprobTokens is the whole output budget: one token. The answer is
	// not read from the reply's text at all, so a second token would be
	// decode time spent on something nothing looks at.
	maxLogprobTokens = 1

	// logprobTopK is how many alternatives are asked for at the answer
	// position. It is the widest value the endpoints that implement this
	// field accept, and width is what makes the second branch findable: a
	// confident model can push "No" well down its list, and each of Yes,
	// yes, " Yes" and YES occupies a slot of its own. A narrower list costs
	// nothing on a confident answer — the single branch that did appear is
	// still usable, see [branchProbability] — but it is where the
	// only-one-branch case comes from.
	//
	// An endpoint that caps the field lower rejects the request rather than
	// silently narrowing it, and says which field it objected to; that is
	// one of the ways [ErrNoLogprobs] arrives.
	logprobTopK = 20
)

// ErrNoLogprobs reports that the logprob scorer asked a provider for token
// logprobs and did not get any: the reply carried no logprob block, or the
// provider rejected the request for asking.
//
// It is an error and not a quiet switch to the chat scorer, and not a 0.5,
// because both of those are degradations that look like answers. This package
// has been bitten by one already — a truncated reply used to score a neutral
// 0.5, which made every candidate of a request identical and turned a
// confident-looking response into pure noise; see [ErrTruncatedReply]. A
// silent fall back to the other scorer would be the same mistake wearing a
// better disguise: the service would keep answering, the numbers would quietly
// come from a different estimator than the one that was configured and
// calibrated against, and nothing in the response would say so.
//
// Callers classify it with [errors.Is]. The message names the setting to
// change and the model that would not answer, because the fix is a
// configuration change and the operator reading it has neither file open.
var ErrNoLogprobs = errors.New("inference: the provider returned no token logprobs, which the logprob scorer has nothing to do without")

// ErrNoDecisionToken reports that the provider did return logprobs, and that
// neither answer branch was anywhere among the most likely first tokens: the
// model was not about to answer Yes or No at all.
//
// This one is an error rather than the chat scorer's forgiving 0.5, and the
// difference between the two cases is whether the failure is about one
// candidate or about every one of them. An unreadable sentence is about one
// candidate: the model wrote prose about this statement and will write a
// number about the next. A first-token distribution with no answer in it is
// not about the candidate at all — the instruction, the layout and the answer
// vocabulary are identical on every call of the wave, so a model that is not
// answering Yes or No here is not answering Yes or No for anything, every
// candidate scores 0.5, and the softmax renders that uniform set as a
// confident answer carrying no information.
var ErrNoDecisionToken = errors.New("inference: neither answer token was among the provider's most likely first tokens")

// scoreByLogprob asks the model for a one-token decision and reads the
// probability out of the token distribution behind it.
//
// The request differs from the chat scorer's in three fields — logprobs,
// top_logprobs and a one-token cap — and in carrying no response_format at
// all. Structured output would be actively wrong here: a json_schema makes the
// first token "{", and the first token is the entire measurement. There is
// therefore nothing to negotiate down, and this path leaves the client's
// negotiated level exactly as it found it, so a chat call on the same client
// is unaffected either way.
//
// Everything else is the chat scorer's behaviour, because it is the same
// transport: the same retries and backoff, the same Retry-After handling, the
// same API key redaction and the same [APIError] classification.
//
// One shared habit does not carry over. Under a one-token cap every reply ends
// with finish_reason "length", because the cap is what ended it, so truncation
// carries no information here and [ErrTruncatedReply] is never returned.
//
// The request's temperature is passed through as it is for the chat scorer,
// and means less: nothing is being sampled that anyone reads. Endpoints
// differ over whether the logprobs they report are the raw ones or the ones
// the temperature scaled, so a temperature other than the usual 0 can move
// these numbers on some providers and not on others — which is a reason to
// leave it where the chat scorer has it rather than a second knob to tune.
func (c *Client) scoreByLogprob(ctx context.Context, req classifier.ScoreRequest) (classifier.ScoreResult, error) {
	model, err := c.resolveModel(req.Model)
	if err != nil {
		return classifier.ScoreResult{}, err
	}

	// Two messages, prefix then suffix, on the same cut and in the same roles
	// as the chat scorer uses; see logprobPrefix.
	body := chatRequest{
		Model: model,
		Messages: []chatMessage{
			{Role: "system", Content: logprobPrefix(req.Examples, req.State)},
			{Role: "user", Content: logprobSuffix(req.Statement)},
		},
		Temperature:     req.Temperature,
		MaxTokens:       maxLogprobTokens,
		Logprobs:        true,
		TopLogprobs:     logprobTopK,
		ReasoningEffort: c.reasoningEffort,
	}

	resp, err := c.complete(ctx, body)
	if err != nil {
		return classifier.ScoreResult{}, logprobCallError(err, model)
	}
	return c.logprobResult(resp, model)
}

// logprobResult turns a completion into a score, or into the error a reply
// with no usable distribution in it has earned.
func (c *Client) logprobResult(resp *chatResponse, model string) (classifier.ScoreResult, error) {
	top, ok := firstTokenDistribution(resp)
	if !ok {
		return classifier.ScoreResult{}, missingLogprobsError(model, resp)
	}
	p, ok := branchProbability(top)
	if !ok {
		return classifier.ScoreResult{}, c.noDecisionError(model, top)
	}
	return classifier.ScoreResult{
		Probability:  p,
		InputTokens:  resp.Usage.PromptTokens,
		OutputTokens: resp.Usage.CompletionTokens,
	}, nil
}

// firstTokenDistribution returns the candidate tokens for the answer
// position, and reports whether the reply carried a distribution at all.
//
// Only the first generated token is looked at, because under the one-token cap
// it is the only one, and because it is the one the answer vocabulary is
// concentrated on.
//
// The token the provider actually sampled is not consulted while a top-k list
// exists. A sample is one draw from the distribution; the distribution is the
// estimate, and reading the draw instead would throw away everything the
// measurement is for. The sampled token is used only as a last resort, for a
// provider that honours logprobs by reporting the chosen token's own
// probability and no alternatives beside it.
func firstTokenDistribution(resp *chatResponse) ([]topLogprob, bool) {
	if len(resp.Choices) == 0 {
		return nil, false
	}
	lp := resp.Choices[0].Logprobs
	if lp == nil || len(lp.Content) == 0 {
		return nil, false
	}
	first := lp.Content[0]
	if len(first.TopLogprobs) > 0 {
		return first.TopLogprobs, true
	}
	if first.Token == "" {
		return nil, false
	}
	return []topLogprob{{Token: first.Token, Logprob: first.Logprob}}, true
}

// branchProbability recovers P(statement is true) from the answer position's
// distribution, and reports whether either branch was in it at all.
//
// With both branches present the answer is the softmax over the two:
//
//	P = exp(l_yes) / (exp(l_yes) + exp(l_no)) = logistic(l_yes − l_no)
//
// which is computed from the difference and never from the two exponentials.
// A confident answer is exactly where the naive form breaks: a logprob of
// −800 exponentiates to zero, 0/(0+0) is NaN, and a NaN scored as the neutral
// 0.5 would turn the model's most certain answers into its least informative
// ones. The difference is small even when the terms are not.
//
// Each branch takes the total mass of every spelling of it that appeared —
// "Yes", " Yes", "yes", "YES" are one branch, not four — summed in log space
// for the same reason.
//
// With only one branch present, that branch's own probability is used as it
// stands: p_yes when it was the yes branch, 1 − p_no when it was the no
// branch. This is sound in the direction that matters. The renormalised value
// is p_yes / (p_yes + p_no), and p_yes + p_no ≤ 1, so the unnormalised p_yes is
// a lower bound on it; symmetrically 1 − p_no is an upper bound on a value
// that is near zero. Both err towards 0.5 — towards the missing mass being an
// answer that was not given — so the one thing this cannot do is invent
// confidence. It is also barely an approximation in practice: a branch absent
// from twenty alternatives has less mass than the twentieth, which is where
// the rest of the vocabulary lives.
//
// Tokens that are neither branch are ignored rather than counted against
// either. The question is which of the two answers the model was giving, not
// how much of its attention the answer had; a model hedging its first token on
// "Based" has not thereby become uncertain about the statement.
func branchProbability(top []topLogprob) (float64, bool) {
	var yes, no []float64
	for _, t := range top {
		// A logprob that is not a finite number is not a measurement. It
		// cannot arrive through JSON, and dropping it here is what keeps a
		// NaN from reaching the arithmetic and coming out as a 0.5.
		v, ok := finite(t.Logprob)
		if !ok {
			continue
		}
		switch branchOf(t.Token) {
		case branchYes:
			yes = append(yes, v)
		case branchNo:
			no = append(no, v)
		}
	}

	switch {
	case len(yes) > 0 && len(no) > 0:
		return clamp01(logistic(logSumExp(yes) - logSumExp(no))), true
	case len(yes) > 0:
		return clamp01(math.Exp(logSumExp(yes))), true
	case len(no) > 0:
		return clamp01(1 - math.Exp(logSumExp(no))), true
	default:
		return 0, false
	}
}

// branch is which answer a token stands for, if either.
type branch int

const (
	branchNone branch = iota
	branchYes
	branchNo
)

// branchOf classifies one token from the answer position.
//
// The two spellings of each branch are the word asked for and the boolean it
// stands for: a model told to answer Yes or No sometimes reaches for true or
// false instead, and both are the same decision.
//
// A single letter is deliberately not a branch. "y" and "n" would be tempting
// — a model does occasionally answer Y — but "n" also begins "no", "not",
// "never", "neither" and "nothing", and a tokenizer that splits any of those
// would hand this function a letter that means nothing of the sort. A branch
// that can be wrong is worse than a branch that is sometimes missing, because
// the missing case is handled and the wrong one is not visible.
func branchOf(token string) branch {
	switch normalizeToken(token) {
	case "yes", "true":
		return branchYes
	case "no", "false":
		return branchNo
	}
	return branchNone
}

// subwordMarkers are the word-boundary marks some tokenizers leave on a token
// that follows a space: U+2581 and U+0120. A provider that decodes its tokens
// to text sends neither, and one that does not sends "▁Yes" where it means
// " Yes".
const subwordMarkers = "▁Ġ"

// answerPunctuation is what may cling to an answer word without changing which
// answer it is: the quote a model opens with, and the punctuation another ends
// on.
const answerPunctuation = "\"'`.,:;!*-"

// normalizeToken reduces a token to the word in it, so that the several ways a
// provider can spell one answer all land on one branch.
//
// Leading spaces are the common case and the reason this exists at all: most
// tokenizers make " Yes" a different token from "Yes" and both are perfectly
// ordinary first tokens. Case is the next: a model that shouts YES means yes.
//
// The trimming can only ever remove characters, so nothing that was not
// already an answer word can be turned into one.
func normalizeToken(token string) string {
	s := strings.TrimLeft(token, subwordMarkers)
	s = strings.TrimFunc(s, func(r rune) bool {
		return unicode.IsSpace(r) || strings.ContainsRune(answerPunctuation, r)
	})
	return strings.ToLower(s)
}

// logistic turns a log-odds difference into a probability, which is the
// softmax over two branches written so that it cannot overflow: the
// exponential is always of a non-positive number, whatever the sign or the
// magnitude of the difference.
func logistic(d float64) float64 {
	if d >= 0 {
		return 1 / (1 + math.Exp(-d))
	}
	e := math.Exp(d)
	return e / (1 + e)
}

// logSumExp returns log(Σ exp(v)) for a non-empty slice of finite values,
// factoring out the largest so that no exponential is ever taken of a positive
// number and the smallest terms underflow harmlessly to zero instead of the
// largest one deciding the sum is zero.
func logSumExp(vs []float64) float64 {
	if len(vs) == 1 {
		return vs[0]
	}
	m := vs[0]
	for _, v := range vs[1:] {
		if v > m {
			m = v
		}
	}
	var sum float64
	for _, v := range vs {
		sum += math.Exp(v - m)
	}
	return m + math.Log(sum)
}

// logprobComplaintPattern matches a provider naming the field it could not
// honour. It is narrow on purpose: the word "logprob" in any of its spellings,
// and nothing looser, so that an error about the caller's model or prompt is
// not read as an error about this setting.
var logprobComplaintPattern = regexp.MustCompile(`(?i)log[_ -]?prob`)

// logprobCallError classifies a failed request. A provider that rejected the
// call because it does not do logprobs has said exactly what [ErrNoLogprobs]
// is for, and reporting it as an unexplained upstream error would bury the one
// sentence that tells the operator what to change. Everything else is passed
// through untouched, so retry classification, redaction and [APIError]
// inspection behave as they do on any other call.
//
// The provider's own error stays in the chain beside the sentinel, so a caller
// that wants the status code still finds the [APIError] with [errors.As].
func logprobCallError(err error, model string) error {
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		return err
	}
	if !logprobComplaintPattern.MatchString(apiErr.Message) &&
		!logprobComplaintPattern.MatchString(apiErr.Body) {
		return err
	}
	return fmt.Errorf("%w: %s asked model %q for logprobs and the provider rejected the request (%w). %s",
		ErrNoLogprobs, setting(), model, err, theWayOut)
}

// missingLogprobsError explains a reply that came back 200 and carried no
// distribution, naming which of the three ways it managed that.
func missingLogprobsError(model string, resp *chatResponse) error {
	var finding string
	switch {
	case len(resp.Choices) == 0:
		finding = "the reply carried no choices at all"
	case resp.Choices[0].Logprobs == nil:
		finding = "the reply carried no logprobs field"
	case len(resp.Choices[0].Logprobs.Content) == 0:
		finding = "the reply's logprobs field held no tokens"
	default:
		finding = "the reply's logprobs field named no token at the answer position"
	}
	return fmt.Errorf("%w: %s asked model %q for logprobs (logprobs=true, top_logprobs=%d) and %s. %s",
		ErrNoLogprobs, setting(), model, logprobTopK, finding, theWayOut)
}

// noDecisionError explains a distribution with no answer in it, and shows what
// the model offered instead — which is the whole diagnosis, and unguessable
// from anywhere else.
func (c *Client) noDecisionError(model string, top []topLogprob) error {
	return fmt.Errorf("%w: %s asked model %q whether a statement is true and none of %s was among its %d most likely first tokens; it offered %s. "+
		"Either the model does not follow the instruction, or it generates something ahead of the answer, such as a reasoning preamble: "+
		"set %s to %q if it reasons, use a model that answers as instructed, or set %s to %q, which reads a probability the model writes out instead",
		ErrNoDecisionToken, setting(), model, answerWords, len(top), c.tokenList(top), envReasoningEffort, effortNone, envScorer, scorerChat)
}

// answerWords is the answer vocabulary as an error message should list it.
const answerWords = `"yes", "no", "true" or "false"`

// setting names the configuration that selected this scorer, as it would be
// written in an environment.
func setting() string { return envScorer + "=" + scorerLogprob }

// theWayOut is the last sentence of every [ErrNoLogprobs]: an operator reading
// a 502 needs both ways out of it, and neither is discoverable from the
// finding alone.
const theWayOut = "Use an endpoint and model that return token logprobs, or set " +
	envScorer + " to " + `"` + scorerChat + `"` + ", which asks the model to write the probability instead"

// maxListedTokens is how many of the offered tokens an error message shows.
// Enough to recognise what the model was doing, few enough that a 502 body
// stays readable.
const maxListedTokens = 5

// tokenList renders the tokens a provider offered, quoted so that a leading
// space is visible, and redacted for the same reason every other provider
// string is: a gateway that echoes a key into its output should not be helped
// to put it somewhere else.
func (c *Client) tokenList(top []topLogprob) string {
	shown := top
	if len(shown) > maxListedTokens {
		shown = shown[:maxListedTokens]
	}
	quoted := make([]string, 0, len(shown))
	for _, t := range shown {
		quoted = append(quoted, strconv.Quote(c.redact(t.Token)))
	}
	out := strings.Join(quoted, ", ")
	if len(top) > len(shown) {
		out += fmt.Sprintf(" and %d more", len(top)-len(shown))
	}
	return out
}

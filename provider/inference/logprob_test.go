package inference

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"strings"
	"testing"

	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
)

// The logprob prompt written out independently of the constants the client
// uses, so that an edit to one is caught by the other.
const wantLogprobInstruction = "You are a calibrated judge. " +
	"Given a STATE and a STATEMENT, decide whether the statement is true based solely on the state. " +
	"Do not invent facts. " +
	"Answer with one word and nothing else: Yes or No."

// wantLogprobPrefix and wantLogprobSuffix are the two messages for
// fixtureRequest. The cut is the chat scorer's cut: everything shared by a
// question's candidates on one side, the one candidate's statement on the
// other.
const (
	wantLogprobPrefix = wantLogprobInstruction + "\n\nSTATE:\nthe sky is grey"
	wantLogprobSuffix = "STATEMENT:\nIt is raining.\n\nIs the statement true? Answer Yes or No."
)

// logprobToken is one alternative at the answer position.
type logprobToken struct {
	token   string
	logprob float64
}

// logprobFixture is one completion as a provider that supports logprobs sends
// it, assembled field by field so that a test can stage each of the ways a
// reply can arrive without a usable distribution in it.
type logprobFixture struct {
	// chosen is the token the provider sampled, and chosenLogprob its own
	// log probability. The scorer is not supposed to read either while
	// there are alternatives, which is what makes them useful in a fixture.
	chosen        string
	chosenLogprob float64
	// top is the alternatives at that position.
	top []logprobToken
	// The three ways a reply can carry no distribution, and the fourth.
	omitTop      bool
	omitContent  bool
	omitLogprobs bool
	noChoices    bool

	finishReason     string
	promptTokens     int
	completionTokens int
}

// body renders the fixture as a response document.
func (f logprobFixture) body() string {
	choice := map[string]any{
		"message": map[string]any{"role": "assistant", "content": f.chosen},
	}
	if f.finishReason != "" {
		choice["finish_reason"] = f.finishReason
	}
	if !f.omitLogprobs {
		content := []any{}
		if !f.omitContent {
			entry := map[string]any{"token": f.chosen, "logprob": f.chosenLogprob}
			if !f.omitTop {
				tops := make([]any, 0, len(f.top))
				for _, t := range f.top {
					tops = append(tops, map[string]any{"token": t.token, "logprob": t.logprob})
				}
				entry["top_logprobs"] = tops
			}
			content = append(content, entry)
		}
		choice["logprobs"] = map[string]any{"content": content}
	}

	choices := []any{choice}
	if f.noChoices {
		choices = []any{}
	}
	document := map[string]any{"choices": choices}
	if f.promptTokens != 0 || f.completionTokens != 0 {
		document["usage"] = map[string]any{
			"prompt_tokens":     f.promptTokens,
			"completion_tokens": f.completionTokens,
		}
	}
	raw, _ := json.Marshal(document)
	return string(raw)
}

// The natural logs of the probabilities the fixtures below are written in
// terms of. A fixture states the probabilities it means — 0.6 of "Yes", 0.2
// of "No" — and the expected score is worked out from those by hand; these
// constants only carry them onto the wire in the units a provider uses.
const (
	lnPoint05 = -2.995732273553991  // ln(0.05)
	lnPoint1  = -2.3025850929940455 // ln(0.1)
	lnPoint2  = -1.6094379124341003 // ln(0.2)
	lnPoint3  = -1.2039728043259361 // ln(0.3)
	lnPoint4  = -0.916290731874155  // ln(0.4)
	lnPoint5  = -0.6931471805599453 // ln(0.5)
	lnPoint6  = -0.5108256237659907 // ln(0.6)
	lnPoint8  = -0.2231435513142097 // ln(0.8)
	lnPoint9  = -0.10536051565782628
)

// yesNo is the distribution most tests want: 0.6 on the yes branch and 0.2 on
// the no branch, which is P = 0.6 / (0.6 + 0.2) = 0.75.
//
// The two are deliberately unequal, and deliberately do not sum to 1. A pair
// that is equally likely scores 0.5 whatever the arithmetic does with it and
// so cannot tell a correct computation from a broken one; a pair that sums to
// 1 cannot tell the renormalised answer from either branch's raw probability.
// Here the three candidate answers are 0.75, 0.6 and 0.8, and only one of them
// is right.
var yesNo = []logprobToken{{" Yes", lnPoint6}, {" No", lnPoint2}}

// wantYesNo is that fixture's score, worked out by hand:
//
//	P = 0.6 / (0.6 + 0.2) = 0.6 / 0.8 = 0.75
const wantYesNo = 0.75

// newLogprobClient builds a client configured for the logprob scorer.
func newLogprobClient(t *testing.T, api *fakeAPI, mutate func(*Config)) (*Client, *sleepLog) {
	t.Helper()
	return newTestClient(t, api, func(cfg *Config) {
		cfg.Scorer = "logprob"
		if mutate != nil {
			mutate(cfg)
		}
	})
}

// closeTo reports whether two probabilities agree to within floating-point
// noise. The tolerance is far tighter than the difference between any two
// answers a fixture here can produce.
func closeTo(got, want float64) bool { return math.Abs(got-want) <= 1e-9 }

// The request is the measurement. Without logprobs there is nothing to read,
// without top_logprobs there is one branch at most, and without the one-token
// cap the reply is a sentence whose first token is the only part anyone looks
// at.
func TestLogprobScorerSendsTheExactRequest(t *testing.T) {
	api := alwaysJSON(t, logprobFixture{chosen: " Yes", chosenLogprob: lnPoint6, top: yesNo, finishReason: "length"}.body())
	client, _ := newLogprobClient(t, api, func(cfg *Config) {
		cfg.Referer = "https://perceptea.example"
		cfg.Title = "Perceptea"
	})

	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}

	req := api.request(t, 0)
	if req.method != http.MethodPost {
		t.Errorf("method = %q, want POST", req.method)
	}
	if req.path != "/chat/completions" {
		t.Errorf("path = %q, want /chat/completions", req.path)
	}
	if got, want := req.header.Get("Authorization"), "Bearer "+testKey; got != want {
		t.Errorf("Authorization = %q, want %q", got, want)
	}
	if got := req.header.Get("HTTP-Referer"); got != "https://perceptea.example" {
		t.Errorf("HTTP-Referer = %q", got)
	}

	body := req.body(t)
	if got, want := body["model"], "request-model"; got != want {
		t.Errorf("model = %v, want %v", got, want)
	}
	if got, want := body["temperature"], 0.25; got != want {
		t.Errorf("temperature = %v, want %v", got, want)
	}
	if got, want := body["max_tokens"], 1.0; got != want {
		t.Errorf("max_tokens = %v, want %v: the answer is one token and the rest is decode nothing reads", got, want)
	}
	if got, want := body["logprobs"], true; got != want {
		t.Errorf("logprobs = %v, want %v: without it the reply carries no distribution", got, want)
	}
	if got, want := body["top_logprobs"], 20.0; got != want {
		t.Errorf("top_logprobs = %v, want %v: a narrow list is where a branch goes missing", got, want)
	}
	// A json_schema would make the first token "{", and the first token is
	// the whole measurement.
	if rf, ok := body["response_format"]; ok {
		t.Errorf("response_format = %v, want none on a logprob call", rf)
	}

	msgs := api.messages(t, 0)
	if msgs[0].Role != "system" {
		t.Errorf("messages[0].role = %q, want system", msgs[0].Role)
	}
	if msgs[0].Content != wantLogprobPrefix {
		t.Errorf("prefix mismatch\n got: %q\nwant: %q", msgs[0].Content, wantLogprobPrefix)
	}
	if msgs[1].Role != "user" {
		t.Errorf("messages[1].role = %q, want user", msgs[1].Role)
	}
	if msgs[1].Content != wantLogprobSuffix {
		t.Errorf("suffix mismatch\n got: %q\nwant: %q", msgs[1].Content, wantLogprobSuffix)
	}

	// Nothing about this call teaches the client anything about structured
	// output, because it asked for none.
	if got := client.outputLevel(); got != levelJSONSchema {
		t.Errorf("the logprob call moved the structured-output level to %v; it sends no response_format and must leave it alone", got)
	}
}

// The prefix is what an endpoint can cache, and it can only cache it while
// every candidate of the wave sends the same bytes. The logprob scorer
// changes the question, not the layout.
func TestTheLogprobPrefixIsByteIdenticalForEveryCandidateOfAQuestion(t *testing.T) {
	const instructions = "Which team should handle this?"
	statements := []string{
		classifier.ChoiceStatement("department", instructions, "billing", "Charges, refunds, invoices"),
		classifier.ChoiceStatement("department", instructions, "technical", "Bugs"),
		classifier.ChoiceStatement("department", instructions, "sales", "Pricing and plans"),
		classifier.ChoiceStatement("department", instructions, "other", ""),
	}
	// Two candidates rendering the same statement would make every assertion
	// below true for the wrong reason.
	for i := range statements {
		for j := i + 1; j < len(statements); j++ {
			if statements[i] == statements[j] {
				t.Fatalf("candidates %d and %d render the same statement %q; "+
					"this fixture cannot see a candidate leaking into the prefix", i, j, statements[i])
			}
		}
	}

	api := alwaysJSON(t, logprobFixture{chosen: " Yes", top: yesNo}.body())
	client, _ := newLogprobClient(t, api, nil)

	for _, statement := range statements {
		req := fixtureRequest
		req.Statement = statement
		req.Examples = fixtureExamples
		if _, err := client.Score(context.Background(), req); err != nil {
			t.Fatalf("Score(%q): %v", statement, err)
		}
	}
	if api.count() != len(statements) {
		t.Fatalf("made %d calls, want %d (one per candidate)", api.count(), len(statements))
	}

	first := api.messages(t, 0)
	for i, statement := range statements {
		msgs := api.messages(t, i)
		if msgs[0].Content != first[0].Content {
			t.Errorf("candidate %d sent a different prefix, so the wave has none to cache\n got: %q\nwant: %q",
				i, msgs[0].Content, first[0].Content)
		}
		if strings.Contains(msgs[0].Content, statement) {
			t.Errorf("candidate %d's statement is in the shared prefix: %q", i, msgs[0].Content)
		}
		if !strings.Contains(msgs[1].Content, statement) {
			t.Errorf("candidate %d's statement is not in its suffix: %q", i, msgs[1].Content)
		}
		if i > 0 && msgs[1].Content == first[1].Content {
			t.Errorf("candidate %d sent candidate 0's suffix; the candidate is not reaching the model", i)
		}
	}
}

// The state and the examples are paid for once per wave, the statement once
// per candidate, so each has to be on its own side of the cut.
func TestTheLogprobPromptPutsTheSharedPartsInThePrefix(t *testing.T) {
	api := alwaysJSON(t, logprobFixture{chosen: " Yes", top: yesNo}.body())
	client, _ := newLogprobClient(t, api, nil)

	req := fixtureRequest
	req.Examples = fixtureExamples
	if _, err := client.Score(context.Background(), req); err != nil {
		t.Fatalf("Score: %v", err)
	}

	msgs := api.messages(t, 0)
	want := wantLogprobInstruction + "\n\n" + fixtureExamples + "\n\nSTATE:\nthe sky is grey"
	if msgs[0].Content != want {
		t.Errorf("prefix mismatch\n got: %q\nwant: %q", msgs[0].Content, want)
	}
	if strings.Contains(msgs[0].Content, fixtureRequest.Statement) {
		t.Errorf("the statement is in the prefix, which every candidate shares: %q", msgs[0].Content)
	}
	if strings.Contains(strings.ToUpper(msgs[1].Content), "EXAMPLE") {
		t.Errorf("the examples reached the per-candidate suffix: %q", msgs[1].Content)
	}
	if strings.Contains(msgs[1].Content, fixtureRequest.State) {
		t.Errorf("the state is in the per-candidate suffix, so every call re-sends it: %q", msgs[1].Content)
	}
}

// The arithmetic, against distributions whose answers were worked out by hand
// from the probabilities each fixture names.
func TestLogprobRecoversTheProbabilityFromTheBranches(t *testing.T) {
	for _, tc := range []struct {
		name string
		top  []logprobToken
		want float64
	}{
		{
			// P = 0.6 / (0.6 + 0.2) = 0.75
			"both branches",
			yesNo,
			0.75,
		},
		{
			// P = 0.1 / (0.1 + 0.3) = 0.25 — the same arithmetic the other
			// way up, so a sign error cannot pass both.
			"the no branch ahead",
			[]logprobToken{{" Yes", lnPoint1}, {" No", lnPoint3}},
			0.25,
		},
		{
			// Spellings of one word are one branch: yes is 0.4 + 0.2 = 0.6
			// against no's 0.2, so P = 0.6 / 0.8 = 0.75. Reading only the
			// likeliest spelling would give 0.4 / 0.6 = 0.667.
			"the spellings of a branch are summed",
			[]logprobToken{{" Yes", lnPoint4}, {"Yes", lnPoint2}, {" No", lnPoint2}},
			0.75,
		},
		{
			// Case is a spelling too: 0.3 + 0.3 = 0.6 against 0.2.
			"an upper case spelling counts",
			[]logprobToken{{" Yes", lnPoint3}, {"YES", lnPoint3}, {" No", lnPoint2}},
			0.75,
		},
		{
			// true and false are the same decision in other words.
			// 0.6 against 0.2 again.
			"the boolean spellings",
			[]logprobToken{{"true", lnPoint6}, {"false", lnPoint2}},
			0.75,
		},
		{
			// A first token that is neither branch is not evidence about the
			// statement: P is still 0.3 / (0.3 + 0.1) = 0.75, not
			// 0.3 / (0.5 + 0.3 + 0.1) = 0.333.
			"tokens that are neither branch are ignored",
			[]logprobToken{{"Maybe", lnPoint5}, {" Yes", lnPoint3}, {" No", lnPoint1}},
			0.75,
		},
		{
			// Only the yes branch made the list, so its own probability
			// stands: 0.9.
			"only the yes branch",
			[]logprobToken{{" Yes", lnPoint9}, {"Probably", lnPoint05}},
			0.9,
		},
		{
			// Only the no branch: 1 − 0.8 = 0.2.
			"only the no branch",
			[]logprobToken{{" No", lnPoint8}, {"Unlikely", lnPoint1}},
			0.2,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := alwaysJSON(t, logprobFixture{
				chosen: tc.top[0].token, chosenLogprob: tc.top[0].logprob, top: tc.top,
				finishReason: "length",
			}.body())
			client, _ := newLogprobClient(t, api, nil)

			got, err := client.Score(context.Background(), fixtureRequest)
			if err != nil {
				t.Fatalf("Score: %v", err)
			}
			if !closeTo(got.Probability, tc.want) {
				t.Errorf("probability = %v, want %v", got.Probability, tc.want)
			}
			if got.Probability < 0 || got.Probability > 1 {
				t.Errorf("probability %v escaped [0,1]", got.Probability)
			}
		})
	}
}

// A confident model is where the naive formula fails: exp(−800) is zero in
// float64, exp(−800)/(exp(−800)+exp(−1000)) is 0/0, and a NaN read as the
// neutral 0.5 would turn the model's most certain answers into its least
// informative ones — silently, and in the direction that looks like a result.
// The difference of the two logprobs is small even when the logprobs are not.
func TestLogprobDoesNotUnderflowOnAConfidentAnswer(t *testing.T) {
	for _, tc := range []struct {
		name              string
		yes, no           float64
		lower, upper      float64
		wantNearCertainty string
	}{
		{
			// l_yes − l_no = 200, so P = 1/(1 + e^−200) and e^−200 ≈ 1.4e−87:
			// P is 1 to every digit float64 has.
			"certain the statement is true", -800, -1000, 1 - 1e-12, 1,
			"a certain yes",
		},
		{
			// The mirror image: P = e^−200/(1 + e^−200) ≈ 1.4e−87. Small, and
			// not zero — the arithmetic has room for it.
			"certain the statement is false", -1000, -800, 1e-90, 1e-80,
			"a certain no",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := alwaysJSON(t, logprobFixture{
				chosen: " Yes", top: []logprobToken{{" Yes", tc.yes}, {" No", tc.no}},
			}.body())
			client, _ := newLogprobClient(t, api, nil)

			got, err := client.Score(context.Background(), fixtureRequest)
			if err != nil {
				t.Fatalf("Score: %v", err)
			}
			if math.IsNaN(got.Probability) {
				t.Fatalf("probability is NaN on %s", tc.wantNearCertainty)
			}
			if got.Probability == fallbackProbability {
				t.Fatalf("%s scored the neutral %v: the exponentials underflowed and the NaN was read as no information",
					tc.wantNearCertainty, fallbackProbability)
			}
			if got.Probability < tc.lower || got.Probability > tc.upper {
				t.Errorf("probability = %g, want between %g and %g for %s",
					got.Probability, tc.lower, tc.upper, tc.wantNearCertainty)
			}
		})
	}
}

// The sampled token is one draw from the distribution; the distribution is
// the estimate. Reading the draw would throw away everything the measurement
// is for — and would score this reply 0.2 rather than 0.75.
func TestLogprobReadsTheDistributionAndNotTheSampledToken(t *testing.T) {
	api := alwaysJSON(t, logprobFixture{
		chosen: " No", chosenLogprob: lnPoint2, top: yesNo, finishReason: "length",
	}.body())
	client, _ := newLogprobClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if !closeTo(got.Probability, wantYesNo) {
		t.Errorf("probability = %v, want %v: the answer comes from the distribution, not from the token that was sampled",
			got.Probability, wantYesNo)
	}
}

// A provider that reports the chosen token's own probability and no
// alternatives beside it has still said something, and it is the only thing
// left to read.
func TestLogprobFallsBackToTheSampledTokenWithoutAlternatives(t *testing.T) {
	api := alwaysJSON(t, logprobFixture{
		chosen: " Yes", chosenLogprob: lnPoint9, omitTop: true,
	}.body())
	client, _ := newLogprobClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if !closeTo(got.Probability, 0.9) {
		t.Errorf("probability = %v, want 0.9: the sampled token's own probability", got.Probability)
	}
}

// Under a one-token cap the cap is what ends every reply, so finish_reason is
// always "length" and it means nothing at all. Treating it as truncation
// would fail every call the scorer ever makes.
func TestALengthFinishReasonIsNormalUnderTheOneTokenCap(t *testing.T) {
	api := alwaysJSON(t, logprobFixture{
		chosen: " Yes", chosenLogprob: lnPoint6, top: yesNo, finishReason: "length",
	}.body())
	client, _ := newLogprobClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v — a one-token reply always stops at the cap; that is not a truncated answer", err)
	}
	if errors.Is(err, ErrTruncatedReply) {
		t.Fatal("a one-token reply was reported as truncated")
	}
	if !closeTo(got.Probability, wantYesNo) {
		t.Errorf("probability = %v, want %v", got.Probability, wantYesNo)
	}
}

// A provider that will not return logprobs is a configuration fault that hits
// every candidate of every request, and the one degradation this package
// refuses to dress up as an answer: no quiet 0.5, and no quiet switch to the
// other scorer.
func TestLogprobRejectsAReplyWithNoLogprobs(t *testing.T) {
	for _, tc := range []struct {
		name    string
		reply   logprobFixture
		finding string
	}{
		{
			"no logprobs field at all",
			logprobFixture{chosen: "Yes", omitLogprobs: true},
			"no logprobs field",
		},
		{
			"a logprobs field with no tokens in it",
			logprobFixture{chosen: "Yes", omitContent: true},
			"held no tokens",
		},
		{
			"a token entry naming no token",
			logprobFixture{chosen: "", omitTop: true},
			"named no token",
		},
		{
			"no choices at all",
			logprobFixture{noChoices: true},
			"no choices",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := alwaysJSON(t, tc.reply.body())
			client, _ := newLogprobClient(t, api, nil)

			got, err := client.Score(context.Background(), fixtureRequest)
			if err == nil {
				t.Fatalf("Score returned %v and no error; a reply with no logprobs must not be scored", got)
			}
			if !errors.Is(err, ErrNoLogprobs) {
				t.Fatalf("Score returned %v, want ErrNoLogprobs", err)
			}
			if got.Probability == fallbackProbability {
				t.Error("the fallback probability came back alongside the error")
			}

			// The message is what an operator gets in a 502: the finding, the
			// model that would not answer, the setting that asked, and both
			// ways out.
			msg := err.Error()
			for _, want := range []string{
				tc.finding, "request-model", "PERCEPTEA_SCORER", "logprob", `"chat"`,
			} {
				if !strings.Contains(msg, want) {
					t.Errorf("the error does not mention %q: %s", want, msg)
				}
			}
		})
	}
}

// Logprobs that contain no answer are the same failure in a different place:
// the instruction and the answer vocabulary are identical on every call, so a
// model that is not answering Yes or No here is not answering for any
// candidate, and every score would be the same 0.5.
func TestLogprobRejectsADistributionWithNoAnswerInIt(t *testing.T) {
	api := alwaysJSON(t, logprobFixture{
		chosen: "Based", chosenLogprob: lnPoint5,
		top: []logprobToken{{"Based", lnPoint5}, {"The", lnPoint3}, {"I", lnPoint1}},
	}.body())
	client, _ := newLogprobClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err == nil {
		t.Fatalf("Score returned %v and no error; a first token that decides nothing is not a probability", got)
	}
	if !errors.Is(err, ErrNoDecisionToken) {
		t.Fatalf("Score returned %v, want ErrNoDecisionToken", err)
	}
	if got.Probability == fallbackProbability {
		t.Error("the fallback probability came back alongside the error")
	}

	msg := err.Error()
	// What the model offered instead is the whole diagnosis and is not
	// available anywhere else.
	for _, want := range []string{`"Based"`, `"The"`, "request-model", "PERCEPTEA_SCORER", `"chat"`} {
		if !strings.Contains(msg, want) {
			t.Errorf("the error does not mention %q: %s", want, msg)
		}
	}
}

// Some providers refuse the request rather than answering it without
// logprobs. That is the same finding arriving earlier, and it deserves the
// same sentence about what to change — but only when the provider actually
// said so.
func TestLogprobReportsAProviderThatRefusesToDoLogprobs(t *testing.T) {
	for _, tc := range []struct {
		name          string
		status        int
		body          string
		wantSentinel  bool
		wantInMessage string
	}{
		{
			"the provider names the field",
			http.StatusBadRequest,
			`{"error":{"message":"logprobs is not supported for this model","type":"invalid_request_error"}}`,
			true,
			"request-model",
		},
		{
			"the provider names it with an underscore, nested where the parser cannot reach",
			http.StatusUnprocessableEntity,
			`{"detail":[{"loc":["body","top_logprobs"],"msg":"extra fields not permitted"}]}`,
			true,
			"PERCEPTEA_SCORER",
		},
		{
			// The control. A rejection that says nothing about logprobs is
			// not evidence about logprobs, and calling it one would send an
			// operator to change a setting that was never the problem.
			"a rejection about something else entirely",
			http.StatusBadRequest,
			`{"error":{"message":"model not found: request-model","type":"invalid_request_error"}}`,
			false,
			"model not found",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
				writeJSON(w, tc.status, tc.body)
			})
			client, _ := newLogprobClient(t, api, nil)

			_, err := client.Score(context.Background(), fixtureRequest)
			if err == nil {
				t.Fatal("Score returned no error")
			}
			if got := errors.Is(err, ErrNoLogprobs); got != tc.wantSentinel {
				t.Errorf("errors.Is(err, ErrNoLogprobs) = %v, want %v: %v", got, tc.wantSentinel, err)
			}
			// The provider's own error stays reachable either way, so a
			// caller that wants the status code still has it.
			var apiErr *APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("the provider's APIError is not in the chain: %v", err)
			}
			if apiErr.StatusCode != tc.status {
				t.Errorf("status = %d, want %d", apiErr.StatusCode, tc.status)
			}
			if !strings.Contains(err.Error(), tc.wantInMessage) {
				t.Errorf("the error does not mention %q: %v", tc.wantInMessage, err)
			}
		})
	}
}

// The logprob scorer is a second question, not a second HTTP path: it retries
// what the chat scorer retries, backs off the same way, and redacts the key
// out of whatever a provider echoes back.
func TestLogprobUsesTheSameTransportAsTheChatScorer(t *testing.T) {
	t.Run("a transient failure is retried", func(t *testing.T) {
		api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
			if n == 0 {
				writeJSON(w, http.StatusInternalServerError, `{"error":{"message":"upstream hiccup"}}`)
				return
			}
			writeJSON(w, http.StatusOK, logprobFixture{chosen: " Yes", top: yesNo}.body())
		})
		client, sleeps := newLogprobClient(t, api, nil)

		got, err := client.Score(context.Background(), fixtureRequest)
		if err != nil {
			t.Fatalf("Score: %v", err)
		}
		if api.count() != 2 {
			t.Errorf("made %d attempts, want 2", api.count())
		}
		if len(sleeps.durations()) != 1 {
			t.Errorf("backed off %d times, want 1", len(sleeps.durations()))
		}
		if !closeTo(got.Probability, wantYesNo) {
			t.Errorf("probability = %v, want %v", got.Probability, wantYesNo)
		}
	})

	t.Run("an echoed key is redacted", func(t *testing.T) {
		api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
			writeJSON(w, http.StatusUnauthorized,
				`{"error":{"message":"key `+testKey+` is not valid for logprobs"}}`)
		})
		client, _ := newLogprobClient(t, api, nil)

		_, err := client.Score(context.Background(), fixtureRequest)
		if err == nil {
			t.Fatal("Score returned no error")
		}
		if strings.Contains(err.Error(), testKey) {
			t.Errorf("the API key is in the error: %v", err)
		}
		if !strings.Contains(err.Error(), "[REDACTED]") {
			t.Errorf("the key was not redacted, it was lost: %v", err)
		}
	})
}

// Usage is reported the same way whichever scorer ran; a caller sums it
// across a wave either way.
func TestLogprobReadsUsage(t *testing.T) {
	api := alwaysJSON(t, logprobFixture{
		chosen: " Yes", top: yesNo, promptTokens: 412, completionTokens: 1,
	}.body())
	client, _ := newLogprobClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got.InputTokens != 412 || got.OutputTokens != 1 {
		t.Errorf("usage = (%d, %d), want (412, 1)", got.InputTokens, got.OutputTokens)
	}
}

// The reasoning effort is a property of the client and reaches every call it
// makes, this one included — and a model that reasons is exactly what
// [ErrNoDecisionToken] warns about, so the setting has to be able to turn it
// off here too.
func TestLogprobSendsTheConfiguredReasoningEffort(t *testing.T) {
	api := alwaysJSON(t, logprobFixture{chosen: " Yes", top: yesNo}.body())
	client, _ := newLogprobClient(t, api, func(cfg *Config) { cfg.ReasoningEffort = "none" })

	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got := api.request(t, 0).body(t)["reasoning_effort"]; got != "none" {
		t.Errorf("reasoning_effort = %v, want \"none\"", got)
	}
}

// wantDefaultScoringRequest is the whole document a default client sends for
// fixtureRequest, byte for byte, in the order the request struct declares its
// fields. The chat scorer is what every existing deployment is calibrated
// against, so adding a second scorer has to leave its request untouched: not
// a field reordered, and above all not a "logprobs":false appearing in it.
//
// The escapes are encoding/json's: the angle brackets of the instruction's
// <number 0 to 1> are written \u003c and \u003e, while the en dash in the
// closing question is emitted as itself.
const wantDefaultScoringRequest = `{"model":"request-model","messages":[{"role":"system","content":"You are a calibrated probability estimator. Given a STATE and a STATEMENT, return only how likely the statement is true based solely on the state. Do not invent facts. Respond with JSON: {\"p\": \u003cnumber 0 to 1\u003e}.\n\nSTATE:\nthe sky is grey"},{"role":"user","content":"STATEMENT:\nIt is raining.\n\nHow likely is the statement true (0–1)?"}],"temperature":0.25,"max_tokens":32,"response_format":{"type":"json_schema","json_schema":{"name":"prob","strict":true,"schema":{"type":"object","properties":{"p":{"type":"number","minimum":0,"maximum":1}},"required":["p"],"additionalProperties":false}}}}`

func TestTheDefaultScorerIsTheChatScorerAndItsRequestIsUnchanged(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.8}`))
	// No Scorer set: exactly the configuration every caller that has never
	// heard of the setting builds.
	client, _ := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if !closeTo(got.Probability, 0.8) {
		t.Errorf("probability = %v, want the 0.8 the model wrote: the default did not go to the chat scorer", got.Probability)
	}
	if body := string(api.request(t, 0).raw); body != wantDefaultScoringRequest {
		t.Errorf("the default scoring request changed\n got: %s\nwant: %s", body, wantDefaultScoringRequest)
	}
}

// "chat" set explicitly is the same client as "chat" by default.
func TestTheChatScorerNamedExplicitlySendsTheSameRequest(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.8}`))
	client, _ := newTestClient(t, api, func(cfg *Config) { cfg.Scorer = "chat" })

	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}
	if body := string(api.request(t, 0).raw); body != wantDefaultScoringRequest {
		t.Errorf("naming the default scorer changed the request\n got: %s\nwant: %s", body, wantDefaultScoringRequest)
	}
}

// A scorer this package does not know is a typo, and a typo that fell through
// to the default would answer every request with the estimator the operator
// had just decided not to use.
func TestAnUnknownScorerIsRejected(t *testing.T) {
	for _, name := range []string{"logits", "Logprob", "logprobs", "none"} {
		t.Run(name, func(t *testing.T) {
			_, err := New(Config{APIKey: testKey, BaseURL: testBaseURL, Scorer: name})
			if err == nil {
				t.Fatalf("New accepted the scorer %q", name)
			}
			for _, want := range []string{name, `"chat"`, `"logprob"`, "PERCEPTEA_SCORER"} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("the error does not mention %q: %v", want, err)
				}
			}
		})
	}
}

// Whitespace around a value read out of an environment is not a typo.
func TestAPaddedScorerNameIsAccepted(t *testing.T) {
	api := alwaysJSON(t, logprobFixture{chosen: " Yes", top: yesNo}.body())
	client, _ := newTestClient(t, api, func(cfg *Config) { cfg.Scorer = "  logprob\n" })

	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got := api.request(t, 0).body(t)["logprobs"]; got != true {
		t.Errorf("logprobs = %v: a padded %q did not select the logprob scorer", got, "logprob")
	}
}

// These errors send an operator to an environment variable, and one that
// names the wrong variable is worse than one that names none. This package
// spells the setting out rather than importing the server's configuration,
// because a provider client is usable on its own; this is the guard that
// keeps the two in step.
func TestTheLogprobErrorsNameTheSettingTheServerActuallyReads(t *testing.T) {
	if envScorer != config.EnvScorer {
		t.Errorf("the errors point at %q; the server reads %q", envScorer, config.EnvScorer)
	}
	if scorerChat != config.ScorerChat {
		t.Errorf("the errors offer %q; the server accepts %q", scorerChat, config.ScorerChat)
	}
	if scorerLogprob != config.ScorerLogprob {
		t.Errorf("this package selects on %q; the server sets %q", scorerLogprob, config.ScorerLogprob)
	}
	// The default has to agree too: Config.Scorer left empty must be the
	// scorer the server would have chosen for itself.
	if scorerChat != config.DefaultScorer {
		t.Errorf("an unset scorer is %q here; the server's default is %q", scorerChat, config.DefaultScorer)
	}
}

// Which token spellings count as an answer, and which deliberately do not.
func TestBranchOfClassifiesTheSpellingsAProviderSends(t *testing.T) {
	for _, tc := range []struct {
		token string
		want  branch
	}{
		{"Yes", branchYes},
		{" Yes", branchYes},
		{"yes", branchYes},
		{"YES", branchYes},
		{"▁Yes", branchYes},
		{"ĠYes", branchYes},
		{`"Yes`, branchYes},
		{"Yes,", branchYes},
		{"true", branchYes},
		{"TRUE", branchYes},
		{"No", branchNo},
		{" no", branchNo},
		{"NO", branchNo},
		{"▁No", branchNo},
		{"false", branchNo},
		// Not answers. A single letter is the tempting one and the trap: "n"
		// begins "no", but it also begins "not", "never" and "neither", and a
		// branch that can be wrong is worse than one that is sometimes
		// missing.
		{"y", branchNone},
		{"n", branchNone},
		{"Nope", branchNone},
		{"Yesterday", branchNone},
		{"Maybe", branchNone},
		{"", branchNone},
		{" ", branchNone},
	} {
		t.Run(tc.token, func(t *testing.T) {
			if got := branchOf(tc.token); got != tc.want {
				t.Errorf("branchOf(%q) = %v, want %v", tc.token, got, tc.want)
			}
		})
	}
}

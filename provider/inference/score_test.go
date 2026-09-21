package inference

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
)

// wantInstruction is the estimator instruction written out independently of
// the constant the client uses, so that an edit to one is caught by the other.
const wantInstruction = `You are a calibrated probability estimator. Given a STATE and a STATEMENT, return only how likely the statement is true based solely on the state. Do not invent facts. Respond with JSON: {"p": <number 0 to 1>}.`

// wantPrefix is the first message for the fixture request: the instruction and
// the state, and — because the fixture question declares no examples — nothing
// else. It carries nothing that identifies the candidate, which is what makes
// it the same bytes for every candidate of a question.
const wantPrefix = wantInstruction + "\n\nSTATE:\nthe sky is grey"

// wantSuffix is the second message for the fixture statement: the one thing
// that changes from candidate to candidate. The dash before the closing
// parenthesis is an en dash.
const wantSuffix = "STATEMENT:\nIt is raining.\n\nHow likely is the statement true (0–1)?"

// fixtureRequest is the score request the prompt constants above describe.
var fixtureRequest = classifier.ScoreRequest{
	Model:       "request-model",
	State:       "the sky is grey",
	Statement:   "It is raining.",
	Temperature: 0.25,
}

func TestScoreSendsTheExactLevelOneRequest(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.8}`))
	client, _ := newTestClient(t, api, func(cfg *Config) {
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
	if got := req.header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", got)
	}
	if got := req.header.Get("HTTP-Referer"); got != "https://perceptea.example" {
		t.Errorf("HTTP-Referer = %q", got)
	}
	if got := req.header.Get("X-Title"); got != "Perceptea" {
		t.Errorf("X-Title = %q", got)
	}

	var body struct {
		Model       string  `json:"model"`
		Temperature float64 `json:"temperature"`
		MaxTokens   int     `json:"max_tokens"`
		Messages    []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		ResponseFormat struct {
			Type       string `json:"type"`
			JSONSchema struct {
				Name   string          `json:"name"`
				Strict bool            `json:"strict"`
				Schema json.RawMessage `json:"schema"`
			} `json:"json_schema"`
		} `json:"response_format"`
	}
	if err := json.Unmarshal(req.raw, &body); err != nil {
		t.Fatalf("decode request body: %v", err)
	}

	if body.Model != "request-model" {
		t.Errorf("model = %q, want the request's model", body.Model)
	}
	if body.Temperature != 0.25 {
		t.Errorf("temperature = %v, want 0.25", body.Temperature)
	}
	if body.MaxTokens != 32 {
		t.Errorf("max_tokens = %d, want 32", body.MaxTokens)
	}
	if len(body.Messages) != 2 {
		t.Fatalf("len(messages) = %d, want 2", len(body.Messages))
	}
	if body.Messages[0].Role != "system" {
		t.Errorf("messages[0].role = %q, want system", body.Messages[0].Role)
	}
	if body.Messages[0].Content != wantPrefix {
		t.Errorf("prefix message mismatch\n got: %q\nwant: %q", body.Messages[0].Content, wantPrefix)
	}
	if body.Messages[1].Role != "user" {
		t.Errorf("messages[1].role = %q, want user", body.Messages[1].Role)
	}
	if body.Messages[1].Content != wantSuffix {
		t.Errorf("suffix message mismatch\n got: %q\nwant: %q", body.Messages[1].Content, wantSuffix)
	}
	if body.ResponseFormat.Type != "json_schema" {
		t.Errorf("response_format.type = %q, want json_schema", body.ResponseFormat.Type)
	}
	if body.ResponseFormat.JSONSchema.Name != "prob" {
		t.Errorf("json_schema.name = %q, want prob", body.ResponseFormat.JSONSchema.Name)
	}
	if !body.ResponseFormat.JSONSchema.Strict {
		t.Error("json_schema.strict = false, want true")
	}

	var schema, want any
	if err := json.Unmarshal(body.ResponseFormat.JSONSchema.Schema, &schema); err != nil {
		t.Fatalf("decode schema: %v", err)
	}
	const wantSchema = `{"type":"object","properties":{"p":{"type":"number","minimum":0,"maximum":1}},"required":["p"],"additionalProperties":false}`
	if err := json.Unmarshal([]byte(wantSchema), &want); err != nil {
		t.Fatalf("decode wanted schema: %v", err)
	}
	if !reflect.DeepEqual(schema, want) {
		t.Errorf("schema mismatch\n got: %v\nwant: %v", schema, want)
	}
}

// fixtureExamples stands in for the block the classifier renders for a
// question that declares worked examples. Its wording is pinned where it is
// built, in TestExamplesBlock; what these tests care about is where it lands.
const fixtureExamples = "EXAMPLES (worked answers for other states, as guidance; judge only the STATE below):" +
	"\n\nEXAMPLE 1 STATE:\nthe pump is stalled\nEXAMPLE 1 ANSWER: the correct option is \"technical\"."

// fixtureCandidates stands in for the block the classifier renders for a
// question with more than one candidate: the fixture statement and one rival.
// Its wording is pinned where it is built, in the classifier's
// TestCandidatesBlock; what these tests care about is where it lands.
const fixtureCandidates = "The candidates for this question, exactly one of which is correct:" +
	"\n- It is raining.\n- " + fixtureRival

// fixtureRival is the candidate in that list which is not the one being
// judged. It is what a test looks for to tell the shared list apart from the
// one statement this call is about: it belongs in the prefix and nowhere
// near the suffix.
const fixtureRival = "It is not raining."

// TestThePrefixIsByteIdenticalForEveryCandidateOfAQuestion is the assertion
// the two-message layout exists for. Every candidate answer of one question is
// judged against the same instruction, the same worked examples and the same
// state, and an endpoint only serves the repeats of that prompt prefix from
// its cache while the bytes match exactly.
//
// Anything candidate-shaped in the prefix — an index, a count, a reordering —
// defeats the caching silently. Every call still answers, every probability is
// still right, and nothing says a word about it except the bill; so the bytes
// are what is asserted, not the intent.
//
// This is the case where the question declares no candidate list, which is
// what lets it assert the strong form: no candidate's statement appears in
// the prefix at all. A question that does declare one names every statement
// there on purpose, and is checked by
// [TestThePrefixIsByteIdenticalWhenTheQuestionDeclaresACandidateList].
func TestThePrefixIsByteIdenticalForEveryCandidateOfAQuestion(t *testing.T) {
	const instructions = "Which team should handle this?"
	// Real statements for real candidates of one question, so that whatever a
	// candidate carries is genuinely present in the inputs and could leak.
	statements := []string{
		classifier.ChoiceStatement("department", instructions, "billing", "Charges, refunds, invoices"),
		classifier.ChoiceStatement("department", instructions, "technical", "Bugs"),
		classifier.ChoiceStatement("department", instructions, "sales", "Pricing and plans"),
		classifier.ChoiceStatement("department", instructions, "other", ""),
	}
	// Two candidates that rendered the same statement would make every
	// assertion below true for the wrong reason: the prefixes would match
	// because the calls were identical, not because the layout keeps them so.
	for i := range statements {
		for j := i + 1; j < len(statements); j++ {
			if statements[i] == statements[j] {
				t.Fatalf("candidates %d and %d render the same statement %q; "+
					"this fixture cannot see a candidate leaking into the prefix", i, j, statements[i])
			}
		}
	}

	api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
	client, _ := newTestClient(t, api, nil)

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

// The state is paid for once per wave and the statement once per candidate, so
// each has to be on its own side of the cut.
func TestTheStateIsInThePrefixAndTheStatementInTheSuffix(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
	client, _ := newTestClient(t, api, nil)

	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}

	msgs := api.messages(t, 0)
	if !strings.Contains(msgs[0].Content, fixtureRequest.State) {
		t.Errorf("the state is not in the prefix: %q", msgs[0].Content)
	}
	if strings.Contains(msgs[0].Content, fixtureRequest.Statement) {
		t.Errorf("the statement is in the prefix, which every candidate shares: %q", msgs[0].Content)
	}
	if !strings.Contains(msgs[1].Content, fixtureRequest.Statement) {
		t.Errorf("the statement is not in the suffix: %q", msgs[1].Content)
	}
	if strings.Contains(msgs[1].Content, fixtureRequest.State) {
		t.Errorf("the state is in the per-candidate suffix, so every call re-sends it: %q", msgs[1].Content)
	}
}

// Examples belong in the shared prefix, between the instruction and the state.
// In the suffix they would be re-sent, and re-charged, once per candidate.
func TestExamplesGoInThePrefixAndNowhereElse(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
	client, _ := newTestClient(t, api, nil)

	req := fixtureRequest
	req.Examples = fixtureExamples
	if _, err := client.Score(context.Background(), req); err != nil {
		t.Fatalf("Score: %v", err)
	}

	msgs := api.messages(t, 0)
	want := wantInstruction + "\n\n" + fixtureExamples + "\n\nSTATE:\nthe sky is grey"
	if msgs[0].Content != want {
		t.Errorf("prefix mismatch\n got: %q\nwant: %q", msgs[0].Content, want)
	}
	if strings.Contains(strings.ToUpper(msgs[1].Content), "EXAMPLE") {
		t.Errorf("the examples reached the per-candidate suffix: %q", msgs[1].Content)
	}
}

// Optional means optional. A question that declares no examples produces the
// prompt it would have produced had they never existed, down to the byte.
func TestAQuestionWithoutExamplesLeavesNoTraceOfThem(t *testing.T) {
	for _, tc := range []struct{ name, examples string }{
		{"none declared", ""},
		{"a block that is only whitespace", "  \n  "},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
			client, _ := newTestClient(t, api, nil)

			req := fixtureRequest
			req.Examples = tc.examples
			if _, err := client.Score(context.Background(), req); err != nil {
				t.Fatalf("Score: %v", err)
			}

			msgs := api.messages(t, 0)
			if msgs[0].Content != wantPrefix {
				t.Errorf("prefix mismatch\n got: %q\nwant: %q", msgs[0].Content, wantPrefix)
			}
			if msgs[1].Content != wantSuffix {
				t.Errorf("suffix mismatch\n got: %q\nwant: %q", msgs[1].Content, wantSuffix)
			}
			for i, m := range msgs {
				if strings.Contains(strings.ToUpper(m.Content), "EXAMPLE") {
					t.Errorf("message %d mentions an example although the question declared none: %q", i, m.Content)
				}
			}
		})
	}
}

// The candidate list is identical for every candidate of a question, so it
// belongs in the shared prefix, between the examples and the state. In the
// suffix it would be re-sent, and re-charged, once per candidate — and being
// free is most of the argument for sending it at all.
func TestTheCandidateListGoesInThePrefixAndNowhereElse(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
	client, _ := newTestClient(t, api, nil)

	req := fixtureRequest
	req.Examples = fixtureExamples
	req.Candidates = fixtureCandidates
	if _, err := client.Score(context.Background(), req); err != nil {
		t.Fatalf("Score: %v", err)
	}

	msgs := api.messages(t, 0)
	want := wantInstruction + "\n\n" + fixtureExamples + "\n\n" + fixtureCandidates + "\n\nSTATE:\nthe sky is grey"
	if msgs[0].Content != want {
		t.Errorf("prefix mismatch\n got: %q\nwant: %q", msgs[0].Content, want)
	}
	if !strings.Contains(msgs[0].Content, fixtureRival) {
		t.Errorf("the rival candidate never reached the prefix, so the model has nothing to compare with: %q", msgs[0].Content)
	}
	if msgs[1].Content != wantSuffix {
		t.Errorf("the suffix carries this one statement and no part of the list\n got: %q\nwant: %q", msgs[1].Content, wantSuffix)
	}
	if strings.Contains(msgs[1].Content, fixtureRival) {
		t.Errorf("a rival candidate reached the per-candidate suffix, where it is paid for again on every call: %q", msgs[1].Content)
	}
}

// A noul has one candidate and a one-option choice has one, so neither has
// anything to choose among and neither renders a list. Such a question
// produces the prompt it produced before candidate lists existed, down to the
// byte.
func TestAQuestionWithoutACandidateListLeavesNoTraceOfIt(t *testing.T) {
	for _, tc := range []struct{ name, candidates string }{
		{"one candidate, so nothing rendered", ""},
		{"a list that is only whitespace", "  \n  "},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
			client, _ := newTestClient(t, api, nil)

			req := fixtureRequest
			req.Candidates = tc.candidates
			if _, err := client.Score(context.Background(), req); err != nil {
				t.Fatalf("Score: %v", err)
			}

			msgs := api.messages(t, 0)
			if msgs[0].Content != wantPrefix {
				t.Errorf("prefix mismatch\n got: %q\nwant: %q", msgs[0].Content, wantPrefix)
			}
			if msgs[1].Content != wantSuffix {
				t.Errorf("suffix mismatch\n got: %q\nwant: %q", msgs[1].Content, wantSuffix)
			}
			for i, m := range msgs {
				if strings.Contains(strings.ToLower(m.Content), "candidate") {
					t.Errorf("message %d speaks of candidates although the question has only one: %q", i, m.Content)
				}
			}
		})
	}
}

// The prefix holds the whole candidate list, which names every candidate
// including the one this call is judging — so the byte-identity is asserted
// here without the companion claim that no statement appears in the prefix.
// That claim is the point of the list and cannot hold beside it; what still
// has to hold is that the bytes do not move from call to call, and that the
// suffix stays one candidate's own.
func TestThePrefixIsByteIdenticalWhenTheQuestionDeclaresACandidateList(t *testing.T) {
	const instructions = "Which team should handle this?"
	statements := []string{
		classifier.ChoiceStatement("department", instructions, "billing", "Charges, refunds, invoices"),
		classifier.ChoiceStatement("department", instructions, "technical", "Bugs"),
		classifier.ChoiceStatement("department", instructions, "sales", "Pricing and plans"),
		classifier.ChoiceStatement("department", instructions, "other", ""),
	}
	// Two candidates that rendered the same statement would make the
	// per-candidate assertions below true for the wrong reason.
	for i := range statements {
		for j := i + 1; j < len(statements); j++ {
			if statements[i] == statements[j] {
				t.Fatalf("candidates %d and %d render the same statement %q; "+
					"this fixture cannot tell one candidate's prompt from another's", i, j, statements[i])
			}
		}
	}

	candidates := classifier.CandidatesBlock(statements)
	// A list that had dropped a candidate would satisfy every assertion
	// about the prefix carrying it.
	for i, statement := range statements {
		if !strings.Contains(candidates, statement) {
			t.Fatalf("the rendered list does not name candidate %d, so this fixture proves nothing about it:\n%q", i, candidates)
		}
	}

	api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
	client, _ := newTestClient(t, api, nil)

	for _, statement := range statements {
		req := fixtureRequest
		req.Statement = statement
		req.Examples = fixtureExamples
		req.Candidates = candidates
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
		if !strings.Contains(msgs[0].Content, candidates) {
			t.Errorf("candidate %d's prefix does not carry the list: %q", i, msgs[0].Content)
		}
		if !strings.Contains(msgs[1].Content, statement) {
			t.Errorf("candidate %d's statement is not in its suffix: %q", i, msgs[1].Content)
		}
		for j, other := range statements {
			if j != i && strings.Contains(msgs[1].Content, other) {
				t.Errorf("candidate %d's suffix names candidate %d too; the list is being repeated per call, not shared: %q",
					i, j, msgs[1].Content)
			}
		}
		if i > 0 && msgs[1].Content == first[1].Content {
			t.Errorf("candidate %d sent candidate 0's suffix; the candidate is not reaching the model", i)
		}
	}
}

// The layout moved; the contract did not. However the prompt is cut up, the
// model is still told to answer with {"p": <number 0 to 1>}, still held to it
// by the schema, and the number still comes back.
func TestTheProbabilityContractStillReachesTheModel(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.77}`))
	client, _ := newTestClient(t, api, nil)

	req := fixtureRequest
	req.Examples = fixtureExamples
	got, err := client.Score(context.Background(), req)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if math.Abs(got.Probability-0.77) > 1e-9 {
		t.Errorf("probability = %v, want 0.77", got.Probability)
	}

	const contract = `Respond with JSON: {"p": <number 0 to 1>}.`
	prompt := strings.Join([]string{api.messages(t, 0)[0].Content, api.messages(t, 0)[1].Content}, "\n")
	if !strings.Contains(prompt, contract) {
		t.Errorf("the prompt no longer asks for %s\nprompt: %q", contract, prompt)
	}

	var body struct {
		ResponseFormat struct {
			Type       string `json:"type"`
			JSONSchema struct {
				Schema struct {
					Required   []string       `json:"required"`
					Properties map[string]any `json:"properties"`
				} `json:"schema"`
			} `json:"json_schema"`
		} `json:"response_format"`
	}
	if err := json.Unmarshal(api.request(t, 0).raw, &body); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	if body.ResponseFormat.Type != "json_schema" {
		t.Fatalf("response_format.type = %q, want json_schema", body.ResponseFormat.Type)
	}
	schema := body.ResponseFormat.JSONSchema.Schema
	if len(schema.Required) != 1 || schema.Required[0] != "p" {
		t.Errorf("schema requires %v, want exactly [p]", schema.Required)
	}
	if _, ok := schema.Properties["p"]; !ok {
		t.Errorf("schema declares no \"p\" property: %v", schema.Properties)
	}
}

func TestScoreTemperatureZeroIsSent(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
	client, _ := newTestClient(t, api, nil)

	if _, err := client.Score(context.Background(), classifier.ScoreRequest{Statement: "x"}); err != nil {
		t.Fatalf("Score: %v", err)
	}
	body := api.request(t, 0).body(t)
	temp, ok := body["temperature"]
	if !ok {
		t.Fatal("temperature is absent from the request; 0 must be sent explicitly")
	}
	if temp != 0.0 {
		t.Errorf("temperature = %v, want 0", temp)
	}
}

func TestScoreFallsBackToConfiguredModel(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.1}`))
	client, _ := newTestClient(t, api, nil)

	if _, err := client.Score(context.Background(), classifier.ScoreRequest{Statement: "x"}); err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got := api.request(t, 0).body(t)["model"]; got != "config-model" {
		t.Errorf("model = %v, want config-model", got)
	}
}

func TestScoreWithoutAnyModelFails(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.1}`))
	client, _ := newTestClient(t, api, func(cfg *Config) { cfg.Model = "" })

	if _, err := client.Score(context.Background(), classifier.ScoreRequest{Statement: "x"}); err == nil {
		t.Fatal("want an error when neither the config nor the request names a model")
	}
	if api.count() != 0 {
		t.Errorf("sent %d requests, want none", api.count())
	}
}

func TestScoreNegotiatesDownThroughEveryLevel(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, r *http.Request, n int) {
		switch n {
		case 0, 1:
			writeJSON(w, http.StatusBadRequest,
				`{"error":{"message":"response_format is not supported","type":"invalid_request_error"}}`)
		default:
			writeJSON(w, http.StatusOK, scoreBody(`{"p":0.6}`))
		}
	})
	client, _ := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got.Probability != 0.6 {
		t.Errorf("probability = %v, want 0.6", got.Probability)
	}
	if api.count() != 3 {
		t.Fatalf("made %d calls, want 3 (one per level)", api.count())
	}
	if f := api.request(t, 0).format(t); f != "json_schema" {
		t.Errorf("call 1 response_format = %q, want json_schema", f)
	}
	if f := api.request(t, 1).format(t); f != "json_object" {
		t.Errorf("call 2 response_format = %q, want json_object", f)
	}
	if f := api.request(t, 2).format(t); f != "" {
		t.Errorf("call 3 response_format = %q, want none", f)
	}
	if _, present := api.request(t, 2).body(t)["response_format"]; present {
		t.Error("call 3 still carries a response_format field")
	}
}

func TestScoreRemembersTheDowngrade(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, r *http.Request, n int) {
		var body struct {
			ResponseFormat *struct {
				Type string `json:"type"`
			} `json:"response_format"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body.ResponseFormat != nil && body.ResponseFormat.Type == "json_schema" {
			writeJSON(w, http.StatusBadRequest, `{"error":{"message":"json_schema unsupported"}}`)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.4}`))
	})
	client, _ := newTestClient(t, api, nil)

	for i := range 3 {
		if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
			t.Fatalf("Score %d: %v", i, err)
		}
	}

	// One rejected probe on the first call, then one call each. A client
	// that forgot the downgrade would make six.
	if api.count() != 4 {
		t.Fatalf("made %d calls, want 4 (the level-one probe is paid once)", api.count())
	}
	for i := 1; i < 4; i++ {
		if f := api.request(t, i).format(t); f != "json_object" {
			t.Errorf("call %d response_format = %q, want json_object", i+1, f)
		}
	}
	if lvl := client.outputLevel(); lvl != levelJSONObject {
		t.Errorf("level = %v, want json_object: json_object answered, so the step is real", lvl)
	}
}

// TestScoreOnlyRemembersALevelThatAnswered is the other half of the rule. A
// request that fails at every level says nothing about which shapes the
// provider supports — it was failing for some other reason — so the client
// must come back to the next request at the level it started from. Without
// this, one bad model name or one over-long prompt permanently strips
// structured output from every later request the client serves, and a client
// cached across requests carries that to everybody.
// A provider that cannot honour a schema does not have to say so with a 400.
// DeepInfra answers "json_schema response format is not supported for model X"
// with a 405, and an allowlist of statuses that were thought to mean "bad
// field" let that fail outright instead of stepping down to a format the model
// does accept.
func TestScoreStepsDownWhenAnUnsupportedFormatIsRejectedWithA405(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, r *http.Request, n int) {
		if n == 0 {
			writeJSON(w, http.StatusMethodNotAllowed,
				`{"error":{"message":"json_schema response format is not supported for model: some/model"}}`)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.42}`))
	})
	client, _ := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got.Probability != 0.42 {
		t.Errorf("probability = %v, want 0.42", got.Probability)
	}
	if api.count() != 2 {
		t.Fatalf("made %d calls, want 2: the schema probe and the step down", api.count())
	}
	if f := api.request(t, 1).format(t); f != "json_object" {
		t.Errorf("the second call sent response_format %q, want json_object", f)
	}
	if lvl := client.outputLevel(); lvl != levelJSONObject {
		t.Errorf("level = %v, want json_object remembered: it is the one that answered", lvl)
	}
}

// A provider may wrap its own upstream's rejection in a 500 and still name
// the field it could not honour. Observed verbatim from a hosted endpoint:
// a 500 whose body carries a <400> and "Format error :
// 'response_format.json_schema.schema'". A 500 is normally retried and never
// probes, so without reading the message this cost three retries and then
// failed, with a format the model would have accepted one step below.
func TestScoreStepsDownWhenAFormatComplaintArrivesAsA500(t *testing.T) {
	const complaint = `{"error":{"message":"<400> InternalError.Algo.InvalidParameter: Format error : ` +
		`'response_format.json_schema.schema'. the specific reason is as follows: None is not of type ` +
		`'object', 'boolean'.","type":"invalid_request_error","code":"invalid_parameter_error"}}`

	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
		if n == 0 {
			writeJSON(w, http.StatusInternalServerError, complaint)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.61}`))
	})
	client, sleeps := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got.Probability != 0.61 {
		t.Errorf("probability = %v, want 0.61", got.Probability)
	}
	if api.count() != 2 {
		t.Fatalf("made %d calls, want 2: the schema attempt and the step down", api.count())
	}
	if n := len(sleeps.durations()); n != 0 {
		t.Errorf("backed off %d times, want 0: naming the field is not a transient failure", n)
	}
	if f := api.request(t, 1).format(t); f != "json_object" {
		t.Errorf("the second call sent response_format %q, want json_object", f)
	}
	if lvl := client.outputLevel(); lvl != levelJSONObject {
		t.Errorf("level = %v, want json_object remembered", lvl)
	}
}

// The converse: a 500 that says nothing about the request is still a
// transient failure, retried and never probed.
func TestScoreStillRetriesAPlainServerError(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
		if n < 2 {
			writeJSON(w, http.StatusInternalServerError, `{"error":{"message":"internal error"}}`)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.2}`))
	})
	client, sleeps := newTestClient(t, api, nil)

	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}
	if n := len(sleeps.durations()); n != 2 {
		t.Errorf("backed off %d times, want 2", n)
	}
	if lvl := client.outputLevel(); lvl != levelJSONSchema {
		t.Errorf("level = %v, want json_schema: a plain 500 says nothing about the format", lvl)
	}
	for i := range 3 {
		if f := api.request(t, i).format(t); f != "json_schema" {
			t.Errorf("call %d sent response_format %q, want json_schema throughout", i, f)
		}
	}
}

func TestScoreOnlyRemembersALevelThatAnswered(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
		if n < 3 {
			writeJSON(w, http.StatusBadRequest,
				`{"error":{"message":"The model does not exist","type":"invalid_request_error","code":"model_not_found"}}`)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.3}`))
	})
	client, _ := newTestClient(t, api, nil)

	if _, err := client.Score(context.Background(), fixtureRequest); err == nil {
		t.Fatal("want the underlying error, not a probability")
	}
	if api.count() != 3 {
		t.Fatalf("made %d calls, want 3 (one probe per level)", api.count())
	}
	if lvl := client.outputLevel(); lvl != levelJSONSchema {
		t.Fatalf("level = %v, want json_schema: nothing answered, so nothing was learned", lvl)
	}

	// The next call, against a healthy endpoint, still asks for a schema.
	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got.Probability != 0.3 {
		t.Errorf("probability = %v, want 0.3", got.Probability)
	}
	if f := api.request(t, 3).format(t); f != "json_schema" {
		t.Errorf("the recovered call sent response_format %q, want json_schema", f)
	}
}

// TestScoreDoesNotProbeOnAFailureUnrelatedToTheRequestShape pins the other
// half of the same finding: these statuses are not a provider rejecting a
// field, so stepping down the levels only multiplies the failure.
func TestScoreDoesNotProbeOnAFailureUnrelatedToTheRequestShape(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
	}{
		// Only these four. Anything else a provider might answer could be its
		// way of saying it cannot honour a field, so the client probes rather
		// than guessing — probing is cheap, because a level is only
		// remembered once it has answered.
		// A timeout and a rate limit are the other two, but those are
		// retried, so they cannot be counted this way;
		// TestUnsupportedShapeClassification covers all four.
		{"a rejected key", http.StatusUnauthorized},
		{"a forbidden key", http.StatusForbidden},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
				writeJSON(w, tc.status, `{"error":{"message":"no"}}`)
			})
			client, _ := newTestClient(t, api, nil)

			if _, err := client.Score(context.Background(), fixtureRequest); err == nil {
				t.Fatalf("status %d: want an error", tc.status)
			}
			if api.count() != 1 {
				t.Errorf("status %d: made %d calls, want 1", tc.status, api.count())
			}
			if lvl := client.outputLevel(); lvl != levelJSONSchema {
				t.Errorf("status %d: level = %v, want json_schema", tc.status, lvl)
			}
		})
	}
}

// TestScoreProbesOnANotImplemented covers the one 5xx that does mean the shape
// is unsupported. It must not be retried on the way down, or the negotiation
// costs the retry count at every level.
func TestScoreProbesOnANotImplemented(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
		if n == 0 {
			writeJSON(w, http.StatusNotImplemented, `{"error":{"message":"json_schema is not implemented"}}`)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.2}`))
	})
	client, sleeps := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got.Probability != 0.2 {
		t.Errorf("probability = %v, want 0.2", got.Probability)
	}
	if api.count() != 2 {
		t.Fatalf("made %d calls, want 2", api.count())
	}
	if n := len(sleeps.durations()); n != 0 {
		t.Errorf("backed off %d times, want 0: a 501 is not transient", n)
	}
	if lvl := client.outputLevel(); lvl != levelJSONObject {
		t.Errorf("level = %v, want json_object", lvl)
	}
}

// TestDowngradeNeverRaisesTheLevel guards the comparison in downgrade. Two
// Score calls run concurrently against a provider that rejects json_schema:
// both discover the rejection, and the one that reaches plain can finish
// after the one that settled on json_object. Without the guard the later,
// staler report would raise the level back to json_object and the next call
// would send a response_format the provider has already refused.
func TestDowngradeNeverRaisesTheLevel(t *testing.T) {
	client, _ := newTestClient(t, alwaysJSON(t, scoreBody(`{"p":0.5}`)), nil)

	client.downgrade(levelPlain)
	client.downgrade(levelJSONObject)
	if lvl := client.outputLevel(); lvl != levelPlain {
		t.Errorf("level = %v after a stale json_object report, want plain", lvl)
	}
	client.downgrade(levelJSONSchema)
	if lvl := client.outputLevel(); lvl != levelPlain {
		t.Errorf("level = %v after a stale json_schema report, want plain", lvl)
	}
	// And the level it is already at is not a raise either.
	client.downgrade(levelPlain)
	if lvl := client.outputLevel(); lvl != levelPlain {
		t.Errorf("level = %v, want plain", lvl)
	}
}

func TestScoreDoesNotDowngradeOnRateLimitOrServerError(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
	}{
		{"rate limited", http.StatusTooManyRequests},
		{"server error", http.StatusInternalServerError},
		{"gateway error", http.StatusBadGateway},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
				if n == 0 {
					writeJSON(w, tc.status, `{"error":{"message":"later"}}`)
					return
				}
				writeJSON(w, http.StatusOK, scoreBody(`{"p":0.9}`))
			})
			client, sleeps := newTestClient(t, api, nil)

			got, err := client.Score(context.Background(), fixtureRequest)
			if err != nil {
				t.Fatalf("Score: %v", err)
			}
			if got.Probability != 0.9 {
				t.Errorf("probability = %v, want 0.9", got.Probability)
			}
			if api.count() != 2 {
				t.Fatalf("made %d calls, want 2", api.count())
			}
			if len(sleeps.durations()) != 1 {
				t.Errorf("backed off %d times, want 1", len(sleeps.durations()))
			}
			for i := range 2 {
				if f := api.request(t, i).format(t); f != "json_schema" {
					t.Errorf("call %d response_format = %q; a %d must not downgrade", i+1, f, tc.status)
				}
			}
			if lvl := client.outputLevel(); lvl != levelJSONSchema {
				t.Errorf("level = %v, want json_schema", lvl)
			}
		})
	}
}

func TestScoreDoesNotDowngradeOnAuthFailure(t *testing.T) {
	for _, status := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
			writeJSON(w, status, `{"error":{"message":"nope"}}`)
		})
		client, _ := newTestClient(t, api, nil)

		if _, err := client.Score(context.Background(), fixtureRequest); err == nil {
			t.Fatalf("status %d: want an error", status)
		}
		if api.count() != 1 {
			t.Errorf("status %d: made %d calls, want 1", status, api.count())
		}
		if lvl := client.outputLevel(); lvl != levelJSONSchema {
			t.Errorf("status %d: level = %v, want json_schema", status, lvl)
		}
	}
}

func TestScoreParsesTheReply(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content string
		want    float64
	}{
		{"p field", `{"p":0.73}`, 0.73},
		{"p field as a string", `{"p":"0.73"}`, 0.73},
		{"probability field", `{"probability":0.42}`, 0.42},
		{"p wins over probability", `{"p":0.2,"probability":0.9}`, 0.2},
		{"p null falls through", `{"p":null,"probability":0.31}`, 0.31},
		{"fenced json", "```json\n{\"p\": 0.64}\n```", 0.64},
		{"fenced json without a tag", "```\n{\"p\": 0.64}\n```", 0.64},
		{"fenced json on one line", "```{\"p\":0.64}```", 0.64},
		// The regexp salvage would read the first number in the raw
		// text, 0.9. Only genuinely unwrapping the fence and parsing
		// the object gives p.
		{"fenced json the regexp would misread", "```json\n{\"probability\": 0.9, \"p\": 0.1}\n```", 0.1},
		{"prose with a decimal", "I would say roughly 0.35 given the state.", 0.35},
		{"prose with a leading dot", "about .87 likely", 0.87},
		{"prose with a bare one", "Certainly: 1", 1},
		{"prose with a bare zero", "Definitely not: 0", 0},
		{"prose with no number", "It is impossible to tell from the state.", 0.5},
		{"empty content", "", 0.5},
		{"clamped above", `{"p":1.7}`, 1},
		{"clamped below", `{"p":-0.4}`, 0},
		{"json object with no probability", `{"answer":"yes"}`, 0.5},
		{"bare json number", `0.28`, 0.28},
		{"json string", `"unsure"`, 0.5},
		{"non-finite string", `{"p":"NaN"}`, 0.5},
		// "NaN" above reaches 0.5 down two paths at once — finite's guard and
		// clamp01's own NaN branch — so it cannot see finite on its own.
		// "Infinity" can: without the guard it is +Inf, which clamp01 reads as
		// a perfectly confident 1.
		{"infinite string", `{"p":"Infinity"}`, 0.5},
		{"negatively infinite string", `{"p":"-Infinity"}`, 0.5},
		{"overflowing number", `{"p":1e400}`, 0.5},
		{"boolean true", `{"p":true}`, 1},
		// A number under a key is a probability the model got wrong, so it
		// is clamped. A bare number outside [0,1] is not a probability at
		// all, and is not read as maximum confidence.
		{"out of range under a key is clamped", `{"p":12}`, 1},
		{"bare number above one", `12`, 0.5},
		{"bare percentage", `85`, 0.5},
		{"bare hundred", `100`, 0.5},
		{"bare negative number", `-0.4`, 0.5},
		{"bare number just above one", `1.7`, 0.5},
		{"bare one is still a probability", `1`, 1},
		{"bare zero is still a probability", `0`, 0},
		{"bare numeric string out of range", `"12"`, 0.5},
		// Reading an LLM's reply generously without inventing confidence it
		// never carried: a shouted key is still the key, while a field the
		// model left blank is no answer rather than a confident zero.
		{"a shouted key", `{"P":0.3}`, 0.3},
		{"a shouted long key", `{"Probability":0.3}`, 0.3},
		{"an empty string is not a zero", `{"p":""}`, 0.5},
		{"a blank string is not a zero", `{"p":" "}`, 0.5},
		{"an empty array is not a zero", `{"p":[]}`, 0.5},
		{"an object is not a zero", `{"p":{}}`, 0.5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := alwaysJSON(t, scoreBody(tc.content))
			client, _ := newTestClient(t, api, nil)

			got, err := client.Score(context.Background(), fixtureRequest)
			if err != nil {
				t.Fatalf("Score: %v", err)
			}
			if math.Abs(got.Probability-tc.want) > 1e-9 {
				t.Errorf("probability = %v, want %v (content %q)", got.Probability, tc.want, tc.content)
			}
			if got.Probability < 0 || got.Probability > 1 {
				t.Errorf("probability %v escaped [0,1]", got.Probability)
			}
		})
	}
}

func TestScoreHandlesAnEmptyChoicesArray(t *testing.T) {
	api := alwaysJSON(t, `{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":0}}`)
	client, _ := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v, want no error for an empty choices array", err)
	}
	if got.Probability != 0.5 {
		t.Errorf("probability = %v, want 0.5", got.Probability)
	}
	if got.InputTokens != 11 {
		t.Errorf("input tokens = %d, want 11", got.InputTokens)
	}
}

// A scoring call caps the reply at 32 tokens, which is room enough for
// {"p":0.87} and none at all for a model that thinks first. Spent on thinking
// tokens, the budget runs out before the JSON — and because the cap is the
// same on every call, every candidate of the request comes back unreadable,
// every score is the 0.5 fallback, and the softmax renders a uniform set of
// scores as a confident-looking answer carrying no information. Failing the
// call is the only way that does not look like a result.
func TestScoreRejectsATruncatedReplyWithNoProbability(t *testing.T) {
	for _, tc := range []struct {
		name  string
		reply replyFixture
	}{
		{
			"nothing but thinking, in the reasoning field",
			replyFixture{reasoning: thinkingOutLoud, finishReason: "length"},
		},
		{
			"nothing but thinking, under the other field name",
			replyFixture{reasoningContent: thinkingOutLoud, finishReason: "length"},
		},
		{
			"prose cut off before any number",
			replyFixture{content: "Based on the state provided, I would estimate the probability to be", finishReason: "length"},
		},
		{
			"the JSON cut off before the value",
			replyFixture{content: `{"p":`, finishReason: "length"},
		},
		{
			"an empty reply with nothing anywhere",
			replyFixture{finishReason: "length"},
		},
		{
			"a finish reason the provider shouted",
			replyFixture{reasoning: thinkingOutLoud, finishReason: "LENGTH"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := alwaysJSON(t, tc.reply.body())
			client, _ := newTestClient(t, api, nil)

			got, err := client.Score(context.Background(), fixtureRequest)
			if err == nil {
				t.Fatalf("Score returned %v and no error; a truncated reply must not be scored as a probability", got)
			}
			if !errors.Is(err, ErrTruncatedReply) {
				t.Fatalf("Score returned %v, want ErrTruncatedReply", err)
			}
			if got.Probability == fallbackProbability {
				t.Error("the fallback probability came back alongside the error")
			}

			// The message is what an operator gets in a 502, so it has to
			// carry the finding, the cause and both ways out.
			msg := err.Error()
			for _, want := range []string{
				"cut off", "output token limit", "32", "thinking tokens",
				"does not reason", "PERCEPTEA_REASONING_EFFORT", `"none"`,
			} {
				if !strings.Contains(msg, want) {
					t.Errorf("the error does not mention %q: %s", want, msg)
				}
			}
		})
	}
}

// The truncation message names a setting, and a message that names the wrong
// setting is worse than one that names none: it sends the operator to an
// environment variable nothing reads. This package spells the name out rather
// than importing the server's configuration, because a provider client is
// usable on its own; this is the guard that keeps the two in step.
func TestTheTruncationMessageNamesTheSettingTheServerActuallyReads(t *testing.T) {
	if envReasoningEffort != config.EnvReasoningEffort {
		t.Errorf("the error points at %q; the server reads %q", envReasoningEffort, config.EnvReasoningEffort)
	}
	if effortNone != config.EffortNone {
		t.Errorf("the error suggests %q; the server accepts %q", effortNone, config.EffortNone)
	}
}

// An empty content field beside a populated reasoning field is the signature
// of this failure and nothing else, so the message says so outright rather
// than leaving the reader to guess which of the two causes they have.
func TestATruncatedReplyNamesTheEmptyContentBesideTheReasoning(t *testing.T) {
	api := alwaysJSON(t, replyFixture{reasoning: thinkingOutLoud, finishReason: "length"}.body())
	client, _ := newTestClient(t, api, nil)

	_, err := client.Score(context.Background(), fixtureRequest)
	if err == nil {
		t.Fatal("want an error")
	}
	for _, want := range []string{"content was empty", "reasoning field"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error does not mention %q: %s", want, err)
		}
	}

	// And it is not claimed of a reply that was simply cut short mid-answer:
	// there the cap is still the fault, but the reasoning field is not the
	// evidence for it.
	api = alwaysJSON(t, replyFixture{content: "I would estimate the probability to be", finishReason: "length"}.body())
	client, _ = newTestClient(t, api, nil)

	_, err = client.Score(context.Background(), fixtureRequest)
	if err == nil {
		t.Fatal("want an error")
	}
	if strings.Contains(err.Error(), "content was empty") {
		t.Errorf("a reply that did have content is described as having none: %s", err)
	}
}

// The rule is about a truncated reply that said nothing, not about
// truncation. A reply that got its number out before the cap stopped it
// answered the question.
func TestScoreAcceptsATruncatedReplyThatStillCarriedAProbability(t *testing.T) {
	for _, tc := range []struct {
		name  string
		reply replyFixture
		want  float64
	}{
		{"the JSON landed, the closing prose did not", replyFixture{content: `{"p":0.87}`, finishReason: "length"}, 0.87},
		{"a number in prose, then the cap", replyFixture{content: "I would say 0.42 based on", finishReason: "length"}, 0.42},
		{"the answer came out of the reasoning field", replyFixture{reasoning: `{"p":0.31}`, finishReason: "length"}, 0.31},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := alwaysJSON(t, tc.reply.body())
			client, _ := newTestClient(t, api, nil)

			got, err := client.Score(context.Background(), fixtureRequest)
			if err != nil {
				t.Fatalf("Score: %v, want the probability the reply did carry", err)
			}
			if math.Abs(got.Probability-tc.want) > 1e-9 {
				t.Errorf("probability = %v, want %v", got.Probability, tc.want)
			}
		})
	}
}

// Everywhere else the forgiveness stands. A reply that finished normally and
// still said nothing readable is a one-off: one candidate, one blunted score,
// no error. Making the parser stricter than the truncation case would fail a
// whole evaluation over one model's odd sentence.
func TestScoreStillForgivesAnUnreadableReplyThatFinishedNormally(t *testing.T) {
	for _, tc := range []struct {
		name  string
		reply replyFixture
	}{
		{"stopped normally with no number", replyFixture{content: "It is impossible to tell from the state.", finishReason: "stop"}},
		{"stopped normally with nothing at all", replyFixture{finishReason: "stop"}},
		{"no finish reason reported", replyFixture{content: "It is impossible to tell from the state."}},
		{"stopped by a content filter", replyFixture{finishReason: "content_filter"}},
		{"stopped for a tool call", replyFixture{finishReason: "tool_calls"}},
		{"a JSON document with no probability in it", replyFixture{content: `{"answer":"yes"}`, finishReason: "stop"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			api := alwaysJSON(t, tc.reply.body())
			client, _ := newTestClient(t, api, nil)

			got, err := client.Score(context.Background(), fixtureRequest)
			if err != nil {
				t.Fatalf("Score: %v, want the 0.5 fallback rather than an error", err)
			}
			if got.Probability != fallbackProbability {
				t.Errorf("probability = %v, want %v", got.Probability, fallbackProbability)
			}
		})
	}
}

// An empty choices list has no finish reason to read, so it keeps the
// fallback it always had.
func TestScoreForgivesAnEmptyChoicesListWhateverElseIsInTheDocument(t *testing.T) {
	api := alwaysJSON(t, `{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":32}}`)
	client, _ := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got.Probability != fallbackProbability {
		t.Errorf("probability = %v, want %v", got.Probability, fallbackProbability)
	}
}

// Unset is the default, and the default is the behaviour this client had
// before the setting existed: no reasoning field on the wire at all, so an
// endpoint that has never seen one is sent exactly what it was sent before.
func TestScoreSendsNoReasoningEffortUnlessOneIsConfigured(t *testing.T) {
	for _, configured := range []string{"", "   "} {
		api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
		client, _ := newTestClient(t, api, func(cfg *Config) { cfg.ReasoningEffort = configured })

		if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
			t.Fatalf("Score: %v", err)
		}
		if _, present := api.request(t, 0).body(t)["reasoning_effort"]; present {
			t.Errorf("reasoning_effort was sent although none is configured (%q)", configured)
		}
	}
}

func TestScoreSendsTheConfiguredReasoningEffort(t *testing.T) {
	for _, effort := range []string{"none", "low", "medium", "high"} {
		t.Run(effort, func(t *testing.T) {
			api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
			client, _ := newTestClient(t, api, func(cfg *Config) { cfg.ReasoningEffort = effort })

			if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
				t.Fatalf("Score: %v", err)
			}
			if got := api.request(t, 0).body(t)["reasoning_effort"]; got != effort {
				t.Errorf("reasoning_effort = %v, want %q", got, effort)
			}
		})
	}
}

// The effort survives the structured-output negotiation: a client that
// discovers json_schema is unsupported must not drop the one field that keeps
// the reply short enough to read.
func TestTheReasoningEffortSurvivesADowngrade(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
		if n == 0 {
			writeJSON(w, http.StatusBadRequest, `{"error":{"message":"json_schema unsupported"}}`)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.5}`))
	})
	client, _ := newTestClient(t, api, func(cfg *Config) { cfg.ReasoningEffort = "none" })

	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}
	if api.count() != 2 {
		t.Fatalf("made %d calls, want 2", api.count())
	}
	if got := api.request(t, 1).body(t)["reasoning_effort"]; got != "none" {
		t.Errorf("the downgraded call sent reasoning_effort %v, want %q", got, "none")
	}
}

func TestScoreFallsBackToTheReasoningField(t *testing.T) {
	api := alwaysJSON(t, `{"choices":[{"message":{"content":null,"reasoning":"weighing it up, p = 0.66"}}]}`)
	client, _ := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if math.Abs(got.Probability-0.66) > 1e-9 {
		t.Errorf("probability = %v, want 0.66", got.Probability)
	}
}

// The other name the same field goes by.
func TestScoreFallsBackToTheReasoningContentField(t *testing.T) {
	api := alwaysJSON(t, replyFixture{reasoningContent: "weighing it up, p = 0.66", finishReason: "stop"}.body())
	client, _ := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if math.Abs(got.Probability-0.66) > 1e-9 {
		t.Errorf("probability = %v, want 0.66", got.Probability)
	}
}

func TestScoreReadsUsage(t *testing.T) {
	api := alwaysJSON(t, `{"choices":[{"message":{"content":"{\"p\":0.5}"}}],"usage":{"prompt_tokens":123,"completion_tokens":7}}`)
	client, _ := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got.InputTokens != 123 || got.OutputTokens != 7 {
		t.Errorf("usage = (%d, %d), want (123, 7)", got.InputTokens, got.OutputTokens)
	}
}

func TestScoreTreatsMissingUsageAsZero(t *testing.T) {
	api := alwaysJSON(t, scoreBody(`{"p":0.5}`))
	client, _ := newTestClient(t, api, nil)

	got, err := client.Score(context.Background(), fixtureRequest)
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if got.InputTokens != 0 || got.OutputTokens != 0 {
		t.Errorf("usage = (%d, %d), want (0, 0)", got.InputTokens, got.OutputTokens)
	}
}

func TestScoreIsSafeForConcurrentUse(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, r *http.Request, n int) {
		var body struct {
			ResponseFormat *struct {
				Type string `json:"type"`
			} `json:"response_format"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body.ResponseFormat != nil && body.ResponseFormat.Type == "json_schema" {
			writeJSON(w, http.StatusBadRequest, `{"error":{"message":"json_schema unsupported"}}`)
			return
		}
		writeJSON(w, http.StatusOK, scoreBody(`{"p":0.55}`))
	})
	client, _ := newTestClient(t, api, nil)

	const workers = 32
	var wg sync.WaitGroup
	errs := make([]error, workers)
	results := make([]classifier.ScoreResult, workers)
	for i := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i], errs[i] = client.Score(context.Background(), fixtureRequest)
		}()
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("worker %d: %v", i, err)
		}
		if math.Abs(results[i].Probability-0.55) > 1e-9 {
			t.Errorf("worker %d: probability = %v, want 0.55", i, results[i].Probability)
		}
	}
	if lvl := client.outputLevel(); lvl != levelJSONObject {
		t.Errorf("level = %v, want json_object", lvl)
	}
}

func TestStripFence(t *testing.T) {
	for _, tc := range []struct{ name, in, want string }{
		{"tagged fence", "```json\n{\"p\": 0.64}\n```", `{"p": 0.64}`},
		{"bare fence", "```\n{\"p\": 0.64}\n```", `{"p": 0.64}`},
		{"single line fence", "```{\"p\":0.64}```", `{"p":0.64}`},
		{"fence with surrounding blanks", "\n  ```json\n{\"p\": 1}\n```  \n", `{"p": 1}`},
		{"unterminated fence", "```json\n{\"p\": 0.64}", `{"p": 0.64}`},
		{"plain json", `{"p":0.5}`, `{"p":0.5}`},
		{"plain prose", "no fence here", "no fence here"},
		{"backticks in the middle", "a ``` in the middle", "a ``` in the middle"},
	} {
		if got := stripFence(tc.in); got != tc.want {
			t.Errorf("%s: stripFence(%q) = %q, want %q", tc.name, tc.in, got, tc.want)
		}
	}
}

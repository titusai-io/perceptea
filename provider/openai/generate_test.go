package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/titusai-io/perceptea/classifier"
)

// generateBody builds a successful completion for the oneshot path.
func generateBody(content string) string {
	raw, _ := json.Marshal(content)
	return `{"choices":[{"message":{"role":"assistant","content":` + string(raw) +
		`}}],"usage":{"prompt_tokens":88,"completion_tokens":44}}`
}

var fixtureGenerate = classifier.GenerateRequest{
	Model:       "oneshot-model",
	System:      "You are a precise decision engine.",
	User:        "STATE:\nx\n\nQUESTIONS:\n- a (noul):",
	Temperature: 0.5,
	JSONObject:  true,
}

func TestGenerateSendsTheExpectedRequest(t *testing.T) {
	api := alwaysJSON(t, generateBody(`{"a":{"noul":0.9}}`))
	client, _ := newTestClient(t, api, nil)

	got, err := client.Generate(context.Background(), fixtureGenerate)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got.Content != `{"a":{"noul":0.9}}` {
		t.Errorf("content = %q", got.Content)
	}
	if got.InputTokens != 88 || got.OutputTokens != 44 {
		t.Errorf("usage = (%d, %d), want (88, 44)", got.InputTokens, got.OutputTokens)
	}

	req := api.request(t, 0)
	body := req.body(t)
	if body["model"] != "oneshot-model" {
		t.Errorf("model = %v", body["model"])
	}
	if body["temperature"] != 0.5 {
		t.Errorf("temperature = %v, want 0.5", body["temperature"])
	}
	if _, present := body["max_tokens"]; present {
		t.Error("max_tokens is set; the oneshot reply must not be capped")
	}
	if f := req.format(t); f != "json_object" {
		t.Errorf("response_format = %q, want json_object", f)
	}
	rf := body["response_format"].(map[string]any)
	if _, present := rf["json_schema"]; present {
		t.Error("Generate must not send a json_schema")
	}

	messages, ok := body["messages"].([]any)
	if !ok || len(messages) != 2 {
		t.Fatalf("messages = %v, want two turns", body["messages"])
	}
	system := messages[0].(map[string]any)
	if system["role"] != "system" || system["content"] != fixtureGenerate.System {
		t.Errorf("system turn = %v", system)
	}
	user := messages[1].(map[string]any)
	if user["role"] != "user" || user["content"] != fixtureGenerate.User {
		t.Errorf("user turn = %v", user)
	}
}

func TestGenerateOmitsResponseFormatWhenNotAskedFor(t *testing.T) {
	api := alwaysJSON(t, generateBody("plain text"))
	client, _ := newTestClient(t, api, nil)

	req := fixtureGenerate
	req.JSONObject = false
	if _, err := client.Generate(context.Background(), req); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if _, present := api.request(t, 0).body(t)["response_format"]; present {
		t.Error("response_format is set although JSONObject is false")
	}
}

func TestGenerateOmitsAnEmptySystemTurn(t *testing.T) {
	api := alwaysJSON(t, generateBody("ok"))
	client, _ := newTestClient(t, api, nil)

	req := fixtureGenerate
	req.System = "   "
	if _, err := client.Generate(context.Background(), req); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	messages := api.request(t, 0).body(t)["messages"].([]any)
	if len(messages) != 1 {
		t.Fatalf("sent %d turns, want 1", len(messages))
	}
	if messages[0].(map[string]any)["role"] != "user" {
		t.Errorf("the only turn is not the user turn: %v", messages[0])
	}
}

func TestGenerateDropsResponseFormatOnRejection(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, r *http.Request, n int) {
		var body struct {
			ResponseFormat *struct {
				Type string `json:"type"`
			} `json:"response_format"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body.ResponseFormat != nil {
			writeJSON(w, http.StatusBadRequest, `{"error":{"message":"response_format is not supported"}}`)
			return
		}
		writeJSON(w, http.StatusOK, generateBody("recovered"))
	})
	client, _ := newTestClient(t, api, nil)

	got, err := client.Generate(context.Background(), fixtureGenerate)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got.Content != "recovered" {
		t.Errorf("content = %q", got.Content)
	}
	if api.count() != 2 {
		t.Fatalf("made %d calls, want 2", api.count())
	}
	if lvl := client.outputLevel(); lvl != levelPlain {
		t.Errorf("level = %v, want plain: a rejected json_object rules out every response_format", lvl)
	}

	// And the downgrade is remembered, by Generate and by Score alike.
	if _, err := client.Generate(context.Background(), fixtureGenerate); err != nil {
		t.Fatalf("second Generate: %v", err)
	}
	if api.count() != 3 {
		t.Fatalf("made %d calls, want 3", api.count())
	}
	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}
	if f := api.request(t, 3).format(t); f != "" {
		t.Errorf("Score sent response_format %q after the shared downgrade", f)
	}
}

// TestGenerateOnlyRemembersADropThatAnswered is Score's rule on the oneshot
// path. The per-call fallback still runs — the request is tried again without
// a response_format — but a call that fails both ways has shown nothing about
// what the provider supports, so the shared level must survive it. Otherwise
// one failing one-shot request strips json_schema from every later Score call
// on the same client without json_schema ever having been rejected.
func TestGenerateOnlyRemembersADropThatAnswered(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
		if n < 2 {
			writeJSON(w, http.StatusBadRequest, `{"error":{"message":"context length exceeded"}}`)
			return
		}
		writeJSON(w, http.StatusOK, generateBody("recovered"))
	})
	client, _ := newTestClient(t, api, nil)

	if _, err := client.Generate(context.Background(), fixtureGenerate); err == nil {
		t.Fatal("want an error when neither shape is accepted")
	}
	if api.count() != 2 {
		t.Fatalf("made %d calls, want 2 (with the response_format, then without)", api.count())
	}
	if lvl := client.outputLevel(); lvl != levelJSONSchema {
		t.Fatalf("level = %v, want json_schema: nothing answered, so nothing was learned", lvl)
	}

	// A later Score still asks for the schema it never had refused.
	if _, err := client.Score(context.Background(), fixtureRequest); err != nil {
		t.Fatalf("Score: %v", err)
	}
	if f := api.request(t, 2).format(t); f != "json_schema" {
		t.Errorf("Score sent response_format %q, want json_schema", f)
	}
}

func TestGenerateDoesNotProbeOnAFailureUnrelatedToTheRequestShape(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		writeJSON(w, http.StatusNotFound, `{"error":{"message":"no such endpoint"}}`)
	})
	client, _ := newTestClient(t, api, nil)

	if _, err := client.Generate(context.Background(), fixtureGenerate); err == nil {
		t.Fatal("want an error")
	}
	if api.count() != 1 {
		t.Errorf("made %d calls, want 1: a 404 says nothing about response_format", api.count())
	}
	if lvl := client.outputLevel(); lvl != levelJSONSchema {
		t.Errorf("level = %v, want json_schema", lvl)
	}
}

func TestGenerateDoesNotDowngradeOnRateLimit(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, n int) {
		if n == 0 {
			writeJSON(w, http.StatusTooManyRequests, `{"error":{"message":"later"}}`)
			return
		}
		writeJSON(w, http.StatusOK, generateBody("ok"))
	})
	client, _ := newTestClient(t, api, nil)

	if _, err := client.Generate(context.Background(), fixtureGenerate); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if api.count() != 2 {
		t.Fatalf("made %d calls, want 2", api.count())
	}
	for i := range 2 {
		if f := api.request(t, i).format(t); f != "json_object" {
			t.Errorf("call %d response_format = %q, want json_object", i+1, f)
		}
	}
	if lvl := client.outputLevel(); lvl != levelJSONSchema {
		t.Errorf("level = %v, want json_schema", lvl)
	}
}

func TestGenerateFallsBackToTheConfiguredModel(t *testing.T) {
	api := alwaysJSON(t, generateBody("ok"))
	client, _ := newTestClient(t, api, nil)

	req := fixtureGenerate
	req.Model = ""
	if _, err := client.Generate(context.Background(), req); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := api.request(t, 0).body(t)["model"]; got != "config-model" {
		t.Errorf("model = %v, want config-model", got)
	}
}

func TestGenerateReturnsAPIErrors(t *testing.T) {
	api := newFakeAPI(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		writeJSON(w, http.StatusUnauthorized, `{"error":{"message":"no"}}`)
	})
	client, _ := newTestClient(t, api, nil)

	if _, err := client.Generate(context.Background(), fixtureGenerate); err == nil {
		t.Fatal("want an error")
	}
}

func TestGenerateWithNoChoicesReturnsEmptyContent(t *testing.T) {
	api := alwaysJSON(t, `{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":0}}`)
	client, _ := newTestClient(t, api, nil)

	got, err := client.Generate(context.Background(), fixtureGenerate)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got.Content != "" {
		t.Errorf("content = %q, want empty", got.Content)
	}
	if got.InputTokens != 3 {
		t.Errorf("input tokens = %d, want 3", got.InputTokens)
	}
}

func TestGenerateReadsArrayShapedContent(t *testing.T) {
	api := alwaysJSON(t, `{"choices":[{"message":{"content":[{"type":"text","text":"half "},{"type":"text","text":"and half"}]}}]}`)
	client, _ := newTestClient(t, api, nil)

	got, err := client.Generate(context.Background(), fixtureGenerate)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got.Content != "half and half" {
		t.Errorf("content = %q, want the concatenated parts", got.Content)
	}
}

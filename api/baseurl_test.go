package api

import (
	"strings"
	"testing"

	"github.com/titusai-io/perceptea/internal/config"
)

// A base URL the caller typed is caller input. Before it was checked here it
// reached the provider factory, whose plain error fell through the status
// table to a 500 — the server taking the blame for the client's typo.
func TestEvaluateRejectsAnUnusableRequestBaseURL(t *testing.T) {
	for _, tc := range []struct {
		name, baseURL, wantIn string
	}{
		{"no scheme", "openrouter.ai/api/v1", "absolute http or https"},
		{"wrong scheme", "ftp://openrouter.ai/v1", "absolute http or https"},
		{"scheme only", "https://", "host"},
		{"not a URL at all", "http://[", "not a valid URL"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, nil)
			w := h.post(withBaseURL(tc.baseURL))
			msg := h.expectError(w, 400, "invalid_request")
			if !strings.Contains(msg, tc.wantIn) {
				t.Errorf("message %q does not mention %q", msg, tc.wantIn)
			}
			// The message names the field the caller sent, under the name
			// they sent it under.
			if !strings.Contains(msg, "inference_base_url") {
				t.Errorf("message %q does not name the field at fault", msg)
			}
			if n := len(h.seenSettings); n != 0 {
				t.Errorf("built %d evaluators, want 0: the request never should have reached the provider", n)
			}
		})
	}
}

func TestEvaluateAcceptsAUsableRequestBaseURL(t *testing.T) {
	h := newHarness(t, nil)
	if w := h.post(withBaseURL("https://openrouter.ai/api/v1")); w.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if got := h.settings().BaseURL; got != "https://openrouter.ai/api/v1" {
		t.Errorf("BaseURL = %q, want the one from the body", got)
	}
}

// The retired spelling of the field is not an alias. It is an unknown field,
// and an unknown field is a 400 naming it — the same rule that stops
// "apikey" from silently spending the server's own credential.
func TestEvaluateRejectsTheRetiredBaseURLField(t *testing.T) {
	h := newHarness(t, nil)
	retired := "base" + "_url"
	body := strings.Replace(sampleBody, `"state"`,
		`"`+retired+`": "https://openrouter.ai/api/v1", "state"`, 1)

	msg := h.expectError(h.post(body), 400, codeInvalidJSON)
	if !strings.Contains(msg, retired) {
		t.Errorf("message = %q, want it to name the field it did not understand", msg)
	}
	if n := len(h.seenSettings); n != 0 {
		t.Errorf("built %d evaluators, want 0", n)
	}
}

// With request credentials disabled the body's base URL is ignored entirely,
// so even an unusable one must not turn into an error.
func TestEvaluateIgnoresABadBaseURLWhenCredentialsAreLocked(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.AllowRequestCredentials = false })
	if w := h.post(withBaseURL("not a url")); w.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if got := h.settings().BaseURL; got != "https://inference.example/v1" {
		t.Errorf("BaseURL = %q, want the configured one", got)
	}
}

// There is no endpoint allowlist: with request credentials on, any usable
// base URL the caller names is called, and the only way to stop that is
// PERCEPTEA_ALLOW_REQUEST_CREDENTIALS=false. A half-measure that let some
// hosts through would be a filter to maintain and a false sense of safety.
func TestEvaluateCallsAnyUsableBaseURLTheCallerNames(t *testing.T) {
	for _, baseURL := range []string{
		"https://anywhere.example/v1",
		"http://192.168.1.50:8000/v1",
		"http://127.0.0.1:11434/v1",
		"https://inference.example:8443/v1",
		"HTTPS://Mixed.Case.example/v1",
	} {
		t.Run(baseURL, func(t *testing.T) {
			h := newHarness(t, nil)
			if w := h.post(withBaseURL(baseURL)); w.Code != 200 {
				t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
			}
			if got := h.settings().BaseURL; got != baseURL {
				t.Errorf("BaseURL = %q, want %q", got, baseURL)
			}
		})
	}
}

// What a caller may name is one question; what the operator configured is
// another, and the server's own endpoint is never second-guessed.
func TestTheServersOwnBaseURLIsUsedWhenTheBodyNamesNone(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.BaseURL = "https://configured.example/v1" })
	if w := h.post(sampleBody); w.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if got := h.settings().BaseURL; got != "https://configured.example/v1" {
		t.Errorf("BaseURL = %q, want the configured one", got)
	}
}

// withBaseURL puts an inference base URL into the sample request body.
func withBaseURL(baseURL string) string {
	return strings.Replace(sampleBody, `"state"`,
		`"inference_base_url": `+quote(baseURL)+`, "state"`, 1)
}

func quote(s string) string {
	return `"` + strings.ReplaceAll(s, `"`, `\"`) + `"`
}

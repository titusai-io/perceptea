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
			body := strings.Replace(sampleBody, `"state"`,
				`"base_url": `+quote(tc.baseURL)+`, "state"`, 1)
			w := h.post(body)
			msg := h.expectError(w, 400, "invalid_request")
			if !strings.Contains(msg, tc.wantIn) {
				t.Errorf("message %q does not mention %q", msg, tc.wantIn)
			}
			if n := len(h.seenSettings); n != 0 {
				t.Errorf("built %d evaluators, want 0: the request never should have reached the provider", n)
			}
		})
	}
}

func TestEvaluateAcceptsAUsableRequestBaseURL(t *testing.T) {
	h := newHarness(t, nil)
	body := strings.Replace(sampleBody, `"state"`,
		`"base_url": "https://openrouter.ai/api/v1", "state"`, 1)
	if w := h.post(body); w.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if got := h.settings().BaseURL; got != "https://openrouter.ai/api/v1" {
		t.Errorf("BaseURL = %q, want the one from the body", got)
	}
}

// With request credentials disabled the body's base URL is ignored entirely,
// so even an unusable one must not turn into an error.
func TestEvaluateIgnoresABadBaseURLWhenCredentialsAreLocked(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.AllowRequestCredentials = false })
	body := strings.Replace(sampleBody, `"state"`, `"base_url": "not a url", "state"`, 1)
	if w := h.post(body); w.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if got := h.settings().BaseURL; got != "https://api.openai.com/v1" {
		t.Errorf("BaseURL = %q, want the configured one", got)
	}
}

// With PERCEPTEA_ALLOWED_BASE_URLS set, a base URL in the request body must
// be one of them: the service will otherwise call any host it can reach, on
// behalf of anyone who can reach it.
func TestEvaluateEnforcesTheAllowedBaseURLs(t *testing.T) {
	allowed := []string{"https://api.openai.com/v1", "http://127.0.0.1:11434/v1"}

	for _, tc := range []struct {
		name    string
		baseURL string
		want    int
	}{
		{"an allowed prefix", "https://api.openai.com/v1", 200},
		{"a longer path under an allowed prefix", "https://api.openai.com/v1/beta", 200},
		{"the local model server", "http://127.0.0.1:11434/v1", 200},
		{"a host that is not listed", "https://evil.example/v1", 400},
		{"another port on an allowed host", "https://api.openai.com:8443/v1", 400},
		{"a scheme downgrade", "http://api.openai.com/v1", 400},
		{"casing does not help", "HTTPS://EVIL.example/v1", 400},
		{"an allowed host in different casing", "HTTPS://API.OpenAI.com/v1", 200},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, func(c *config.Config) { c.AllowedBaseURLs = allowed })
			body := strings.Replace(sampleBody, `"state"`, `"base_url": `+quote(tc.baseURL)+`, "state"`, 1)

			w := h.post(body)

			if tc.want == 200 {
				if w.Code != 200 {
					t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
				}
				return
			}
			msg := h.expectError(w, 400, "invalid_request")
			if strings.Contains(msg, tc.baseURL) {
				t.Errorf("the message repeats the caller's base URL: %q", msg)
			}
			if !strings.Contains(msg, config.EnvAllowedBaseURLs) {
				t.Errorf("message = %q, want it to name the setting at fault", msg)
			}
			if n := len(h.seenSettings); n != 0 {
				t.Errorf("built %d evaluators, want 0: the call was made anyway", n)
			}
		})
	}
}

// The default is no restriction: pointing this at a local model server is a
// real use case, and an empty list must not quietly become an empty allowlist.
func TestAnEmptyAllowListAllowsEverything(t *testing.T) {
	h := newHarness(t, nil)
	body := strings.Replace(sampleBody, `"state"`, `"base_url": "http://192.168.1.50:8000/v1", "state"`, 1)
	if w := h.post(body); w.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
}

// The list bounds what a caller may name, not what the operator configured.
func TestTheAllowListDoesNotApplyToTheServersOwnBaseURL(t *testing.T) {
	h := newHarness(t, func(c *config.Config) {
		c.BaseURL = "https://configured.example/v1"
		c.AllowedBaseURLs = []string{"https://api.openai.com/v1"}
	})
	if w := h.post(sampleBody); w.Code != 200 {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if got := h.settings().BaseURL; got != "https://configured.example/v1" {
		t.Errorf("BaseURL = %q, want the configured one", got)
	}
}

func quote(s string) string {
	return `"` + strings.ReplaceAll(s, `"`, `\"`) + `"`
}

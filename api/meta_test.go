package api

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/titusai-io/perceptea/internal/config"
	"github.com/titusai-io/perceptea/provider/openai"
)

func TestHealthPayload(t *testing.T) {
	h := newHarness(t, nil)
	w := h.get("/api/health")

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	const want = `{"ok":true,"service":"perceptea","provider":"openai","model":"gpt-4o-mini","api_key_configured":true}`
	if got := strings.TrimSpace(w.Body.String()); got != want {
		t.Errorf("health:\n got %s\nwant %s", got, want)
	}
}

func TestHealthReportsAMissingKey(t *testing.T) {
	h := newHarness(t, func(c *config.Config) {
		c.APIKey = ""
		c.Provider = "zai"
		c.Model = "glm-5.3-flash"
	})
	w := h.get("/api/health")

	const want = `{"ok":true,"service":"perceptea","provider":"zai","model":"glm-5.3-flash","api_key_configured":false}`
	if got := strings.TrimSpace(w.Body.String()); got != want {
		t.Errorf("health:\n got %s\nwant %s", got, want)
	}
}

func TestProvidersPayload(t *testing.T) {
	h := newHarness(t, func(c *config.Config) {
		c.Provider = "openrouter"
		c.BaseURL = "https://openrouter.ai/api/v1"
		c.Model = "z-ai/glm-5.3-flash"
		c.AllowRequestCredentials = false
	})
	w := h.get("/api/providers")

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	var got providersResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("decoding %s: %v", w.Body.String(), err)
	}

	wantActive := activeInfo{
		Provider:                "openrouter",
		BaseURL:                 "https://openrouter.ai/api/v1",
		Model:                   "z-ai/glm-5.3-flash",
		APIKeyConfigured:        true,
		AllowRequestCredentials: false,
	}
	if got.Active != wantActive {
		t.Errorf("active:\n got %+v\nwant %+v", got.Active, wantActive)
	}

	presets := config.Providers()
	if len(got.Providers) != len(presets) {
		t.Fatalf("listed %d providers, want %d", len(got.Providers), len(presets))
	}
	for i, p := range presets {
		want := providerInfo{
			Name:         p.Name,
			BaseURL:      p.BaseURL,
			DefaultModel: p.DefaultModel,
			EnvKey:       p.EnvKey,
			Active:       p.Name == "openrouter",
		}
		if got.Providers[i] != want {
			t.Errorf("provider %d:\n got %+v\nwant %+v", i, got.Providers[i], want)
		}
	}
}

func TestMetaEndpointsRejectAWrongMethod(t *testing.T) {
	h := newHarness(t, nil)
	for _, path := range []string{"/api/health", "/api/providers"} {
		w := h.send(httptest.NewRequest(http.MethodPost, path, strings.NewReader("")))
		// The status was all this asserted, and the body was text/plain: a
		// client promised one error shape for every failure had to parse
		// prose for this one.
		message := h.expectError(w, http.StatusMethodNotAllowed, "method_not_allowed")
		if !strings.Contains(message, http.MethodGet) {
			t.Errorf("POST %s: message = %q, want it to name the method that works", path, message)
		}
		if got := w.Header().Get("Allow"); got != http.MethodGet {
			t.Errorf("POST %s: Allow = %q, want %q", path, got, http.MethodGet)
		}
		if ct := w.Header().Get("Content-Type"); ct != "application/json; charset=utf-8" {
			t.Errorf("POST %s: Content-Type = %q", path, ct)
		}
	}
}

// A base URL is a credential carrier: an operator pointing this at a gateway
// puts the key in the userinfo or the query, and /api/providers answers
// anyone who can reach the port.
func TestProvidersNeverExposesACredentialInTheBaseURL(t *testing.T) {
	const gateway = "https://svc:sk-secret-0123456789@internal.example:8443/v1?key=abc123def#frag"
	h := newHarness(t, func(c *config.Config) { c.BaseURL = gateway })

	w := h.get("/api/providers")

	body := w.Body.String()
	for _, secret := range []string{"sk-secret-0123456789", "svc:", "key=abc123def"} {
		if strings.Contains(body, secret) {
			t.Errorf("the providers payload contains %q:\n%s", secret, body)
		}
	}
	var got providersResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("decoding %s: %v", body, err)
	}
	// What is left still says where the calls are going.
	if want := "https://internal.example:8443/v1"; got.Active.BaseURL != want {
		t.Errorf("active.base_url = %q, want %q", got.Active.BaseURL, want)
	}
}

// TestTheAPIKeyNeverEscapes drives every endpoint, including failures whose
// upstream text embeds the key, and then looks for the key in everything that
// left the process.
func TestTheAPIKeyNeverEscapes(t *testing.T) {
	h := newHarness(t, nil)

	var bodies []string
	record := func(body string) { bodies = append(bodies, body) }

	h.answer(sampleResponse())
	record(h.post(sampleBody).Body.String())
	record(h.get("/api/health").Body.String())
	record(h.get("/api/providers").Body.String())

	// A provider that echoes the key back at us must not be relayed verbatim.
	h.failWith(&openai.APIError{
		StatusCode: http.StatusUnauthorized,
		Type:       "invalid_api_key",
		Message:    "Incorrect API key provided: " + testKey,
		Body:       `{"error":{"message":"Incorrect API key provided: ` + testKey + `"}}`,
	})
	record(h.post(validBody).Body.String())

	h.failWith(errors.New("dial https://user:" + testKey + "@api.openai.com: refused"))
	record(h.post(validBody).Body.String())

	for i, body := range bodies {
		if strings.Contains(body, testKey) {
			t.Errorf("response %d contains the API key: %s", i, body)
		}
	}
	if logs := h.logs.String(); strings.Contains(logs, testKey) {
		t.Errorf("the log contains the API key:\n%s", logs)
	}
}

func TestTheRequestKeyNeverEscapes(t *testing.T) {
	const bodyKey = "sk-from-the-body-abcdefghij"
	h := newHarness(t, nil)
	h.failWith(&openai.APIError{
		StatusCode: http.StatusUnauthorized,
		Message:    "Incorrect API key provided: " + bodyKey,
	})

	body := `{"state":"s","questions":{"q":{"type":"noul"}},"api_key":"` + bodyKey + `"}`
	w := h.post(body)

	if strings.Contains(w.Body.String(), bodyKey) {
		t.Errorf("the response echoed the caller's key: %s", w.Body.String())
	}
	if strings.Contains(h.logs.String(), bodyKey) {
		t.Errorf("the log contains the caller's key:\n%s", h.logs.String())
	}
}

func TestNewServerFillsInMissingLimits(t *testing.T) {
	srv, err := NewServer(Options{})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	cfg := srv.Config()
	if cfg.MaxConcurrency != config.DefaultMaxConcurrency {
		t.Errorf("MaxConcurrency = %d, want %d", cfg.MaxConcurrency, config.DefaultMaxConcurrency)
	}
	if cfg.RequestTimeout != config.DefaultRequestTimeout {
		t.Errorf("RequestTimeout = %s, want %s", cfg.RequestTimeout, config.DefaultRequestTimeout)
	}
	if cfg.MaxBodyBytes != config.DefaultMaxBodyBytes {
		t.Errorf("MaxBodyBytes = %d, want %d", cfg.MaxBodyBytes, config.DefaultMaxBodyBytes)
	}
	if cfg.Provider != config.DefaultProvider {
		t.Errorf("Provider = %q, want %q", cfg.Provider, config.DefaultProvider)
	}
	if srv.Handler() == nil {
		t.Error("Handler() is nil")
	}
}

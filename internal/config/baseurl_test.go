package config

import (
	"strings"
	"testing"
)

func TestValidateBaseURL(t *testing.T) {
	for _, tc := range []struct {
		name, raw, wantIn string
	}{
		{"an https URL", "https://inference.example/v1", ""},
		{"an http URL", "http://localhost:11434/v1", ""},
		{"padded", "  https://inference.example/v1  ", ""},
		{"no scheme", "openrouter.ai/api/v1", "absolute http or https"},
		{"wrong scheme", "ftp://openrouter.ai/v1", "absolute http or https"},
		{"scheme only", "https://", "host"},
		{"not a URL at all", "http://[", "not a valid URL"},
		{"empty", "", "absolute http or https"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateBaseURL(tc.raw)
			if tc.wantIn == "" {
				if err != nil {
					t.Fatalf("ValidateBaseURL(%q) = %v, want nil", tc.raw, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("ValidateBaseURL(%q) = nil, want an error", tc.raw)
			}
			if !strings.Contains(err.Error(), tc.wantIn) {
				t.Errorf("error %q does not mention %q", err, tc.wantIn)
			}
			// The value may be a credential; the reason may never repeat it.
			if tc.raw != "" && strings.Contains(err.Error(), tc.raw) {
				t.Errorf("the error repeats the value: %q", err)
			}
		})
	}
}

func TestDisplayBaseURL(t *testing.T) {
	for _, tc := range []struct{ name, raw, want string }{
		{"nothing to strip", "https://inference.example/v1", "https://inference.example/v1"},
		{
			"a key in the userinfo",
			"https://svc:sk-secret-0123456789@internal.example/v1",
			"https://internal.example/v1",
		},
		{"a bare userinfo", "https://sk-secret-0123456789@gw.example/v1", "https://gw.example/v1"},
		{"a key in the query", "https://gw.example/v1?key=abc123", "https://gw.example/v1"},
		{"an empty query", "https://gw.example/v1?", "https://gw.example/v1"},
		{"a fragment", "https://gw.example/v1#notes", "https://gw.example/v1"},
		{
			"all of them at once",
			"https://svc:sk-secret@internal.example:8443/v1?key=abc#f",
			"https://internal.example:8443/v1",
		},
		{"a port survives", "http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1"},
		{"empty", "", ""},
		{"unparseable", "http://[", "[unparseable url]"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := DisplayBaseURL(tc.raw); got != tc.want {
				t.Errorf("DisplayBaseURL(%q) = %q, want %q", tc.raw, got, tc.want)
			}
		})
	}
}

// A base URL that cannot be called used to start the server anyway: health
// reported ok and every request failed with a 500 for as long as it ran.
func TestLoadFromRejectsAnUnusableBaseURL(t *testing.T) {
	for _, raw := range []string{"not a url", "ftp://gw.example/v1", "https://", "http://["} {
		t.Run(raw, func(t *testing.T) {
			cfg, err := LoadFrom(envOf(map[string]string{"PERCEPTEA_INFERENCE_BASE_URL": raw}))
			if err == nil {
				t.Fatalf("LoadFrom(%q) = %+v, want a startup error", raw, cfg)
			}
			if !strings.Contains(err.Error(), EnvInferenceBaseURL) {
				t.Errorf("error %q does not name %s", err, EnvInferenceBaseURL)
			}
			if strings.Contains(err.Error(), raw) {
				t.Errorf("the error repeats the value, which may hold a credential: %q", err)
			}
		})
	}
}

func TestLoadFromAcceptsAUsableBaseURL(t *testing.T) {
	cfg := loadWith(t, map[string]string{"PERCEPTEA_INFERENCE_BASE_URL": "http://localhost:11434/v1"})
	if cfg.BaseURL != "http://localhost:11434/v1" {
		t.Errorf("BaseURL = %q", cfg.BaseURL)
	}
}

// The default must pass the check an operator's own value faces, or an
// unconfigured server would refuse to start.
func TestTheDefaultInferenceBaseURLIsUsable(t *testing.T) {
	if err := ValidateBaseURL(DefaultInferenceBaseURL); err != nil {
		t.Errorf("the default inference base URL is unusable: %v", err)
	}
}

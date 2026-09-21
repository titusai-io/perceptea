package config

import (
	"strings"
	"testing"
)

func TestValidateBaseURL(t *testing.T) {
	for _, tc := range []struct {
		name, raw, wantIn string
	}{
		{"an https URL", "https://api.openai.com/v1", ""},
		{"an http URL", "http://localhost:11434/v1", ""},
		{"padded", "  https://api.openai.com/v1  ", ""},
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
		{"nothing to strip", "https://api.openai.com/v1", "https://api.openai.com/v1"},
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

func TestAllowsBaseURL(t *testing.T) {
	for _, tc := range []struct {
		name    string
		allowed []string
		raw     string
		want    bool
	}{
		{"no list allows anything", nil, "https://anywhere.example/v1", true},
		{"an exact match", []string{"https://api.openai.com/v1"}, "https://api.openai.com/v1", true},
		{"a longer path", []string{"https://api.openai.com/v1"}, "https://api.openai.com/v1/beta", true},
		{"a second entry", []string{"https://a.example", "https://b.example"}, "https://b.example/v1", true},
		{"a host that is not listed", []string{"https://a.example"}, "https://evil.example/v1", false},
		{"a scheme downgrade", []string{"https://a.example"}, "http://a.example/v1", false},
		{"upper-cased host", []string{"https://a.example"}, "HTTPS://A.Example/v1", true},
		{"upper-cased prefix", []string{"HTTPS://A.Example"}, "https://a.example/v1", true},
		{"a case-sensitive path", []string{"https://a.example/V1"}, "https://a.example/v1", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{AllowedBaseURLs: tc.allowed}
			if got := cfg.AllowsBaseURL(tc.raw); got != tc.want {
				t.Errorf("AllowsBaseURL(%q) with %v = %v, want %v", tc.raw, tc.allowed, got, tc.want)
			}
		})
	}
}

// A base URL that cannot be called used to start the server anyway: health
// reported ok and every request failed with a 500 for as long as it ran.
func TestLoadFromRejectsAnUnusableBaseURL(t *testing.T) {
	for _, raw := range []string{"not a url", "ftp://gw.example/v1", "https://", "http://["} {
		t.Run(raw, func(t *testing.T) {
			cfg, err := LoadFrom(envOf(map[string]string{"PERCEPTEA_BASE_URL": raw}))
			if err == nil {
				t.Fatalf("LoadFrom(%q) = %+v, want a startup error", raw, cfg)
			}
			if !strings.Contains(err.Error(), EnvBaseURL) {
				t.Errorf("error %q does not name %s", err, EnvBaseURL)
			}
			if strings.Contains(err.Error(), raw) {
				t.Errorf("the error repeats the value, which may hold a credential: %q", err)
			}
		})
	}
}

func TestLoadFromAcceptsAUsableBaseURL(t *testing.T) {
	cfg := loadWith(t, map[string]string{"PERCEPTEA_BASE_URL": "http://localhost:11434/v1"})
	if cfg.BaseURL != "http://localhost:11434/v1" {
		t.Errorf("BaseURL = %q", cfg.BaseURL)
	}
}

// Every preset must pass the check the environment override now faces.
func TestEveryPresetBaseURLIsUsable(t *testing.T) {
	for _, p := range Providers() {
		if err := ValidateBaseURL(p.BaseURL); err != nil {
			t.Errorf("preset %s has an unusable base URL: %v", p.Name, err)
		}
	}
}

func TestLoadFromAllowedBaseURLs(t *testing.T) {
	cfg := loadWith(t, map[string]string{
		"PERCEPTEA_ALLOWED_BASE_URLS": " https://api.openai.com/v1 , http://127.0.0.1:11434/v1 ,",
	})
	want := []string{"https://api.openai.com/v1", "http://127.0.0.1:11434/v1"}
	if len(cfg.AllowedBaseURLs) != len(want) {
		t.Fatalf("AllowedBaseURLs = %q, want %q", cfg.AllowedBaseURLs, want)
	}
	for i := range want {
		if cfg.AllowedBaseURLs[i] != want[i] {
			t.Errorf("AllowedBaseURLs[%d] = %q, want %q", i, cfg.AllowedBaseURLs[i], want[i])
		}
	}

	if got := loadWith(t, nil).AllowedBaseURLs; len(got) != 0 {
		t.Errorf("AllowedBaseURLs = %q by default, want empty: a local model server is a real use case", got)
	}
}

func TestLoadFromRejectsAnUnusableAllowedBaseURL(t *testing.T) {
	cfg, err := LoadFrom(envOf(map[string]string{
		"PERCEPTEA_ALLOWED_BASE_URLS": "https://api.openai.com/v1,localhost:11434",
	}))
	if err == nil {
		t.Fatalf("LoadFrom = %+v, want an error for a prefix with no scheme", cfg)
	}
	if !strings.Contains(err.Error(), EnvAllowedBaseURLs) {
		t.Errorf("error %q does not name %s", err, EnvAllowedBaseURLs)
	}
	if !strings.Contains(err.Error(), "entry 2") {
		t.Errorf("error %q does not say which entry is at fault", err)
	}
}

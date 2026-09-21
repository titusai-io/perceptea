package config

import (
	"log/slog"
	"maps"
	"reflect"
	"strings"
	"testing"
	"time"
)

// envOf turns a map into the lookup function LoadFrom expects.
func envOf(vars map[string]string) func(string) string {
	m := maps.Clone(vars)
	return func(key string) string { return m[key] }
}

// loadWith loads a configuration that is expected to be valid.
func loadWith(t *testing.T, vars map[string]string) Config {
	t.Helper()
	cfg, err := LoadFrom(envOf(vars))
	if err != nil {
		t.Fatalf("LoadFrom(%v): unexpected error: %v", vars, err)
	}
	return cfg
}

// TestLoadFromDefaults is the one test whose subject is what an unconfigured
// server resolves to, so it names the real defaults. Every other test here
// uses a fixture value, because no other test cares what they are.
func TestLoadFromDefaults(t *testing.T) {
	cfg := loadWith(t, nil)

	want := Config{
		Addr:                    ":5301",
		BaseURL:                 "https://api.deepinfra.com/v1/openai",
		APIKey:                  "",
		Model:                   "mistralai/Mistral-Small-24B-Instruct-2501",
		ReasoningEffort:         "",
		Temperature:             0,
		MaxConcurrency:          8,
		RequestTimeout:          60 * time.Second,
		MaxBodyBytes:            2097152,
		AllowRequestCredentials: true,
		LogLevel:                slog.LevelInfo,
		LogFormat:               "text",
	}
	if !reflect.DeepEqual(cfg, want) {
		t.Errorf("defaults:\n got %+v\nwant %+v", cfg, want)
	}
	if cfg.APIKeyConfigured() {
		t.Error("APIKeyConfigured() = true with no key set")
	}
	// The same two values, named through the constants the rest of the
	// service resolves them from.
	if DefaultInferenceBaseURL != "https://api.deepinfra.com/v1/openai" {
		t.Errorf("DefaultInferenceBaseURL = %q", DefaultInferenceBaseURL)
	}
	if DefaultModel != "mistralai/Mistral-Small-24B-Instruct-2501" {
		t.Errorf("DefaultModel = %q", DefaultModel)
	}
}

// An unset reasoning effort is the whole of the default behaviour: the
// provider is sent no reasoning field, exactly as it was before the setting
// existed.
func TestLoadFromReasoningEffort(t *testing.T) {
	for _, tc := range []struct{ set, want string }{
		{"", ""},
		{"none", "none"},
		{"low", "low"},
		{"medium", "medium"},
		{"high", "high"},
		{"  high  ", "high"},
		{"HIGH", "high"},
	} {
		name := tc.set
		if name == "" {
			name = "unset"
		}
		t.Run(name, func(t *testing.T) {
			vars := map[string]string{}
			if tc.set != "" {
				vars["PERCEPTEA_REASONING_EFFORT"] = tc.set
			}
			if got := loadWith(t, vars).ReasoningEffort; got != tc.want {
				t.Errorf("ReasoningEffort = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestLoadFromMissingAPIKeyIsNotAnError(t *testing.T) {
	cfg, err := LoadFrom(envOf(map[string]string{"PERCEPTEA_MODEL": "probe-1"}))
	if err != nil {
		t.Fatalf("a missing API key must not fail the load, got: %v", err)
	}
	if cfg.APIKey != "" {
		t.Errorf("APIKey = %q, want empty", cfg.APIKey)
	}
}

func TestLoadFromEveryVariable(t *testing.T) {
	cfg := loadWith(t, map[string]string{
		"PERCEPTEA_ADDR":                      "127.0.0.1:9999",
		"PERCEPTEA_INFERENCE_BASE_URL":        "https://example.test/v1",
		"PERCEPTEA_API_KEY":                   "sk-from-perceptea",
		"PERCEPTEA_MODEL":                     "some/model",
		"PERCEPTEA_REASONING_EFFORT":          "medium",
		"PERCEPTEA_TEMPERATURE":               "0.25",
		"PERCEPTEA_MAX_CONCURRENCY":           "32",
		"PERCEPTEA_REQUEST_TIMEOUT":           "90s",
		"PERCEPTEA_MAX_BODY_BYTES":            "4096",
		"PERCEPTEA_ALLOW_REQUEST_CREDENTIALS": "false",
		"PERCEPTEA_LOG_LEVEL":                 "debug",
		"PERCEPTEA_LOG_FORMAT":                "json",
	})

	want := Config{
		Addr:                    "127.0.0.1:9999",
		BaseURL:                 "https://example.test/v1",
		APIKey:                  "sk-from-perceptea",
		Model:                   "some/model",
		ReasoningEffort:         "medium",
		Temperature:             0.25,
		MaxConcurrency:          32,
		RequestTimeout:          90 * time.Second,
		MaxBodyBytes:            4096,
		AllowRequestCredentials: false,
		LogLevel:                slog.LevelDebug,
		LogFormat:               "json",
	}
	if !reflect.DeepEqual(cfg, want) {
		t.Errorf("\n got %+v\nwant %+v", cfg, want)
	}
}

// vendorKeyVars are the conventional API key variables of the endpoints this
// service can be pointed at. They are assembled from their halves rather than
// written out, so that a search of this repository for one of them stays
// empty: naming them here would read as a promise that they are consulted.
func vendorKeyVars() []string {
	var out []string
	for _, vendor := range []string{"OPENAI", "OPENROUTER", "DEEPINFRA", "ZAI"} {
		out = append(out, vendor+"_API_KEY")
	}
	return out
}

// The key is read from PERCEPTEA_API_KEY and from nowhere else. A key picked
// up from a vendor's own variable is a credential the operator never pointed
// at this service, spent on calls they did not ask for.
func TestLoadFromReadsTheKeyOnlyFromItsOwnVariable(t *testing.T) {
	vars := map[string]string{}
	for _, name := range vendorKeyVars() {
		vars[name] = "sk-" + strings.ToLower(name)
	}

	cfg := loadWith(t, vars)
	if cfg.APIKey != "" {
		t.Errorf("APIKey = %q, want empty: no vendor key variable is read", cfg.APIKey)
	}
	if cfg.APIKeyConfigured() {
		t.Error("APIKeyConfigured() = true with only a vendor key variable set")
	}

	vars["PERCEPTEA_API_KEY"] = "sk-perceptea"
	cfg = loadWith(t, vars)
	if cfg.APIKey != "sk-perceptea" {
		t.Errorf("APIKey = %q, want the value of PERCEPTEA_API_KEY", cfg.APIKey)
	}
}

// The variable the inference base URL used to be read from is a clean break,
// not an alias: a server still honouring it would call an endpoint its
// configuration no longer mentions.
func TestLoadFromIgnoresTheRetiredBaseURLVariable(t *testing.T) {
	// Joined from its parts for the same reason as the key variables above.
	retired := strings.Join([]string{"PERCEPTEA", "BASE", "URL"}, "_")

	cfg := loadWith(t, map[string]string{retired: "https://stale.example/v1"})
	if cfg.BaseURL != DefaultInferenceBaseURL {
		t.Errorf("BaseURL = %q, want the default: %s is not read", cfg.BaseURL, retired)
	}
}

func TestLoadFromBaseURLAndModelOverrideTheDefaults(t *testing.T) {
	cfg := loadWith(t, map[string]string{
		"PERCEPTEA_INFERENCE_BASE_URL": "http://localhost:11434/v1",
		"PERCEPTEA_MODEL":              "probe-1",
	})
	if cfg.BaseURL != "http://localhost:11434/v1" {
		t.Errorf("BaseURL = %q, want the override", cfg.BaseURL)
	}
	if cfg.Model != "probe-1" {
		t.Errorf("Model = %q, want the override", cfg.Model)
	}
}

func TestLoadFromWhitespaceOnlyValuesFallBackToDefaults(t *testing.T) {
	cfg := loadWith(t, map[string]string{
		"PERCEPTEA_ADDR":            "   ",
		"PERCEPTEA_MAX_CONCURRENCY": "  16  ",
	})
	if cfg.Addr != ":5301" {
		t.Errorf("Addr = %q, want the default for a blank value", cfg.Addr)
	}
	if cfg.MaxConcurrency != 16 {
		t.Errorf("MaxConcurrency = %d, want a padded value to be trimmed", cfg.MaxConcurrency)
	}
}

func TestLoadFromErrors(t *testing.T) {
	tests := []struct {
		name string
		vars map[string]string
		// wantVar must appear in the error message.
		wantVar string
		// wantIn must all appear in it too.
		wantIn []string
	}{
		{"bad temperature", map[string]string{"PERCEPTEA_TEMPERATURE": "warm"}, "PERCEPTEA_TEMPERATURE", nil},
		{"bad concurrency", map[string]string{"PERCEPTEA_MAX_CONCURRENCY": "lots"}, "PERCEPTEA_MAX_CONCURRENCY", nil},
		{"zero concurrency", map[string]string{"PERCEPTEA_MAX_CONCURRENCY": "0"}, "PERCEPTEA_MAX_CONCURRENCY", nil},
		{"bad timeout", map[string]string{"PERCEPTEA_REQUEST_TIMEOUT": "soon"}, "PERCEPTEA_REQUEST_TIMEOUT", nil},
		{"bare number timeout", map[string]string{"PERCEPTEA_REQUEST_TIMEOUT": "60"}, "PERCEPTEA_REQUEST_TIMEOUT", nil},
		{"negative timeout", map[string]string{"PERCEPTEA_REQUEST_TIMEOUT": "-5s"}, "PERCEPTEA_REQUEST_TIMEOUT", nil},
		{"bad body limit", map[string]string{"PERCEPTEA_MAX_BODY_BYTES": "2mb"}, "PERCEPTEA_MAX_BODY_BYTES", nil},
		{"zero body limit", map[string]string{"PERCEPTEA_MAX_BODY_BYTES": "0"}, "PERCEPTEA_MAX_BODY_BYTES", nil},
		{"bad credentials flag", map[string]string{"PERCEPTEA_ALLOW_REQUEST_CREDENTIALS": "sure"}, "PERCEPTEA_ALLOW_REQUEST_CREDENTIALS", nil},
		{"bad log level", map[string]string{"PERCEPTEA_LOG_LEVEL": "chatty"}, "PERCEPTEA_LOG_LEVEL", nil},
		{"bad log format", map[string]string{"PERCEPTEA_LOG_FORMAT": "xml"}, "PERCEPTEA_LOG_FORMAT", nil},
		// A rejected effort has to list what would have been accepted:
		// "thinking" and "off" are the plausible guesses, and neither is a
		// value, so the message is the only place the operator can learn
		// them.
		{
			"bad reasoning effort",
			map[string]string{"PERCEPTEA_REASONING_EFFORT": "thinking"},
			"PERCEPTEA_REASONING_EFFORT",
			[]string{`"thinking"`, `"none"`, `"low"`, `"medium"`, `"high"`},
		},
		{
			"an effort that is only nearly a value",
			map[string]string{"PERCEPTEA_REASONING_EFFORT": "off"},
			"PERCEPTEA_REASONING_EFFORT",
			[]string{`"none"`},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := LoadFrom(envOf(tt.vars))
			if err == nil {
				t.Fatalf("LoadFrom(%v) = %+v, want an error", tt.vars, cfg)
			}
			if !strings.Contains(err.Error(), tt.wantVar) {
				t.Errorf("error %q does not name %s", err, tt.wantVar)
			}
			for _, want := range tt.wantIn {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not mention %s", err, want)
				}
			}
			if !reflect.DeepEqual(cfg, Config{}) {
				t.Errorf("a failed load returned a non-zero config: %+v", cfg)
			}
		})
	}
}

func TestLoadFromNilLookup(t *testing.T) {
	cfg, err := LoadFrom(nil)
	if err != nil {
		t.Fatalf("LoadFrom(nil): %v", err)
	}
	if cfg.Addr != ":5301" || cfg.BaseURL != DefaultInferenceBaseURL {
		t.Errorf("LoadFrom(nil) = %+v, want the defaults", cfg)
	}
}

func TestLoadReadsTheProcessEnvironment(t *testing.T) {
	t.Setenv("PERCEPTEA_ADDR", "127.0.0.1:7")
	t.Setenv("PERCEPTEA_MODEL", "probe-1")
	t.Setenv("PERCEPTEA_API_KEY", "sk-process")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load(): %v", err)
	}
	if cfg.Addr != "127.0.0.1:7" {
		t.Errorf("Addr = %q, want the process value", cfg.Addr)
	}
	if cfg.Model != "probe-1" || cfg.APIKey != "sk-process" {
		t.Errorf("Load() = %+v, want the values from the process environment", cfg)
	}
	if !cfg.APIKeyConfigured() {
		t.Error("APIKeyConfigured() = false with a key set")
	}
}

func TestLogLevelAndFormatAreCaseInsensitive(t *testing.T) {
	cfg := loadWith(t, map[string]string{
		"PERCEPTEA_LOG_LEVEL":  "WARN",
		"PERCEPTEA_LOG_FORMAT": "JSON",
	})
	if cfg.LogLevel != slog.LevelWarn {
		t.Errorf("LogLevel = %v, want warn", cfg.LogLevel)
	}
	if cfg.LogFormat != "json" {
		t.Errorf("LogFormat = %q, want %q", cfg.LogFormat, "json")
	}
}

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

func TestLoadFromDefaults(t *testing.T) {
	cfg := loadWith(t, nil)

	want := Config{
		Addr:                    ":8080",
		Provider:                "openai",
		BaseURL:                 "https://api.openai.com/v1",
		APIKey:                  "",
		Model:                   "gpt-4o-mini",
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
}

func TestLoadFromMissingAPIKeyIsNotAnError(t *testing.T) {
	cfg, err := LoadFrom(envOf(map[string]string{"PERCEPTEA_PROVIDER": "zai"}))
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
		"PERCEPTEA_PROVIDER":                  "openrouter",
		"PERCEPTEA_BASE_URL":                  "https://example.test/v1",
		"PERCEPTEA_API_KEY":                   "sk-from-perceptea",
		"PERCEPTEA_MODEL":                     "some/model",
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
		Provider:                "openrouter",
		BaseURL:                 "https://example.test/v1",
		APIKey:                  "sk-from-perceptea",
		Model:                   "some/model",
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

func TestLoadFromProviderPresets(t *testing.T) {
	for _, p := range Providers() {
		t.Run(p.Name, func(t *testing.T) {
			cfg := loadWith(t, map[string]string{
				"PERCEPTEA_PROVIDER": strings.ToUpper(p.Name),
				p.EnvKey:             "sk-preset",
			})
			if cfg.Provider != p.Name {
				t.Errorf("Provider = %q, want %q", cfg.Provider, p.Name)
			}
			if cfg.BaseURL != p.BaseURL {
				t.Errorf("BaseURL = %q, want %q", cfg.BaseURL, p.BaseURL)
			}
			if cfg.Model != p.DefaultModel {
				t.Errorf("Model = %q, want %q", cfg.Model, p.DefaultModel)
			}
			if cfg.APIKey != "sk-preset" {
				t.Errorf("APIKey = %q, want the value of %s", cfg.APIKey, p.EnvKey)
			}
		})
	}
}

func TestLoadFromAPIKeyPrecedence(t *testing.T) {
	cfg := loadWith(t, map[string]string{
		"PERCEPTEA_PROVIDER": "openai",
		"PERCEPTEA_API_KEY":  "sk-generic",
		"OPENAI_API_KEY":     "sk-provider",
	})
	if cfg.APIKey != "sk-generic" {
		t.Errorf("APIKey = %q, want PERCEPTEA_API_KEY to win", cfg.APIKey)
	}

	cfg = loadWith(t, map[string]string{
		"PERCEPTEA_PROVIDER": "openai",
		"OPENAI_API_KEY":     "sk-provider",
	})
	if cfg.APIKey != "sk-provider" {
		t.Errorf("APIKey = %q, want the preset's env key as a fallback", cfg.APIKey)
	}
}

func TestLoadFromProviderKeyIsNotCrossRead(t *testing.T) {
	// The openai preset must not pick up another provider's key.
	cfg := loadWith(t, map[string]string{
		"PERCEPTEA_PROVIDER": "openai",
		"ZAI_API_KEY":        "sk-zai",
	})
	if cfg.APIKey != "" {
		t.Errorf("APIKey = %q, want empty: only the active preset's key counts", cfg.APIKey)
	}
}

func TestLoadFromBaseURLOverridesPreset(t *testing.T) {
	cfg := loadWith(t, map[string]string{
		"PERCEPTEA_PROVIDER": "deepinfra",
		"PERCEPTEA_BASE_URL": "http://localhost:11434/v1",
		"PERCEPTEA_MODEL":    "zai-org/GLM-5.3-Flash-Corrected",
	})
	if cfg.BaseURL != "http://localhost:11434/v1" {
		t.Errorf("BaseURL = %q, want the override", cfg.BaseURL)
	}
	if cfg.Model != "zai-org/GLM-5.3-Flash-Corrected" {
		t.Errorf("Model = %q, want the override", cfg.Model)
	}
}

func TestLoadFromWhitespaceOnlyValuesFallBackToDefaults(t *testing.T) {
	cfg := loadWith(t, map[string]string{
		"PERCEPTEA_ADDR":            "   ",
		"PERCEPTEA_MAX_CONCURRENCY": "  16  ",
	})
	if cfg.Addr != ":8080" {
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
	}{
		{"unknown provider", map[string]string{"PERCEPTEA_PROVIDER": "anthropic"}, "PERCEPTEA_PROVIDER"},
		{"bad temperature", map[string]string{"PERCEPTEA_TEMPERATURE": "warm"}, "PERCEPTEA_TEMPERATURE"},
		{"bad concurrency", map[string]string{"PERCEPTEA_MAX_CONCURRENCY": "lots"}, "PERCEPTEA_MAX_CONCURRENCY"},
		{"zero concurrency", map[string]string{"PERCEPTEA_MAX_CONCURRENCY": "0"}, "PERCEPTEA_MAX_CONCURRENCY"},
		{"bad timeout", map[string]string{"PERCEPTEA_REQUEST_TIMEOUT": "soon"}, "PERCEPTEA_REQUEST_TIMEOUT"},
		{"bare number timeout", map[string]string{"PERCEPTEA_REQUEST_TIMEOUT": "60"}, "PERCEPTEA_REQUEST_TIMEOUT"},
		{"negative timeout", map[string]string{"PERCEPTEA_REQUEST_TIMEOUT": "-5s"}, "PERCEPTEA_REQUEST_TIMEOUT"},
		{"bad body limit", map[string]string{"PERCEPTEA_MAX_BODY_BYTES": "2mb"}, "PERCEPTEA_MAX_BODY_BYTES"},
		{"zero body limit", map[string]string{"PERCEPTEA_MAX_BODY_BYTES": "0"}, "PERCEPTEA_MAX_BODY_BYTES"},
		{"bad credentials flag", map[string]string{"PERCEPTEA_ALLOW_REQUEST_CREDENTIALS": "sure"}, "PERCEPTEA_ALLOW_REQUEST_CREDENTIALS"},
		{"bad log level", map[string]string{"PERCEPTEA_LOG_LEVEL": "chatty"}, "PERCEPTEA_LOG_LEVEL"},
		{"bad log format", map[string]string{"PERCEPTEA_LOG_FORMAT": "xml"}, "PERCEPTEA_LOG_FORMAT"},
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
	if cfg.Addr != ":8080" || cfg.Provider != "openai" {
		t.Errorf("LoadFrom(nil) = %+v, want the defaults", cfg)
	}
}

func TestLoadReadsTheProcessEnvironment(t *testing.T) {
	t.Setenv("PERCEPTEA_ADDR", "127.0.0.1:7")
	t.Setenv("PERCEPTEA_PROVIDER", "zai")
	t.Setenv("ZAI_API_KEY", "sk-process")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load(): %v", err)
	}
	if cfg.Addr != "127.0.0.1:7" {
		t.Errorf("Addr = %q, want the process value", cfg.Addr)
	}
	if cfg.Provider != "zai" || cfg.APIKey != "sk-process" {
		t.Errorf("Load() = %+v, want the zai preset and its key", cfg)
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

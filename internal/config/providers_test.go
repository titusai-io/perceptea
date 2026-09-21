package config

import (
	"strings"
	"testing"
)

func TestProvidersTable(t *testing.T) {
	want := []Provider{
		{"openai", "https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"},
		{"openrouter", "https://openrouter.ai/api/v1", "z-ai/glm-5.3-flash", "OPENROUTER_API_KEY"},
		{"deepinfra", "https://api.deepinfra.com/v1/openai", "zai-org/GLM-5.3-Flash", "DEEPINFRA_API_KEY"},
		{"zai", "https://api.z.ai/api/paas/v4", "glm-5.3-flash", "ZAI_API_KEY"},
	}
	got := Providers()
	if len(got) != len(want) {
		t.Fatalf("Providers() returned %d presets, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("preset %d:\n got %+v\nwant %+v", i, got[i], want[i])
		}
	}
}

func TestProvidersReturnsACopy(t *testing.T) {
	first := Providers()
	first[0].BaseURL = "http://tampered.invalid"
	if Providers()[0].BaseURL == "http://tampered.invalid" {
		t.Error("Providers() handed out the package's own slice")
	}
}

func TestLookupIsCaseInsensitive(t *testing.T) {
	for _, name := range []string{"openai", "OpenAI", "OPENAI", "  openai  "} {
		p, ok := Lookup(name)
		if !ok {
			t.Fatalf("Lookup(%q) = false, want the openai preset", name)
		}
		if p.Name != "openai" || p.EnvKey != "OPENAI_API_KEY" {
			t.Errorf("Lookup(%q) = %+v, want the openai preset", name, p)
		}
	}
	for _, name := range []string{"DeepInfra", "ZAI", "OpenRouter"} {
		if _, ok := Lookup(name); !ok {
			t.Errorf("Lookup(%q) = false, want a preset", name)
		}
	}
}

func TestLookupUnknown(t *testing.T) {
	p, ok := Lookup("nope")
	if ok {
		t.Errorf("Lookup(\"nope\") = %+v, true; want false", p)
	}
	if p != (Provider{}) {
		t.Errorf("Lookup of an unknown provider returned %+v, want the zero value", p)
	}
	if _, ok := Lookup(""); ok {
		t.Error("Lookup(\"\") = true, want false")
	}
}

func TestProviderNamesListsEveryPreset(t *testing.T) {
	names := providerNames()
	for _, p := range Providers() {
		if !strings.Contains(names, p.Name) {
			t.Errorf("providerNames() = %q, missing %q", names, p.Name)
		}
	}
}

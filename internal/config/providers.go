// Package config resolves the Perceptea server's settings from a small set of
// provider presets and the process environment.
//
// Nothing here reaches the network, and nothing here reads a file except
// [LoadDotEnv], which is a best-effort convenience for local development.
package config

import "strings"

// Provider is a preset for one OpenAI-compatible endpoint: where it lives,
// which model to use when the caller does not say, and the conventional
// environment variable its API key is read from.
type Provider struct {
	// Name is the value PERCEPTEA_PROVIDER is matched against, lowercase.
	Name string
	// BaseURL is the OpenAI-compatible API root, without a trailing slash.
	BaseURL string
	// DefaultModel is used when PERCEPTEA_MODEL is unset and the request body
	// does not name a model.
	DefaultModel string
	// EnvKey is the provider's conventional API key variable, consulted after
	// PERCEPTEA_API_KEY.
	EnvKey string
}

// presets is the built-in provider table. It is package-private so that
// [Providers] can hand out a copy: a preset is a value, not shared state.
var presets = []Provider{
	{
		Name:         "openai",
		BaseURL:      "https://api.openai.com/v1",
		DefaultModel: "gpt-4o-mini",
		EnvKey:       "OPENAI_API_KEY",
	},
	{
		Name:         "openrouter",
		BaseURL:      "https://openrouter.ai/api/v1",
		DefaultModel: "z-ai/glm-5.3-flash",
		EnvKey:       "OPENROUTER_API_KEY",
	},
	{
		// The DeepInfra model id is a best-effort guess at the provider's
		// naming; override it with PERCEPTEA_MODEL if it is wrong.
		Name:         "deepinfra",
		BaseURL:      "https://api.deepinfra.com/v1/openai",
		DefaultModel: "zai-org/GLM-5.3-Flash",
		EnvKey:       "DEEPINFRA_API_KEY",
	},
	{
		Name:         "zai",
		BaseURL:      "https://api.z.ai/api/paas/v4",
		DefaultModel: "glm-5.3-flash",
		EnvKey:       "ZAI_API_KEY",
	},
}

// Providers returns the built-in presets, in a stable order. The returned
// slice is a copy; mutating it does not affect later calls.
func Providers() []Provider {
	out := make([]Provider, len(presets))
	copy(out, presets)
	return out
}

// Lookup finds a preset by name, ignoring case and surrounding whitespace.
func Lookup(name string) (Provider, bool) {
	name = strings.TrimSpace(name)
	for _, p := range presets {
		if strings.EqualFold(p.Name, name) {
			return p, true
		}
	}
	return Provider{}, false
}

// providerNames lists the known preset names, for error messages.
func providerNames() string {
	names := make([]string, 0, len(presets))
	for _, p := range presets {
		names = append(names, p.Name)
	}
	return strings.Join(names, ", ")
}

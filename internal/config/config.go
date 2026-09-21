// Package config resolves the Perceptea server's settings from the process
// environment.
//
// Nothing here reaches the network, and nothing here reads a file except
// [LoadDotEnv], which is a best-effort convenience for local development.
package config

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"
)

// The environment variables Perceptea reads. Every one of them is optional.
const (
	EnvAddr                    = "PERCEPTEA_ADDR"
	EnvInferenceBaseURL        = "PERCEPTEA_INFERENCE_BASE_URL"
	EnvAPIKey                  = "PERCEPTEA_API_KEY"
	EnvModel                   = "PERCEPTEA_MODEL"
	EnvReasoningEffort         = "PERCEPTEA_REASONING_EFFORT"
	EnvTemperature             = "PERCEPTEA_TEMPERATURE"
	EnvMaxConcurrency          = "PERCEPTEA_MAX_CONCURRENCY"
	EnvRequestTimeout          = "PERCEPTEA_REQUEST_TIMEOUT"
	EnvMaxBodyBytes            = "PERCEPTEA_MAX_BODY_BYTES"
	EnvAllowRequestCredentials = "PERCEPTEA_ALLOW_REQUEST_CREDENTIALS"
	EnvLogLevel                = "PERCEPTEA_LOG_LEVEL"
	EnvLogFormat               = "PERCEPTEA_LOG_FORMAT"
)

// Defaults applied when a variable is unset or empty.
const (
	DefaultAddr = ":8080"
	// DefaultInferenceBaseURL is the API root the scoring calls go to when
	// PERCEPTEA_INFERENCE_BASE_URL is unset. It is an API root, not an
	// endpoint: the client appends /chat/completions to it.
	DefaultInferenceBaseURL = "https://api.deepinfra.com/v1/openai"
	// DefaultModel is the model scored when neither PERCEPTEA_MODEL nor the
	// request body names one. It is a model the endpoint above publishes,
	// and it is chosen for the shape of this workload rather than for its
	// size: it does not reason, so a 32 token scoring reply is never spent
	// on thinking tokens; it supports the JSON schema response format the
	// scorer asks for first; and input tokens are what a fan-out of small
	// calls is billed for, which is where it is cheapest. See the README's
	// "Choosing a model".
	DefaultModel = "Qwen/Qwen3.8-Flash"
	// DefaultTemperature is 0 because it is the only setting that makes a
	// classification reproducible.
	DefaultTemperature = 0.0
	// DefaultMaxConcurrency bounds the micro-calls one evaluation may have in
	// flight at once.
	DefaultMaxConcurrency = 8
	// DefaultRequestTimeout bounds one whole /api/evaluate request.
	DefaultRequestTimeout = 60 * time.Second
	// DefaultMaxBodyBytes is 2 MiB: far more than a state and a question set
	// plausibly need, and still small enough that a body this size cannot be
	// used to tie the server up.
	DefaultMaxBodyBytes = int64(2 * 1024 * 1024)
	// DefaultAllowRequestCredentials lets a request carry its own key, which
	// is what makes the service usable straight out of the box. Turn it off
	// outside development.
	DefaultAllowRequestCredentials = true
	// DefaultLogLevel is info.
	DefaultLogLevel = slog.LevelInfo
	// DefaultLogFormat is human-readable text.
	DefaultLogFormat = LogFormatText
)

// Log formats understood by PERCEPTEA_LOG_FORMAT.
const (
	LogFormatText = "text"
	LogFormatJSON = "json"
)

// Reasoning efforts understood by PERCEPTEA_REASONING_EFFORT. They are the
// values the OpenAI-compatible reasoning_effort field takes. Unset is not one
// of them: it means the field is not sent at all.
const (
	EffortNone   = "none"
	EffortLow    = "low"
	EffortMedium = "medium"
	EffortHigh   = "high"
)

// ReasoningEfforts lists the accepted values in the order an error message
// should offer them.
func ReasoningEfforts() []string {
	return []string{EffortNone, EffortLow, EffortMedium, EffortHigh}
}

// Config is the resolved server configuration.
type Config struct {
	// Addr is the listen address, as accepted by net.Listen.
	Addr string
	// BaseURL is the inference API root the scoring calls actually go to: the
	// root of the service that runs the model, which the client turns into
	// <BaseURL>/chat/completions. PERCEPTEA_INFERENCE_BASE_URL sets it.
	BaseURL string
	// APIKey is the server's own key, from PERCEPTEA_API_KEY. It may be empty:
	// the server still starts, and requests that cannot supply a key are
	// rejected with 401.
	APIKey string
	// Model is the default model for requests that do not name one.
	Model string
	// ReasoningEffort is sent to the provider as reasoning_effort on every
	// scoring and generate call. Empty — the default — sends no reasoning
	// field at all, which is what a provider that has never seen one
	// expects. It is server-side only: a request body cannot override it,
	// because it is a property of the model the operator chose and not of
	// the question being asked.
	ReasoningEffort string
	// Temperature is the default sampling temperature.
	Temperature float64
	// MaxConcurrency bounds in-flight scoring calls per evaluation.
	MaxConcurrency int
	// RequestTimeout bounds one HTTP request end to end.
	RequestTimeout time.Duration
	// MaxBodyBytes bounds a request body.
	MaxBodyBytes int64
	// AllowRequestCredentials lets a request body override api_key and
	// inference_base_url. Exposing the service with this on makes it an open
	// proxy; turning it off is the whole answer, and there is no partial one.
	AllowRequestCredentials bool
	// LogLevel is the minimum level logged.
	LogLevel slog.Level
	// LogFormat is either LogFormatText or LogFormatJSON.
	LogFormat string
}

// APIKeyConfigured reports whether the server holds a key of its own. It never
// exposes the key itself.
func (c Config) APIKeyConfigured() bool { return c.APIKey != "" }

// quotedList renders values as `"a", "b" or "c"` for an error message.
func quotedList(values []string) string {
	quoted := make([]string, 0, len(values))
	for _, v := range values {
		quoted = append(quoted, strconv.Quote(v))
	}
	switch len(quoted) {
	case 0:
		return ""
	case 1:
		return quoted[0]
	default:
		return strings.Join(quoted[:len(quoted)-1], ", ") + " or " + quoted[len(quoted)-1]
	}
}

// Load reads the configuration from the process environment.
func Load() (Config, error) { return LoadFrom(os.Getenv) }

// LoadFrom reads the configuration through env, which behaves like
// [os.Getenv]: it returns the empty string for a variable that is not set. It
// exists so that configuration can be tested without touching the process.
//
// A missing API key is not an error: the server is allowed to start without
// one and reject the requests that need it.
func LoadFrom(env func(string) string) (Config, error) {
	if env == nil {
		env = func(string) string { return "" }
	}
	get := func(key string) string { return strings.TrimSpace(env(key)) }

	cfg := Config{
		Addr:                    DefaultAddr,
		BaseURL:                 DefaultInferenceBaseURL,
		Model:                   DefaultModel,
		Temperature:             DefaultTemperature,
		MaxConcurrency:          DefaultMaxConcurrency,
		RequestTimeout:          DefaultRequestTimeout,
		MaxBodyBytes:            DefaultMaxBodyBytes,
		AllowRequestCredentials: DefaultAllowRequestCredentials,
		LogLevel:                DefaultLogLevel,
		LogFormat:               DefaultLogFormat,
	}

	if v := get(EnvAddr); v != "" {
		cfg.Addr = v
	}

	if v := get(EnvInferenceBaseURL); v != "" {
		cfg.BaseURL = v
	}
	// A base URL that cannot be called makes every request fail with a 500
	// while /api/health still reports ok, so it is a startup error like any
	// other malformed setting. The message names the variable and not the
	// value: a base URL may carry a credential.
	if err := ValidateBaseURL(cfg.BaseURL); err != nil {
		return Config{}, fmt.Errorf("%s %w", EnvInferenceBaseURL, err)
	}
	if v := get(EnvModel); v != "" {
		cfg.Model = v
	}

	// An effort the provider would reject is a 400 on every scoring call of
	// every request, so it is caught here like any other malformed setting.
	if v := get(EnvReasoningEffort); v != "" {
		effort := strings.ToLower(v)
		switch effort {
		case EffortNone, EffortLow, EffortMedium, EffortHigh:
			cfg.ReasoningEffort = effort
		default:
			return Config{}, fmt.Errorf("%s: %q is not a reasoning effort; expected %s",
				EnvReasoningEffort, v, quotedList(ReasoningEfforts()))
		}
	}

	// PERCEPTEA_API_KEY is the only variable a key is read from. Nothing here
	// consults a vendor-specific one: a key picked up from a variable the
	// operator did not point at this service is a credential spent by
	// accident.
	cfg.APIKey = get(EnvAPIKey)

	if v := get(EnvTemperature); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return Config{}, fmt.Errorf("%s: %q is not a number", EnvTemperature, v)
		}
		cfg.Temperature = f
	}

	if v := get(EnvMaxConcurrency); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("%s: %q is not a whole number", EnvMaxConcurrency, v)
		}
		if n < 1 {
			return Config{}, fmt.Errorf("%s: must be at least 1, got %d", EnvMaxConcurrency, n)
		}
		cfg.MaxConcurrency = n
	}

	if v := get(EnvRequestTimeout); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil {
			return Config{}, fmt.Errorf("%s: %q is not a duration such as %q", EnvRequestTimeout, v, "60s")
		}
		if d <= 0 {
			return Config{}, fmt.Errorf("%s: must be positive, got %s", EnvRequestTimeout, d)
		}
		cfg.RequestTimeout = d
	}

	if v := get(EnvMaxBodyBytes); v != "" {
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return Config{}, fmt.Errorf("%s: %q is not a whole number of bytes", EnvMaxBodyBytes, v)
		}
		if n < 1 {
			return Config{}, fmt.Errorf("%s: must be at least 1, got %d", EnvMaxBodyBytes, n)
		}
		cfg.MaxBodyBytes = n
	}

	if v := get(EnvAllowRequestCredentials); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return Config{}, fmt.Errorf("%s: %q is not a boolean such as %q or %q", EnvAllowRequestCredentials, v, "true", "false")
		}
		cfg.AllowRequestCredentials = b
	}

	if v := get(EnvLogLevel); v != "" {
		var lvl slog.Level
		if err := lvl.UnmarshalText([]byte(v)); err != nil {
			return Config{}, fmt.Errorf("%s: %q is not a level such as %q, %q, %q or %q", EnvLogLevel, v, "debug", "info", "warn", "error")
		}
		cfg.LogLevel = lvl
	}

	if v := get(EnvLogFormat); v != "" {
		switch strings.ToLower(v) {
		case LogFormatText:
			cfg.LogFormat = LogFormatText
		case LogFormatJSON:
			cfg.LogFormat = LogFormatJSON
		default:
			return Config{}, fmt.Errorf("%s: %q is not a format; expected %q or %q", EnvLogFormat, v, LogFormatText, LogFormatJSON)
		}
	}

	return cfg, nil
}

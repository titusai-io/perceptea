package main

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/titusai-io/perceptea/internal/config"
)

func TestRunRejectsABadConfiguration(t *testing.T) {
	t.Setenv("PERCEPTEA_INFERENCE_BASE_URL", "not a url")

	err := run(nil)
	if err == nil {
		t.Fatal("run() = nil, want a startup error")
	}
	if !strings.Contains(err.Error(), "PERCEPTEA_INFERENCE_BASE_URL") {
		t.Errorf("error %q does not name the variable at fault", err)
	}
}

// TestRunPrefersTheAddrFlag points the environment at an address that can
// never be bound and the flag at one that is already taken. Whichever error
// comes back says which of the two the server actually tried.
func TestRunPrefersTheAddrFlag(t *testing.T) {
	taken, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("taking a port: %v", err)
	}
	defer taken.Close()

	t.Setenv("PERCEPTEA_ADDR", "256.256.256.256:8080")

	err = run([]string{"-addr", taken.Addr().String()})
	if err == nil {
		t.Fatal("run() = nil, want a listen error")
	}
	if !strings.Contains(err.Error(), taken.Addr().String()) {
		t.Errorf("error %q, want the flag's address to have been used", err)
	}
}

// The three listener-side timeouts are caller-visible — a body has 30 seconds
// to arrive, a connection two minutes to be reused — and all three could be
// set to a millisecond with the suite still green.
func TestNewHTTPServerTimeouts(t *testing.T) {
	cfg := config.Config{Addr: "127.0.0.1:0", RequestTimeout: 90 * time.Second}
	srv := newHTTPServer(cfg, http.NotFoundHandler(), slog.New(slog.DiscardHandler))

	if srv.Addr != cfg.Addr {
		t.Errorf("Addr = %q, want %q", srv.Addr, cfg.Addr)
	}
	if srv.Handler == nil {
		t.Error("Handler is nil")
	}
	if srv.ReadHeaderTimeout != 10*time.Second {
		t.Errorf("ReadHeaderTimeout = %s, want 10s", srv.ReadHeaderTimeout)
	}
	if srv.ReadTimeout != 30*time.Second {
		t.Errorf("ReadTimeout = %s, want 30s", srv.ReadTimeout)
	}
	if srv.IdleTimeout != 120*time.Second {
		t.Errorf("IdleTimeout = %s, want 120s", srv.IdleTimeout)
	}
	// The write timeout has to clear a whole evaluation, or the socket closes
	// under a request that was still allowed to be running.
	if want := cfg.RequestTimeout + 30*time.Second; srv.WriteTimeout != want {
		t.Errorf("WriteTimeout = %s, want %s", srv.WriteTimeout, want)
	}
	if srv.WriteTimeout <= cfg.RequestTimeout {
		t.Errorf("WriteTimeout = %s, which cuts off a request the server still allows", srv.WriteTimeout)
	}
	if srv.ErrorLog == nil {
		t.Error("ErrorLog is nil: net/http would write to the standard logger")
	}
}

// The grace was capped at two minutes: with a ten-minute request timeout,
// SIGTERM cut an in-flight request off and the process exited 1.
func TestShutdownGraceIsNeverShorterThanARequest(t *testing.T) {
	for _, tc := range []struct {
		timeout, want time.Duration
	}{
		{10 * time.Minute, 10*time.Minute + shutdownSlack},
		{5 * time.Minute, 5*time.Minute + shutdownSlack},
		{60 * time.Second, 60*time.Second + shutdownSlack},
		{time.Second, minShutdownWait},
		{0, minShutdownWait},
	} {
		got := shutdownGrace(tc.timeout)
		if got != tc.want {
			t.Errorf("shutdownGrace(%s) = %s, want %s", tc.timeout, got, tc.want)
		}
		if got < tc.timeout {
			t.Errorf("shutdownGrace(%s) = %s, shorter than a request is allowed to be", tc.timeout, got)
		}
	}
}

func TestStartupLineCarriesNoCredential(t *testing.T) {
	cfg := config.Config{
		BaseURL:        "https://svc:sk-secret-0123456789@gw.example/v1?key=abc123",
		APIKey:         "sk-server-0123456789",
		Model:          "probe-1",
		RequestTimeout: 60 * time.Second,
		MaxConcurrency: 8,
	}

	var buf bytes.Buffer
	logger := newLogger(config.Config{LogFormat: config.LogFormatJSON, LogLevel: slog.LevelInfo}, &buf)
	logger.LogAttrs(context.Background(), slog.LevelInfo, "perceptea listening", startupAttrs(cfg, "[::]:8080")...)

	line := buf.String()
	for _, secret := range []string{"sk-secret-0123456789", "svc:", "key=abc123", cfg.APIKey} {
		if strings.Contains(line, secret) {
			t.Errorf("the startup line contains %q:\n%s", secret, line)
		}
	}
	var entry map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(line)), &entry); err != nil {
		t.Fatalf("the line is not JSON: %v", err)
	}
	if entry["inference_base_url"] != "https://gw.example/v1" {
		t.Errorf("inference_base_url = %v, want the endpoint without its credential", entry["inference_base_url"])
	}
	// The preset the endpoint used to be chosen from is gone, and so is the
	// field that named it.
	if _, ok := entry["provider"]; ok {
		t.Errorf("the startup line still reports a provider: %s", line)
	}
	if entry["api_key_configured"] != true {
		t.Errorf("api_key_configured = %v, want true", entry["api_key_configured"])
	}
	if entry["addr"] != "[::]:8080" {
		t.Errorf("addr = %v", entry["addr"])
	}
}

// The startup line is what the operator reads to check the process will do
// what they configured. A reasoning effort changes every request body, so it
// is on the line when it is set — and absent when it is not, because the
// field is then not sent at all and reporting it would be a line to be
// misled by.
func TestStartupLineReportsTheReasoningEffortOnlyWhenSet(t *testing.T) {
	for _, tc := range []struct {
		name    string
		effort  string
		wantSet bool
	}{
		{"unset", "", false},
		{"none", config.EffortNone, true},
		{"high", config.EffortHigh, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := config.Config{
				BaseURL:         "https://gw.example/v1",
				Model:           "probe-1",
				ReasoningEffort: tc.effort,
				RequestTimeout:  60 * time.Second,
				MaxConcurrency:  8,
			}

			var buf bytes.Buffer
			logger := newLogger(config.Config{LogFormat: config.LogFormatJSON, LogLevel: slog.LevelInfo}, &buf)
			logger.LogAttrs(context.Background(), slog.LevelInfo, "perceptea listening", startupAttrs(cfg, "[::]:8080")...)

			var entry map[string]any
			if err := json.Unmarshal([]byte(strings.TrimSpace(buf.String())), &entry); err != nil {
				t.Fatalf("the line is not JSON: %v", err)
			}
			got, present := entry["reasoning_effort"]
			if present != tc.wantSet {
				t.Fatalf("reasoning_effort present = %v, want %v: %s", present, tc.wantSet, buf.String())
			}
			if tc.wantSet && got != tc.effort {
				t.Errorf("reasoning_effort = %v, want %q", got, tc.effort)
			}
			// The rest of the line is unaffected either way.
			if entry["model"] != "probe-1" {
				t.Errorf("model = %v", entry["model"])
			}
			if entry["max_concurrency"] != float64(8) {
				t.Errorf("max_concurrency = %v", entry["max_concurrency"])
			}
		})
	}
}

func TestNewLogger(t *testing.T) {
	var buf bytes.Buffer

	jsonLogger := newLogger(config.Config{LogFormat: config.LogFormatJSON, LogLevel: slog.LevelWarn}, &buf)
	jsonLogger.Info("dropped")
	jsonLogger.Warn("kept", slog.String("k", "v"))

	var entry map[string]any
	line := strings.TrimSpace(buf.String())
	if strings.Contains(line, "dropped") {
		t.Errorf("the level was not applied: %s", line)
	}
	if err := json.Unmarshal([]byte(line), &entry); err != nil {
		t.Fatalf("the json format did not produce JSON (%q): %v", line, err)
	}
	if entry["msg"] != "kept" || entry["k"] != "v" {
		t.Errorf("entry = %v", entry)
	}

	buf.Reset()
	textLogger := newLogger(config.Config{LogFormat: config.LogFormatText, LogLevel: slog.LevelInfo}, &buf)
	textLogger.Info("hello", slog.String("k", "v"))
	if got := buf.String(); !strings.Contains(got, "msg=hello") || !strings.Contains(got, "k=v") {
		t.Errorf("text output = %q", got)
	}
	if json.Valid([]byte(strings.TrimSpace(buf.String()))) {
		t.Error("the text format produced JSON")
	}
}

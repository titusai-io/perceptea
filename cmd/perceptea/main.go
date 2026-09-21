// Command perceptea serves the Perceptea classifier API over HTTP.
//
// Configuration comes from the environment (see internal/config); a .env file
// in the working directory is read first as a development convenience and
// never overrides a variable that is already set.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/titusai-io/perceptea/api"
	"github.com/titusai-io/perceptea/internal/config"
)

// version is stamped at build time with -ldflags "-X main.version=...". It
// stays "dev" for an ordinary go build, which is the honest answer: nothing
// outside a release pipeline knows what to call the binary.
var version = "dev"

// dotEnvPath is the optional file read before the environment is examined.
const dotEnvPath = ".env"

// Server timeouts. The write timeout has to clear a whole evaluation, which
// may take up to the configured request timeout, with room for the response
// itself; the others are ordinary hygiene against a slow or idle peer. The
// read timeout is caller-visible — a body still arriving after it gets a 504 —
// and is documented in the README alongside the rest.
const (
	readHeaderTimeout = 10 * time.Second
	readTimeout       = 30 * time.Second
	writeTimeoutSlack = 30 * time.Second
	idleTimeout       = 120 * time.Second
	shutdownSlack     = 5 * time.Second
	// minShutdownWait is the floor under the drain: with a tiny configured
	// request timeout, there is still a connection to let go of tidily.
	minShutdownWait = 30 * time.Second
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "perceptea: %v\n", err)
		os.Exit(1)
	}
}

// run starts the server and returns once it has stopped, or immediately with
// the reason it could not start.
func run(args []string) error {
	flags := flag.NewFlagSet("perceptea", flag.ContinueOnError)
	addr := flags.String("addr", "", "listen address, overriding "+config.EnvAddr)
	check := flags.Bool("healthcheck", false, "probe a running instance's /api/health and exit")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *check {
		return healthcheck(*addr)
	}

	// A .env file is a convenience, not a configuration source of record: a
	// real environment variable always wins, and a missing file is normal.
	if err := config.LoadDotEnv(dotEnvPath); err != nil {
		fmt.Fprintf(os.Stderr, "perceptea: ignoring %s: %v\n", dotEnvPath, err)
	}

	cfg, err := config.Load()
	if err != nil {
		return err
	}
	if *addr != "" {
		cfg.Addr = *addr
	}

	logger := newLogger(cfg, os.Stderr)
	slog.SetDefault(logger)

	srv, err := api.NewServer(api.Options{Config: cfg, Logger: logger})
	if err != nil {
		return fmt.Errorf("building the server: %w", err)
	}

	httpServer := newHTTPServer(cfg, srv.Handler(), logger)

	// Listen before announcing anything so that a port clash is a startup
	// error rather than a line in the log followed by silence.
	listener, err := net.Listen("tcp", cfg.Addr)
	if err != nil {
		return fmt.Errorf("listening on %s: %w", cfg.Addr, err)
	}

	logger.LogAttrs(context.Background(), slog.LevelInfo, "perceptea listening",
		startupAttrs(cfg, listener.Addr().String())...)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	serveErr := make(chan error, 1)
	go func() { serveErr <- httpServer.Serve(listener) }()

	select {
	case err := <-serveErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	case <-ctx.Done():
	}

	// Stop catching signals so that a second one kills the process outright
	// instead of waiting behind the drain.
	stop()
	grace := shutdownGrace(cfg.RequestTimeout)
	logger.Info("shutting down", slog.Duration("grace", grace))

	shutdownCtx, cancel := context.WithTimeout(context.Background(), grace)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("shutdown: %w", err)
	}
	if err := <-serveErr; err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	logger.Info("stopped")
	return nil
}

// startupAttrs is the one line that says what this process will do. It never
// carries a credential: not the key, and not a base URL that has one in it —
// a gateway URL routinely does, in its userinfo or its query, and a log is a
// file somebody else reads.
func startupAttrs(cfg config.Config, addr string) []slog.Attr {
	attrs := []slog.Attr{
		slog.String("version", version),
		slog.String("addr", addr),
		slog.String("inference_base_url", config.DisplayBaseURL(cfg.BaseURL)),
		slog.String("model", cfg.Model),
	}
	// Only when it is set. An unset effort sends no reasoning field at all,
	// and a line reporting a field that never goes on the wire would be a
	// line to debug from and be misled by.
	if cfg.ReasoningEffort != "" {
		attrs = append(attrs, slog.String("reasoning_effort", cfg.ReasoningEffort))
	}
	return append(attrs,
		slog.Bool("api_key_configured", cfg.APIKeyConfigured()),
		slog.Bool("allow_request_credentials", cfg.AllowRequestCredentials),
		slog.Duration("request_timeout", cfg.RequestTimeout),
		slog.Int("max_concurrency", cfg.MaxConcurrency),
	)
}

// newHTTPServer wires the listener-side timeouts. The write timeout has to
// clear a whole evaluation, which may take up to the configured request
// timeout, with room for the response itself.
func newHTTPServer(cfg config.Config, handler http.Handler, logger *slog.Logger) *http.Server {
	return &http.Server{
		Addr:              cfg.Addr,
		Handler:           handler,
		ReadHeaderTimeout: readHeaderTimeout,
		ReadTimeout:       readTimeout,
		WriteTimeout:      cfg.RequestTimeout + writeTimeoutSlack,
		IdleTimeout:       idleTimeout,
		ErrorLog:          slog.NewLogLogger(logger.Handler(), slog.LevelWarn),
	}
}

// shutdownGrace is how long a drain may take.
//
// It is never shorter than a request is allowed to be: capping it cut an
// in-flight request off on SIGTERM and turned a clean stop into exit 1, which
// is the opposite of what a graceful shutdown is for. It is a ceiling, not a
// wait — Shutdown returns as soon as the last connection goes idle — so a
// generous one costs nothing.
func shutdownGrace(requestTimeout time.Duration) time.Duration {
	return max(requestTimeout+shutdownSlack, minShutdownWait)
}

// newLogger builds the logger the configuration asks for.
func newLogger(cfg config.Config, w io.Writer) *slog.Logger {
	opts := &slog.HandlerOptions{Level: cfg.LogLevel}
	if cfg.LogFormat == config.LogFormatJSON {
		return slog.New(slog.NewJSONHandler(w, opts))
	}
	return slog.New(slog.NewTextHandler(w, opts))
}

// healthcheckTimeout bounds the probe. A container runtime gives a healthcheck
// its own deadline; this one only has to be shorter than that.
const healthcheckTimeout = 5 * time.Second

// healthcheck probes a running instance and reports whether it is serving.
//
// It exists so that a container image needs no shell, no curl and no wget:
// the binary is already in the image, so it can check itself. The address
// comes from -addr, then the ordinary configuration, so the probe follows the
// port the server was told to use rather than assuming one.
//
// Only the status matters. A body that says ok:false is not something this
// server produces, and inventing a second meaning for the payload here would
// put the probe and the endpoint out of step the first time either changed.
func healthcheck(addr string) error {
	if addr == "" {
		cfg, err := config.Load()
		if err != nil {
			return err
		}
		addr = cfg.Addr
	}
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("healthcheck: %q is not a listen address: %w", addr, err)
	}
	// A server bound to every interface is reached over the loopback, which
	// is the only one a probe inside the same container can rely on.
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}

	ctx, cancel := context.WithTimeout(context.Background(), healthcheckTimeout)
	defer cancel()

	url := "http://" + net.JoinHostPort(host, port) + "/api/health"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("healthcheck: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("healthcheck: %w", err)
	}
	defer func() {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("healthcheck: %s answered %d", url, resp.StatusCode)
	}
	return nil
}

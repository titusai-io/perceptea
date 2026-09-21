// Package api is Perceptea's HTTP layer: it turns an evaluation request into
// a [classifier.Request], hands it to an evaluator, and renders the typed
// answers — or a typed error — as JSON.
//
// Everything that reaches the network sits behind [Options.NewEvaluator], so
// the whole package can be exercised without a provider and without a socket.
package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"runtime/debug"
	"strconv"
	"time"
	"unicode/utf8"

	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
)

// Evaluator answers one request's worth of questions. It is the subset of
// *classifier.Evaluator this package needs, so a test can supply its own.
type Evaluator interface {
	Evaluate(ctx context.Context, req classifier.Request) (classifier.Response, error)
}

// Settings are the per-request credentials and limits an evaluator is built
// with. A request may carry its own key and base URL when the server allows
// it, so an evaluator cannot simply be built once at startup.
type Settings struct {
	// APIKey is the resolved provider key. It may be empty, in which case
	// building the evaluator is expected to fail with openai.ErrNoAPIKey.
	APIKey string
	// BaseURL is the resolved inference API root: the root of the service
	// that runs the model, which the client turns into
	// <BaseURL>/chat/completions.
	BaseURL string
	// Model is the resolved model id.
	Model string
	// MaxConcurrency bounds the scoring calls one evaluation runs at once.
	MaxConcurrency int
}

// Options configures [NewServer].
type Options struct {
	// Config is the resolved server configuration. Zero-valued limits are
	// replaced by their defaults.
	Config config.Config
	// Logger receives one line per request. Nil means slog.Default().
	Logger *slog.Logger
	// NewEvaluator builds the evaluator for one request. Nil means the real
	// one: a provider/openai client behind classifier.New, sharing a single
	// HTTP client across requests.
	NewEvaluator func(Settings) (Evaluator, error)
}

// Server serves the Perceptea HTTP API.
type Server struct {
	cfg          config.Config
	log          *slog.Logger
	newEvaluator func(Settings) (Evaluator, error)
	handler      http.Handler
}

// NewServer builds a server from opts. It does not listen; see [Server.Handler].
func NewServer(opts Options) (*Server, error) {
	cfg := opts.Config
	if cfg.MaxConcurrency < 1 {
		cfg.MaxConcurrency = config.DefaultMaxConcurrency
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = config.DefaultRequestTimeout
	}
	if cfg.MaxBodyBytes < 1 {
		cfg.MaxBodyBytes = config.DefaultMaxBodyBytes
	}

	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}

	s := &Server{
		cfg:          cfg,
		log:          logger,
		newEvaluator: opts.NewEvaluator,
	}
	if s.newEvaluator == nil {
		s.newEvaluator = defaultNewEvaluator(logger)
	}

	// Every route is registered twice: once with its method, and once as a
	// bare path that catches every other method. The mux's own 405 answers in
	// text/plain, and a client that is promised one error shape for every
	// failure should not have to parse prose for two of them. With the
	// path-only patterns in place nothing depends on the automatic 405 any
	// more, so a "/" catch-all can render the 404 in the same shape.
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/evaluate", s.handleEvaluate)
	mux.HandleFunc("/api/evaluate", s.methodNotAllowed(http.MethodPost))
	mux.HandleFunc("GET /api/health", s.handleHealth)
	mux.HandleFunc("/api/health", s.methodNotAllowed(http.MethodGet))
	mux.HandleFunc("/", s.handleNotFound)
	s.handler = s.withRequestScope(mux)

	return s, nil
}

// methodNotAllowed answers a request that reached a known path by the wrong
// method. The allowed method is advertised in the Allow header, as RFC 9110
// requires, and the body is the same shape as every other failure's.
func (s *Server) methodNotAllowed(allowed string) http.HandlerFunc {
	message := "this endpoint only accepts " + allowed
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Allow", allowed)
		s.writeError(w, r, http.StatusMethodNotAllowed, codeMethodNotAllowed, message)
	}
}

// handleNotFound answers a path this service does not serve. The message does
// not echo the path: a caller's own text does not belong in a response body.
func (s *Server) handleNotFound(w http.ResponseWriter, r *http.Request) {
	s.writeError(w, r, http.StatusNotFound, codeNotFound, "no such endpoint")
}

// Config reports the configuration the server resolved, limits included. The
// API key is part of it; do not render this anywhere a caller can see.
func (s *Server) Config() config.Config { return s.cfg }

// Handler returns the routed handler: the mux above, wrapped so that every
// request gets a deadline and exactly one log line.
func (s *Server) Handler() http.Handler { return s.handler }

// requestScope carries what the log line needs but only the handler knows.
type requestScope struct {
	questions int
	mode      string
	// secret is the API key resolved for this request, so that it can be
	// scrubbed from anything on its way out.
	secret string
	// baseURL is the endpoint resolved for this request. It is here for the
	// same reason: a gateway URL can hold a credential of its own.
	baseURL string
}

// secrets lists everything that must not appear in a message or a log line
// for this request.
func (s *Server) secrets(scope *requestScope) []string {
	out := []string{s.cfg.APIKey, scope.secret}
	out = append(out, config.CredentialsIn(s.cfg.BaseURL)...)
	if scope.baseURL != "" && scope.baseURL != s.cfg.BaseURL {
		out = append(out, config.CredentialsIn(scope.baseURL)...)
	}
	return out
}

type scopeKey struct{}

// scopeFrom returns the current request's scope, or a throwaway one if the
// handler was called without the middleware.
func scopeFrom(ctx context.Context) *requestScope {
	if sc, ok := ctx.Value(scopeKey{}).(*requestScope); ok {
		return sc
	}
	return &requestScope{}
}

// withRequestScope bounds every request by RequestTimeout, records the status
// it ends with, and logs one structured line. The state and the questions are
// never logged: they are the caller's data.
func (s *Server) withRequestScope(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout)
		defer cancel()
		scope := &requestScope{}
		ctx = context.WithValue(ctx, scopeKey{}, scope)

		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		// The scoped request is what the handler sees, and what the recovery
		// below answers with: anything it writes is scrubbed against this
		// request's own credentials, not an empty scope's.
		scoped := r.WithContext(ctx)

		// The log line is written from a defer so that a handler that panics
		// still produces one — after the panic has been turned into the 500
		// the caller was promised. net/http would otherwise keep the process
		// alive and leave the caller with a bare EOF and nothing in the log.
		defer func() {
			if v := recover(); v != nil {
				// http.ErrAbortHandler is the one panic value that means
				// "drop this connection deliberately"; net/http expects to
				// see it itself, and suppresses its own stack for it. The log
				// line is written first so that the request is still counted.
				if v == http.ErrAbortHandler {
					s.logRequest(ctx, r, rec, scope, start)
					panic(v)
				}
				s.failPanic(rec, scoped, v)
			}
			s.logRequest(ctx, r, rec, scope, start)
		}()

		next.ServeHTTP(rec, scoped)
	})
}

// failPanic turns a recovered panic into the documented 500 and keeps the
// stack for the log. A handler that had already started writing gets no
// second body: the status is on the wire and cannot be taken back.
func (s *Server) failPanic(rec *statusRecorder, r *http.Request, v any) {
	rec.failed = true
	note(rec, fmt.Sprintf("panic: %v\n%s", v, debug.Stack()))
	if rec.written {
		return
	}
	s.writeError(rec, r, http.StatusInternalServerError, codeInternal, "internal error")
}

// logRequest writes the one line a request is worth.
func (s *Server) logRequest(ctx context.Context, r *http.Request, rec *statusRecorder, scope *requestScope, start time.Time) {
	attrs := []slog.Attr{
		slog.String("method", r.Method),
		// The path is the caller's text; a log line is not the place to copy
		// a kilobyte of it.
		slog.String("path", clip(r.URL.Path, maxLoggedValue)),
		slog.Int("status", rec.status),
		// Rounded to the microsecond: an evaluation is measured in
		// hundreds of milliseconds, but a rejected request is not, and
		// "0s" in the log tells nobody anything.
		slog.Duration("duration", time.Since(start).Round(time.Microsecond)),
	}
	if scope.questions > 0 {
		attrs = append(attrs, slog.Int("questions", scope.questions))
	}
	if scope.mode != "" {
		attrs = append(attrs, slog.String("mode", scope.mode))
	}
	// A detail is recorded for failures the caller is not told the whole of,
	// and that now includes the ones below 500: a body read that timed out or
	// was cut short says nothing on the wire, so the log is the only record.
	if rec.detail != "" {
		attrs = append(attrs, slog.String("error", scrub(rec.detail, s.secrets(scope)...)))
	}
	if rec.status >= 500 || rec.failed {
		s.log.LogAttrs(ctx, slog.LevelError, "request failed", attrs...)
		return
	}
	s.log.LogAttrs(ctx, slog.LevelInfo, "request", attrs...)
}

// maxLoggedValue bounds a caller-supplied value on its way into a log line.
const maxLoggedValue = 64

// clip shortens s to at most n bytes without splitting a rune, marking that it
// did so.
func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := s[:n]
	for len(cut) > 0 && !utf8.ValidString(cut) {
		cut = cut[:len(cut)-1]
	}
	return cut + "…(truncated)"
}

// statusRecorder remembers the status a handler wrote, plus the detail of a
// failure that should be logged but not returned.
type statusRecorder struct {
	http.ResponseWriter
	status  int
	written bool
	detail  string
	// failed marks a request that went wrong in a way the status alone does
	// not show: a panic after the handler had already answered.
	failed bool
}

func (r *statusRecorder) WriteHeader(status int) {
	if r.written {
		return
	}
	r.status = status
	r.written = true
	r.ResponseWriter.WriteHeader(status)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	if !r.written {
		r.written = true
	}
	return r.ResponseWriter.Write(b)
}

// Unwrap lets http.ResponseController reach the underlying writer.
func (r *statusRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

// note records the detail of a failure for the log line only.
func note(w http.ResponseWriter, detail string) {
	if rec, ok := w.(*statusRecorder); ok {
		rec.detail = detail
	}
}

// writeJSON renders v and sends it with status. A value that cannot be
// encoded is reported to the caller as an internal error instead.
func (s *Server) writeJSON(w http.ResponseWriter, status int, v any) {
	body, err := json.Marshal(v)
	if err != nil {
		note(w, "encoding the response: "+err.Error())
		s.writeErrorBody(w, http.StatusInternalServerError, codeInternal, "the response could not be encoded")
		return
	}
	body = append(body, '\n')
	h := w.Header()
	h.Set("Content-Type", "application/json; charset=utf-8")
	h.Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(status)
	// A failed write means the client is gone; the log line still records the
	// status, and there is nothing useful left to say on the wire.
	_, _ = w.Write(body)
}

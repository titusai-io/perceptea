package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"

	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
	"github.com/titusai-io/perceptea/provider/openai"
)

// The machine-readable codes in an error body. They are part of the API: a
// caller should branch on the code, not on the message.
const (
	codeInvalidJSON         = "invalid_json"
	codeInvalidRequest      = "invalid_request"
	codeMissingAPIKey       = "missing_api_key"
	codePayloadTooLarge     = "payload_too_large"
	codeUnsupportedMode     = "unsupported_mode"
	codeUpstreamRateLimited = "upstream_rate_limited"
	codeUpstreamError       = "upstream_error"
	codeTimeout             = "timeout"
	codeNotFound            = "not_found"
	codeMethodNotAllowed    = "method_not_allowed"
	codeInternal            = "internal"
)

// StatusClientClosedRequest is nginx's 499: the caller hung up before the
// answer was ready, so no body is sent and none would be read.
const StatusClientClosedRequest = 499

// errorResponse is the body every failed request returns, except a 499.
type errorResponse struct {
	Error string `json:"error"`
	Code  string `json:"code"`
}

// failure is how one error is reported: what the caller is told, and what is
// kept for the log.
type failure struct {
	status  int
	code    string
	message string
	// detail is logged and never sent to the caller.
	detail string
}

// classify maps an evaluation error onto a status and a code.
func (s *Server) classify(err error) failure {
	if err == nil {
		return failure{status: http.StatusOK}
	}

	// The provider's answer outranks the deadline. When a retry sleep runs
	// into the request deadline the provider reports both in one error, and
	// the deadline is the less useful half: a 429 carrying a Retry-After is
	// the one signal that tells a caller to back off rather than retry now.
	var upstream *openai.APIError
	if errors.As(err, &upstream) {
		if upstream.StatusCode == http.StatusTooManyRequests {
			return failure{
				status:  http.StatusTooManyRequests,
				code:    codeUpstreamRateLimited,
				message: upstreamMessage(upstream),
				detail:  err.Error(),
			}
		}
		return failure{
			status:  http.StatusBadGateway,
			code:    codeUpstreamError,
			message: upstreamMessage(upstream),
			detail:  err.Error(),
		}
	}

	switch {
	// A deadline beats a cancellation: the timeout we imposed also cancels
	// the request context, and the caller deserves the more specific answer.
	case errors.Is(err, context.DeadlineExceeded):
		return failure{
			status:  http.StatusGatewayTimeout,
			code:    codeTimeout,
			message: fmt.Sprintf("the evaluation exceeded the server's %s deadline", s.cfg.RequestTimeout),
		}
	case errors.Is(err, context.Canceled):
		return failure{status: StatusClientClosedRequest}

	case errors.Is(err, openai.ErrNoAPIKey):
		return failure{
			status:  http.StatusUnauthorized,
			code:    codeMissingAPIKey,
			message: s.missingKeyMessage(),
		}
	case errors.Is(err, classifier.ErrOneshotUnsupported):
		return failure{
			status:  http.StatusBadRequest,
			code:    codeUnsupportedMode,
			message: "this provider cannot serve mode \"oneshot\"; use \"parallel\"",
		}
	// A mode the classifier does not know is the caller's mistake in exactly
	// the same way an unsupported one is, and is worth the same code.
	case errors.Is(err, classifier.ErrUnknownMode):
		return failure{
			status:  http.StatusBadRequest,
			code:    codeUnsupportedMode,
			message: fmt.Sprintf("unknown mode; expected %q or %q", classifier.ModeParallel, classifier.ModeOneshot),
		}
	case errors.Is(err, classifier.ErrNoQuestions):
		return failure{
			status:  http.StatusBadRequest,
			code:    codeInvalidRequest,
			message: "no questions to answer",
		}
	}

	var invalid *classifier.ValidationError
	if errors.As(err, &invalid) {
		return failure{status: http.StatusBadRequest, code: codeInvalidRequest, message: invalid.Error()}
	}

	// A transport failure never reached the provider, so the caller learns
	// only that the hop failed; the specifics go to the log, where they may
	// name an internal host.
	var urlErr *url.Error
	var netErr net.Error
	if errors.As(err, &urlErr) || errors.As(err, &netErr) {
		return failure{
			status:  http.StatusBadGateway,
			code:    codeUpstreamError,
			message: "the upstream provider could not be reached",
			detail:  err.Error(),
		}
	}

	return failure{
		status:  http.StatusInternalServerError,
		code:    codeInternal,
		message: "internal error",
		detail:  err.Error(),
	}
}

// classifyRead maps a failure to *read* the request body — as opposed to a
// failure to parse what was read — onto a status and a code.
//
// None of these carry the error's own text to the caller. A socket error
// reads "read tcp 10.0.3.17:8080->203.0.113.9:54321: i/o timeout": it names
// the address this process is bound to, which an unauthenticated caller has
// no business learning. The text goes to the log instead, exactly as a failed
// upstream hop does.
func (s *Server) classifyRead(err error) failure {
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		return failure{
			status:  http.StatusRequestEntityTooLarge,
			code:    codePayloadTooLarge,
			message: fmt.Sprintf("request body is larger than the %d byte limit", s.cfg.MaxBodyBytes),
		}
	}

	detail := "reading the request body: " + err.Error()

	// A slowloris client trips http.Server.ReadTimeout, which surfaces as a
	// net.Error that reports a timeout; the request scope's own deadline
	// surfaces as context.DeadlineExceeded.
	var netErr net.Error
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, os.ErrDeadlineExceeded) ||
		(errors.As(err, &netErr) && netErr.Timeout()) {
		return failure{
			status:  http.StatusGatewayTimeout,
			code:    codeTimeout,
			message: "the request body did not arrive before the server's read deadline",
			detail:  detail,
		}
	}

	// Everything else is a connection that stopped delivering: a cancelled
	// request, a reset, a client that hung up mid-body. Nobody is listening,
	// so nothing is written back.
	return failure{status: StatusClientClosedRequest, detail: detail}
}

// upstreamMessage renders a provider error without echoing its raw body.
func upstreamMessage(e *openai.APIError) string {
	msg := strings.TrimSpace(e.Message)
	if msg == "" {
		msg = strings.TrimSpace(e.Code)
	}
	if msg == "" {
		msg = strings.TrimSpace(e.Type)
	}
	if msg == "" {
		msg = "no message"
	}
	return fmt.Sprintf("upstream provider error (status %d): %s", e.StatusCode, msg)
}

// missingKeyMessage tells the caller where a key could come from, naming the
// variable it is read from — never a key.
func (s *Server) missingKeyMessage() string {
	var b strings.Builder
	b.WriteString("no API key: set ")
	b.WriteString(config.EnvAPIKey)
	b.WriteString(" in the server's environment")
	if s.cfg.AllowRequestCredentials {
		b.WriteString(", or send \"api_key\" in the request body")
	}
	return b.String()
}

// fail reports err to the caller with the status its kind earns.
func (s *Server) fail(w http.ResponseWriter, r *http.Request, err error) {
	s.report(w, r, s.classify(err))
}

// report sends one already-classified failure.
func (s *Server) report(w http.ResponseWriter, r *http.Request, f failure) {
	if f.detail != "" {
		note(w, f.detail)
	}
	if f.status == StatusClientClosedRequest {
		// Nobody is listening; send the status for the log and stop.
		w.WriteHeader(StatusClientClosedRequest)
		return
	}
	s.writeError(w, r, f.status, f.code, f.message)
}

// writeError sends one error body, with any credential scrubbed out of the
// message on the way.
func (s *Server) writeError(w http.ResponseWriter, r *http.Request, status int, code, message string) {
	message = scrub(message, s.secrets(scopeFrom(r.Context()))...)
	s.writeErrorBody(w, status, code, message)
}

// writeErrorBody is the last step for an error, and the one path that must
// not itself be able to fail.
func (s *Server) writeErrorBody(w http.ResponseWriter, status int, code, message string) {
	body, err := json.Marshal(errorResponse{Error: message, Code: code})
	if err != nil {
		body = []byte(`{"error":"internal error","code":"internal"}`)
	}
	body = append(body, '\n')
	h := w.Header()
	h.Set("Content-Type", "application/json; charset=utf-8")
	h.Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

// scrub removes credentials from a string bound for a caller or a log.
//
// A secret shorter than shortestCredential is left alone. The threshold is a
// trade-off in both directions: local servers conventionally take a throwaway
// key — "ollama", "EMPTY", "lm-studio" — and a key that is never scrubbed is
// the bug this guards against, while scrubbing a one or two character secret
// would redact every stray letter in a message and leave it useless. Four is
// short enough to cover the conventional placeholders and long enough that a
// match is not an accident.
func scrub(s string, secrets ...string) string {
	const shortestCredential = 4
	for _, secret := range secrets {
		if len(secret) < shortestCredential {
			continue
		}
		s = strings.ReplaceAll(s, secret, "[redacted]")
	}
	return s
}

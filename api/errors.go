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
	"github.com/titusai-io/perceptea/provider/inference"
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
	var upstream *inference.APIError
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
	// A reply the provider cut off at its output token limit is an upstream
	// failure like any other 502, and the one whose message the caller most
	// needs: it is a configuration fault that will hit every candidate of
	// every request until someone changes the model or the reasoning
	// setting, and the error's own text says how. Both the parallel and the
	// one-shot path can raise it.
	case errors.Is(err, inference.ErrTruncatedReply), errors.Is(err, classifier.ErrTruncatedReply):
		return failure{
			status:  http.StatusBadGateway,
			code:    codeUpstreamError,
			message: withoutPackagePrefix(err.Error()),
			detail:  err.Error(),
		}

	// The logprob scorer's two refusals belong with the truncation above,
	// for the same reason: both are configuration faults that repeat on
	// every candidate of every request, and both carry the fix in their own
	// text. An endpoint that will not return logprobs will not return them
	// for the next candidate either, and a model that answers a Yes-or-No
	// question with neither word answers the next one the same way — which
	// is exactly why neither is allowed to become a neutral score.
	case errors.Is(err, inference.ErrNoLogprobs), errors.Is(err, inference.ErrNoDecisionToken):
		return failure{
			status:  http.StatusBadGateway,
			code:    codeUpstreamError,
			message: withoutPackagePrefix(err.Error()),
			detail:  err.Error(),
		}

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

	case errors.Is(err, inference.ErrNoAPIKey):
		return failure{
			status:  http.StatusUnauthorized,
			code:    codeMissingAPIKey,
			message: s.missingKeyMessage(),
		}
	case errors.Is(err, classifier.ErrOneshotUnsupported):
		return failure{
			status:  http.StatusBadRequest,
			code:    codeUnsupportedMode,
			message: oneshotUnsupportedMessage,
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
	// The batch handler rejects an empty "items" before it gets this far, so
	// this is the belt to that braces: were the check ever to move, an empty
	// batch would still be the caller's 400 rather than the server's 500.
	case errors.Is(err, classifier.ErrNoItems):
		return failure{
			status:  http.StatusBadRequest,
			code:    codeInvalidRequest,
			message: "no items to evaluate",
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

// withoutPackagePrefix strips the package name an error was tagged with, so
// that a message written to be read by an operator reaches them as it was
// written rather than as "inference: …". It is used for the faults whose own
// text is the diagnosis and the fix — a truncated reply, an endpoint that
// returns no logprobs, a model that will not answer the question.
//
// It takes the text rather than the error because the batch path has nothing
// else: see [Server.classifyItem].
func withoutPackagePrefix(msg string) string {
	for _, prefix := range []string{"inference: ", "classifier: "} {
		if after, ok := strings.CutPrefix(msg, prefix); ok {
			return after
		}
	}
	return msg
}

// oneshotUnsupportedMessage is what a caller is told when the provider behind
// this server cannot run one-shot mode. It is a constant because both tables
// below report it.
const oneshotUnsupportedMessage = `this provider cannot serve mode "oneshot"; use "parallel"`

// unreachableMessage is what a caller is told about a hop that never reached
// the provider. It says nothing else on purpose: the text underneath names
// the endpoint this server calls, which may be an internal host.
const unreachableMessage = "the upstream provider could not be reached"

// itemFailure is one batch item's error, classified.
//
// It is the per-item counterpart of [failure] and carries every part of one
// but the status: a batch item is not an HTTP response, because the request
// itself was served, so the status has nothing to be. What is left is the
// three things that do still apply — the message the caller reads, the code
// they branch on, and the detail only the log gets.
type itemFailure struct {
	code    string
	message string
	// detail is logged and never sent to the caller.
	detail string
}

// upstreamStatusPrefix is what [inference.APIError.Error] writes before the
// HTTP status. It is derived from the type rather than copied out of it, so
// that a change to the wording cannot silently stop [upstreamStatus]
// matching; TestClassifyItemReadsAnUpstreamStatus pins the pairing either
// way.
var upstreamStatusPrefix = strings.TrimSuffix((&inference.APIError{StatusCode: 0}).Error(), "0")

// The per-item faults [classifier.Evaluator.EvaluateBatch] reports for an
// item it never attempted. Both are the caller's own mistake, both name
// nothing but the caller's own document, and the classifier exports no
// sentinel to match them on — so they are matched as text, and
// TestBatchClassifiesAnItemTheClassifierRejects drives the real classifier so
// that a change to either wording is a red test rather than a caller who
// suddenly reads "the upstream provider could not be reached" about their own
// missing field.
const (
	itemNoStatePrefix     = `classifier: item has no "state"`
	itemRenderStatePrefix = "classifier: rendering state:"
)

// classifyItem maps one batch item's error onto what its caller is told, the
// code they can branch on, and what goes to the log.
//
// [Server.classify] matches on error values, with errors.Is and errors.As.
// This cannot: [classifier.BatchResult] reports an item's failure as a
// string, so by the time a result reaches this package the *inference.APIError
// or *url.Error behind it has been rendered and thrown away. Classifying the
// text is what is left, and it has to be done, because the reason the
// request-level table exists applies here unchanged — the text of a transport
// failure names the endpoint this server calls, and an unauthenticated caller
// has no business learning an internal host name.
//
// The table is therefore an allow-list. Only the faults this project writes
// itself, whose own text is the diagnosis and the fix, reach the caller;
// anything unrecognised is reported as the same opaque upstream failure the
// single endpoint gives, with its text sent to the log. A deny-list would
// have to enumerate every way a URL can reach a message, and the one it
// missed would be the leak.
//
// A rule that passes the text on anchors at the start of it, never on a
// substring. `Post "http://gateway.internal/v1": context deadline exceeded`
// contains the text of context.DeadlineExceeded, and a Contains rule that
// returned its input would hand the caller the host. A rule whose message is
// a fixed string may match anywhere, because none of the original survives.
func (s *Server) classifyItem(text string) itemFailure {
	if text == "" {
		return itemFailure{}
	}

	// The faults whose own text is the diagnosis and the fix, treated exactly
	// as the request-level table treats them: the message as it was written,
	// with the package tag taken off.
	for _, sentinel := range []error{
		inference.ErrTruncatedReply,
		inference.ErrNoLogprobs,
		inference.ErrNoDecisionToken,
		classifier.ErrTruncatedReply,
	} {
		if strings.HasPrefix(text, sentinel.Error()) {
			return itemFailure{code: codeUpstreamError, message: withoutPackagePrefix(text)}
		}
	}

	// A provider that answered, with a status. The rate limit is split out for
	// the reason it is at request level: it is the one signal that tells a
	// caller to back off rather than retry now, and on this endpoint the item
	// is where they read it.
	if status, message, ok := upstreamStatus(text); ok {
		code := codeUpstreamError
		if status == http.StatusTooManyRequests {
			code = codeUpstreamRateLimited
		}
		return itemFailure{
			code:    code,
			message: fmt.Sprintf("upstream provider error (status %d): %s", status, message),
			detail:  text,
		}
	}

	// The caller's own item, rejected before any call was made.
	if strings.HasPrefix(text, itemNoStatePrefix) || strings.HasPrefix(text, itemRenderStatePrefix) {
		return itemFailure{code: codeInvalidRequest, message: withoutPackagePrefix(text)}
	}

	// Fixed messages from here down, so these rules may match anywhere in the
	// text without carrying any of it to the caller.
	if strings.Contains(text, classifier.ErrOneshotUnsupported.Error()) {
		return itemFailure{code: codeUnsupportedMode, message: oneshotUnsupportedMessage, detail: text}
	}
	if strings.Contains(text, context.DeadlineExceeded.Error()) {
		return itemFailure{
			code:    codeTimeout,
			message: fmt.Sprintf("the evaluation exceeded the server's %s deadline", s.cfg.RequestTimeout),
			detail:  text,
		}
	}

	return itemFailure{code: codeUpstreamError, message: unreachableMessage, detail: text}
}

// upstreamStatus reads the HTTP status out of a rendered
// [inference.APIError], with whatever the provider said after it.
//
// The fallback for a provider that said nothing is the one
// [upstreamMessage] uses, so that the two endpoints report the same silence
// the same way.
func upstreamStatus(text string) (status int, message string, ok bool) {
	rest, found := strings.CutPrefix(text, upstreamStatusPrefix)
	if !found {
		return 0, "", false
	}
	digits := 0
	for digits < len(rest) && rest[digits] >= '0' && rest[digits] <= '9' {
		digits++
	}
	if digits == 0 {
		return 0, "", false
	}
	status, err := strconv.Atoi(rest[:digits])
	if err != nil {
		return 0, "", false
	}
	message = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(rest[digits:]), ":"))
	if message == "" {
		message = "no message"
	}
	return status, message, true
}

// upstreamMessage renders a provider error without echoing its raw body.
func upstreamMessage(e *inference.APIError) string {
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

package api

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
)

// requestSettings is the part of a request body that is a transport concern
// rather than a question: where to send the calls, with whose key, which model
// and how to ask.
//
// It is embedded in every endpoint's own wire type and resolved in one place,
// because the rules are the same wherever it appears and a second copy of them
// is a second place for them to drift.
type requestSettings struct {
	Model string `json:"model"`
	// InferenceBaseURL is the API root this one request's scoring calls
	// should go to, in place of the server's own.
	InferenceBaseURL string `json:"inference_base_url"`
	APIKey           string `json:"api_key"`
	// Temperature is a pointer so that an explicit 0 is distinguishable from
	// an absent field; 0 is a meaningful value here, and the usual one.
	Temperature *float64 `json:"temperature"`
	Mode        string   `json:"mode"`
}

// evaluateRequest is the wire shape of POST /api/evaluate.
type evaluateRequest struct {
	State     classifier.State     `json:"state"`
	Questions classifier.Questions `json:"questions"`
	requestSettings
}

// resolution is what one request's shared fields came to.
type resolution struct {
	settings    Settings
	model       string
	temperature float64
	mode        classifier.Mode
}

// decodeJSONBody reads the one JSON document a request body has to be into v.
//
// It reports false when it has already answered the caller, which is every
// path but the happy one.
func (s *Server) decodeJSONBody(w http.ResponseWriter, r *http.Request, v any) bool {
	// The reader is wrapped so that a failure on the wire can be told apart
	// from a failure to parse: once encoding/json has passed an error through,
	// a socket that died and a custom UnmarshalJSON that objected look exactly
	// alike, and only one of them may be quoted back to the caller.
	reader := &trackingReader{r: http.MaxBytesReader(w, r.Body, s.cfg.MaxBodyBytes)}
	r.Body = reader

	dec := json.NewDecoder(reader)
	// A caller who writes "apikey" instead of "api_key" has it silently
	// ignored, and the server's own credential is spent on their behalf. A
	// rejection is the only way they ever find out. (A miscased "Api_Key" is
	// understood: encoding/json matches field names case-insensitively.) This
	// binds the top level only: a question's own decoder is deliberately
	// tolerant, and stays so.
	dec.DisallowUnknownFields()

	if err := dec.Decode(v); err != nil {
		s.failDecode(w, r, reader.err, err)
		return false
	}
	// encoding/json stops at the end of the first value and ignores whatever
	// follows it, so a body of two documents would be half read and wholly
	// accepted.
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		s.failDecode(w, r, reader.err, errTrailingContent)
		return false
	}
	return true
}

// resolveSettings turns the shared body fields into the settings this
// request's evaluator is built from, and records on the request scope what the
// log line and the scrubber will need.
//
// It reports false when it has already answered the caller.
func (s *Server) resolveSettings(w http.ResponseWriter, r *http.Request, body requestSettings) (resolution, bool) {
	scope := scopeFrom(r.Context())

	mode := classifier.Mode(strings.TrimSpace(body.Mode))
	if mode == "" {
		mode = classifier.ModeParallel
	}
	// The mode is the caller's text and is recorded before it is validated, so
	// what reaches the log is clipped: a 20 KB mode produced a 20 KB log line.
	scope.mode = clip(string(mode), maxLoggedValue)

	apiKey, baseURL := s.cfg.APIKey, s.cfg.BaseURL
	bodyKey, bodyBase := strings.TrimSpace(body.APIKey), strings.TrimSpace(body.InferenceBaseURL)
	if s.cfg.AllowRequestCredentials {
		if bodyKey != "" {
			apiKey = bodyKey
		}
		if bodyBase != "" {
			// A base URL the caller typed is caller input, so a bad one is a
			// 400. Left to the provider factory it would surface as a 500,
			// which blames the server for the client's typo. The message
			// never repeats the value: a caller may have embedded a
			// credential in it, and the request scope does not yet know to
			// scrub one.
			if err := config.ValidateBaseURL(bodyBase); err != nil {
				s.writeError(w, r, http.StatusBadRequest, codeInvalidRequest, `"inference_base_url" `+err.Error())
				return resolution{}, false
			}
			baseURL = bodyBase
		}
	} else if bodyKey != "" || bodyBase != "" {
		// Silently ignored rather than rejected: a caller pointed at a locked
		// down server should still get its answers.
		s.log.DebugContext(r.Context(), "ignoring request-supplied credentials",
			slog.Bool("api_key", bodyKey != ""),
			slog.Bool("inference_base_url", bodyBase != ""))
	}
	scope.secret = apiKey
	scope.baseURL = baseURL

	model := strings.TrimSpace(body.Model)
	if model == "" {
		model = s.cfg.Model
	}
	temperature := s.cfg.Temperature
	if body.Temperature != nil {
		temperature = *body.Temperature
	}

	return resolution{
		settings: Settings{
			APIKey:  apiKey,
			BaseURL: baseURL,
			Model:   model,
			// Server-side only, deliberately: the reasoning effort describes
			// the model the operator chose, and letting a caller raise it
			// would let them spend the server's key on thinking tokens.
			ReasoningEffort: s.cfg.ReasoningEffort,
			Scorer:          s.cfg.Scorer,
			MaxConcurrency:  s.cfg.MaxConcurrency,
		},
		model:       model,
		temperature: temperature,
		mode:        mode,
	}, true
}

// handleEvaluate answers one set of questions about one state.
func (s *Server) handleEvaluate(w http.ResponseWriter, r *http.Request) {
	scope := scopeFrom(r.Context())

	var body evaluateRequest
	if !s.decodeJSONBody(w, r, &body) {
		return
	}

	// An explicitly null state counts as absent: a caller who sends null has
	// given the questions nothing to be asked about.
	if body.State.IsZero() || body.State.IsNull() {
		s.writeError(w, r, http.StatusBadRequest, codeInvalidRequest, `missing "state"`)
		return
	}
	if body.Questions.Len() == 0 {
		s.writeError(w, r, http.StatusBadRequest, codeInvalidRequest, `missing "questions": at least one question is required`)
		return
	}
	scope.questions = body.Questions.Len()

	resolved, ok := s.resolveSettings(w, r, body.requestSettings)
	if !ok {
		return
	}

	evaluator, err := s.newEvaluator(resolved.settings)
	if err != nil {
		s.fail(w, r, err)
		return
	}

	resp, err := evaluator.Evaluate(r.Context(), classifier.Request{
		State:       body.State,
		Questions:   body.Questions,
		Model:       resolved.model,
		Temperature: resolved.temperature,
		Mode:        resolved.mode,
	})
	if err != nil {
		s.fail(w, r, err)
		return
	}

	s.writeJSON(w, http.StatusOK, resp)
}

// errTrailingContent marks a body that carries something after its JSON
// object. It is a parse failure, not a read failure.
var errTrailingContent = errors.New("unexpected content after the JSON object")

// trackingReader remembers the last error the underlying reader produced, so
// that a decode failure can be attributed to the wire or to the document.
// io.EOF is not recorded: it is how a body ends, and encoding/json turns it
// into the parse error it deserves.
type trackingReader struct {
	r   io.ReadCloser
	err error
}

func (t *trackingReader) Read(p []byte) (int, error) {
	n, err := t.r.Read(p)
	if err != nil && !errors.Is(err, io.EOF) {
		t.err = err
	}
	return n, err
}

// Close keeps this usable as the request's Body, which is an io.ReadCloser.
func (t *trackingReader) Close() error { return t.r.Close() }

// failDecode reports a body that could not be turned into a request.
//
// readErr is what the wire produced, if anything; err is what the decoder
// returned. The two are reported differently on purpose: a parse failure is
// about the caller's own document and may be quoted back to them, while a
// read failure is about the connection, and its text names this server's
// socket.
func (s *Server) failDecode(w http.ResponseWriter, r *http.Request, readErr, err error) {
	// A syntax error was found in bytes that did arrive, so it describes the
	// document even if the connection later failed as well.
	var syntax *json.SyntaxError
	var wrongType *json.UnmarshalTypeError
	if errors.As(err, &syntax) || errors.As(err, &wrongType) {
		s.failParse(w, r, err)
		return
	}
	if readErr != nil {
		s.report(w, r, s.classifyRead(readErr))
		return
	}
	// A caller that hung up or ran out of time before the decoder noticed.
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		s.fail(w, r, err)
		return
	}
	s.failParse(w, r, err)
}

// failParse reports a body that arrived intact and is not the document this
// endpoint takes.
func (s *Server) failParse(w http.ResponseWriter, r *http.Request, err error) {
	s.writeError(w, r, http.StatusBadRequest, codeInvalidJSON, "could not decode the request body: "+err.Error())
}

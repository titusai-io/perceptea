package api

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
)

// BatchEvaluator answers many states against one question set as a single job.
//
// It is separate from [Evaluator] rather than part of it because the two are
// asked for at different times: every request needs an evaluator, and only a
// batch needs this. *classifier.Evaluator implements both, and an evaluator
// that does not implement this one cannot serve the batch endpoint.
type BatchEvaluator interface {
	EvaluateBatch(ctx context.Context, req classifier.BatchRequest) (classifier.BatchResponse, error)
}

// batchRequest is the wire shape of POST /api/evaluate/batch: many states, one
// shared question set, and the same transport fields every endpoint takes.
type batchRequest struct {
	Items     []classifier.BatchItem `json:"items"`
	Questions classifier.Questions   `json:"questions"`
	requestSettings
}

// batchResponse is the wire shape of a batch answer.
//
// It is this package's own type rather than [classifier.BatchResponse]
// rendered straight out, because a per-item failure has to carry one thing
// the classifier cannot give it: the machine-readable code. A caller of
// /api/evaluate branches on the `code` of an error body; a caller of this
// endpoint gets a 200 whatever happened to the items, so without a per-item
// code the only way to tell a rate limit from an unreachable host would be to
// match on prose — which is exactly what the error table exists to stop.
type batchResponse struct {
	Model   string                `json:"model"`
	Results []batchItemResult     `json:"results"`
	Usage   classifier.Usage      `json:"usage"`
	Meta    *classifier.BatchMeta `json:"meta,omitempty"`
}

// batchItemResult is one item's result with the code its failure earned.
//
// The classifier's result is embedded, so every field it declares is on the
// wire exactly where it was, and the code is added beside them.
type batchItemResult struct {
	classifier.BatchResult
	// ErrorCode is the machine-readable reason for Error, from the same
	// vocabulary the error table uses: "upstream_error",
	// "upstream_rate_limited", "timeout", "invalid_request" or
	// "unsupported_mode". It is absent when the item succeeded, and present
	// whenever Error is.
	ErrorCode string `json:"error_code,omitempty"`
}

// maxLoggedItemDetails bounds how many distinct per-item details one batch
// contributes to its log line. A hundred items failing the same way are worth
// one line; a hundred failing in a hundred ways are worth a line nobody
// reads, so past this many the rest are counted rather than printed.
const maxLoggedItemDetails = 5

// itemDetails joins the details of a batch's failed items into the one string
// the log line carries, dropping repeats: when the provider is unreachable
// every item says so, and saying it a hundred times adds nothing.
func itemDetails(details []string) string {
	seen := make(map[string]struct{}, len(details))
	unique := make([]string, 0, len(details))
	for _, d := range details {
		if _, ok := seen[d]; ok {
			continue
		}
		seen[d] = struct{}{}
		unique = append(unique, d)
	}
	if len(unique) <= maxLoggedItemDetails {
		return strings.Join(unique, "; ")
	}
	return fmt.Sprintf("%s; (+%d more)",
		strings.Join(unique[:maxLoggedItemDetails], "; "), len(unique)-maxLoggedItemDetails)
}

// handleBatchEvaluate answers one set of questions about many states.
//
// The whole job shares one concurrency budget, which is the difference from a
// caller sending the same states one request at a time: a hundred requests
// each fan out to the configured limit on their own, and a hundred fan-outs
// competing for one provider is the problem this endpoint exists to solve.
func (s *Server) handleBatchEvaluate(w http.ResponseWriter, r *http.Request) {
	scope := scopeFrom(r.Context())

	var body batchRequest
	if !s.decodeJSONBody(w, r, &body) {
		return
	}

	if len(body.Items) == 0 {
		s.writeError(w, r, http.StatusBadRequest, codeInvalidRequest,
			`missing "items": at least one item is required`)
		return
	}
	// A batch fans out into items times candidates calls, so the ceiling is
	// what stops one request committing the server's key to an unbounded
	// amount of provider spend. The message names the setting and both
	// numbers, so that an operator reading a caller's bug report knows both
	// what was asked for and which variable to raise.
	if limit := s.cfg.MaxBatchItems; len(body.Items) > limit {
		s.writeError(w, r, http.StatusBadRequest, codeInvalidRequest, fmt.Sprintf(
			"a batch may carry at most %d items, got %d; raise %s to allow more",
			limit, len(body.Items), config.EnvMaxBatchItems))
		return
	}
	if body.Questions.Len() == 0 {
		s.writeError(w, r, http.StatusBadRequest, codeInvalidRequest,
			`missing "questions": at least one question is required`)
		return
	}
	scope.questions = body.Questions.Len()
	scope.items = len(body.Items)

	resolved, ok := s.resolveSettings(w, r, body.requestSettings)
	if !ok {
		return
	}

	evaluator, err := s.newEvaluator(resolved.settings)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	batcher, ok := evaluator.(BatchEvaluator)
	if !ok {
		// A wiring fault, not the caller's: the evaluator this server was
		// built with cannot do batches at all. The type goes to the log,
		// where an operator can act on it.
		note(w, fmt.Sprintf("the configured evaluator (%T) does not implement BatchEvaluator", evaluator))
		s.writeError(w, r, http.StatusInternalServerError, codeInternal, "internal error")
		return
	}

	resp, err := batcher.EvaluateBatch(r.Context(), classifier.BatchRequest{
		Items:       body.Items,
		Questions:   body.Questions,
		Model:       resolved.model,
		Temperature: resolved.temperature,
		Mode:        resolved.mode,
	})
	// Only a whole-request failure comes back as an error and is reported
	// from the error table. A batch in which every single item failed is not
	// one of those: the request was served, the items were not, and the
	// caller is told which is which by reading the results.
	if err != nil {
		s.fail(w, r, err)
		return
	}

	// A per-item error is a message on its way to a caller like any other, so
	// it goes through the same classification and the same scrubbing a
	// whole-request failure does. Scrubbing alone is not enough and was the
	// bug: it removes the credential from a URL and leaves the URL, so an
	// unreachable gateway told every caller the name of an internal host —
	// on the one endpoint where the caller could not have named it
	// themselves. See [Server.classifyItem] for what is passed on and why.
	secrets := s.secrets(scope)
	out := batchResponse{
		Model:   resp.Model,
		Results: make([]batchItemResult, len(resp.Results)),
		Usage:   resp.Usage,
		Meta:    resp.Meta,
	}
	var details []string
	for i, r := range resp.Results {
		item := batchItemResult{BatchResult: r}
		if f := s.classifyItem(r.Error); f.code != "" {
			item.Error = scrub(f.message, secrets...)
			item.ErrorCode = f.code
			if f.detail != "" {
				details = append(details, f.detail)
			}
		}
		out.Results[i] = item
	}
	// The transport detail the caller did not get. It is the only record of
	// why a batch came back empty, and logRequest scrubs it on the way out.
	if len(details) > 0 {
		note(w, itemDetails(details))
	}

	s.writeJSON(w, http.StatusOK, out)
}

package api

import (
	"context"
	"fmt"
	"net/http"

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
	// it goes through the same scrubbing: an upstream failure can quote a URL
	// that carries a credential, and nothing below this line looks at it
	// again.
	secrets := s.secrets(scope)
	for i := range resp.Results {
		resp.Results[i].Error = scrub(resp.Results[i].Error, secrets...)
	}

	s.writeJSON(w, http.StatusOK, resp)
}

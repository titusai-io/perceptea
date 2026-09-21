package classifier

// BatchItem is one state to be evaluated against the batch's shared questions.
type BatchItem struct {
	// ID is the caller's own identifier, echoed back on the matching result
	// so that results can be reassociated without relying on order. It is
	// optional; results are returned in request order either way.
	ID string `json:"id,omitempty"`
	// State is the material the questions are asked about.
	State State `json:"state"`
}

// BatchRequest evaluates many states against one set of questions.
//
// The point of the batch is that the whole job shares one concurrency budget
// rather than one per item, which is what lets a hundred states be classified
// without a hundred independent fan-outs competing for the same provider.
type BatchRequest struct {
	Items     []BatchItem `json:"items"`
	Questions Questions   `json:"questions"`

	Model       string  `json:"model,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	Mode        Mode    `json:"mode,omitempty"`
}

// BatchResult is one item's outcome. Exactly one of Answers and Error is
// meaningful.
type BatchResult struct {
	// ID echoes the item's identifier.
	ID string `json:"id,omitempty"`
	// Index is the item's position in the request, always present, so a
	// result can be placed even when no ID was given.
	Index int `json:"index"`
	// Answers is the evaluation, when it succeeded.
	Answers Answers `json:"answers,omitempty"`
	// Usage is what this item cost.
	Usage Usage `json:"usage"`
	// Error says why this item has no answers. It is per-item on purpose:
	// one state that fails must not discard the ones that succeeded.
	//
	// This is the opposite of a single evaluation, where the first error
	// cancels the whole wave, and the difference is deliberate. Within one
	// evaluation every candidate feeds the same normalisation, so a missing
	// one corrupts the answer; across a batch the items are independent and
	// a partial result is a useful result.
	Error string `json:"error,omitempty"`
}

// BatchMeta describes how a batch ran.
type BatchMeta struct {
	Mode Mode `json:"mode"`
	// LatencyMS is the elapsed time for the whole batch.
	LatencyMS int64 `json:"latency_ms"`
	// ParallelCalls is the total scorer calls issued across every item.
	ParallelCalls int `json:"parallel_calls"`
	// Items, Succeeded and Failed count the outcomes, so a caller can tell
	// a wholly successful batch from a partly successful one without
	// walking the results.
	Items     int `json:"items"`
	Succeeded int `json:"succeeded"`
	Failed    int `json:"failed"`
}

// BatchResponse is the result of a batch, in request order.
type BatchResponse struct {
	Model   string        `json:"model"`
	Results []BatchResult `json:"results"`
	// Usage totals every item's usage, failed ones included: a call that
	// failed late still cost what it cost.
	Usage Usage      `json:"usage"`
	Meta  *BatchMeta `json:"meta,omitempty"`
}

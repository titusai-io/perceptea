package api

import (
	"fmt"
	"strings"
	"testing"

	"github.com/titusai-io/perceptea/internal/config"
	"github.com/titusai-io/perceptea/provider/inference"
)

// The logprob scorer's two refusals carry their own diagnosis and their own
// fix. Unmapped they reach the caller as a bare 500 "internal error" with all
// of that only in the log, which is the shape of failure this service has
// twice been bitten by.
func TestTheLogprobRefusalsAreActionable502s(t *testing.T) {
	for _, tc := range []struct {
		name   string
		err    error
		wantIn string
	}{
		{
			"an endpoint that returns no logprobs",
			fmt.Errorf("wrapped: %w", inference.ErrNoLogprobs),
			"logprob",
		},
		{
			"a model that answers neither Yes nor No",
			fmt.Errorf("wrapped: %w", inference.ErrNoDecisionToken),
			"",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, nil)
			h.failWith(tc.err)

			msg := h.expectError(h.post(sampleBody), 502, codeUpstreamError)

			// The package tag is for the log, not for whoever has to act.
			if strings.HasPrefix(msg, "inference: ") {
				t.Errorf("message reaches the caller still tagged: %q", msg)
			}
			// The sentinel's own words have to survive to the caller; a
			// generic "upstream provider error" would defeat the point.
			if !strings.Contains(msg, strings.TrimPrefix(tc.err.Error(), "wrapped: inference: ")) {
				t.Errorf("message %q does not carry the error's own text", msg)
			}
			if tc.wantIn != "" && !strings.Contains(msg, tc.wantIn) {
				t.Errorf("message %q does not mention %q", msg, tc.wantIn)
			}
		})
	}
}

// The scorer is the operator's choice and has to survive the whole way to the
// provider client. It is also part of what the client is built from, so two
// settings that differ only by it must not share a cached client.
func TestTheScorerReachesTheProviderAndTheCacheKey(t *testing.T) {
	h := newHarness(t, func(c *config.Config) { c.Scorer = config.ScorerLogprob })
	if w := h.post(sampleBody); w.Code != 200 {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if got := h.settings().Scorer; got != config.ScorerLogprob {
		t.Errorf("Settings.Scorer = %q, want %q", got, config.ScorerLogprob)
	}

	base := Settings{BaseURL: "https://inference.example/v1", Model: "probe-1", APIKey: testKey}
	chat, logprob := base, base
	chat.Scorer, logprob.Scorer = config.ScorerChat, config.ScorerLogprob
	if clientKey(chat) == clientKey(logprob) {
		t.Error("two settings differing only by scorer share a cache key, so the second would be served the first's client")
	}
}

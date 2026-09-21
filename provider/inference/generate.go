package inference

import (
	"context"
	"strings"

	"github.com/titusai-io/perceptea/classifier"
)

// Generate runs a plain chat completion — the classifier's oneshot mode.
//
// Unlike [Client.Score] it sets no token cap: the reply is a whole answer
// document. It also only ever asks for json_object, never json_schema: a
// one-shot reply has no fixed schema to declare.
//
// The provider's finish reason is reported on the result rather than judged
// here, because only the caller can tell a document the model got wrong from
// one it never reached the end of.
//
// When req.JSONObject is set the call asks for a JSON object and drops that
// request for the rest of the call if the provider rejects the shape, on the
// same negotiation Score uses and under the same rule: the drop is only
// remembered on the client once the plain call has actually answered. Then it
// takes the shared level straight to plain rather than one step, because a
// provider that rejects json_object and answers without any response_format
// has said it supports none — and json_schema is strictly more than
// json_object, so it cannot be the survivor. A call that fails both ways
// leaves the level alone; nothing was learned about the request's shape.
func (c *Client) Generate(ctx context.Context, req classifier.GenerateRequest) (classifier.GenerateResult, error) {
	model, err := c.resolveModel(req.Model)
	if err != nil {
		return classifier.GenerateResult{}, err
	}

	messages := make([]chatMessage, 0, 2)
	if strings.TrimSpace(req.System) != "" {
		messages = append(messages, chatMessage{Role: "system", Content: req.System})
	}
	messages = append(messages, chatMessage{Role: "user", Content: req.User})

	asking := req.JSONObject && c.outputLevel() < levelPlain
	dropped := false
	for {
		body := chatRequest{
			Model:           model,
			Messages:        messages,
			Temperature:     req.Temperature,
			ReasoningEffort: c.reasoningEffort,
		}
		if asking {
			body.ResponseFormat = &responseFormat{Type: "json_object"}
		}

		resp, err := c.complete(ctx, body)
		if err != nil {
			if body.ResponseFormat != nil && unsupportedShape(err) {
				asking, dropped = false, true
				continue
			}
			return classifier.GenerateResult{}, err
		}
		if dropped {
			c.downgrade(levelPlain)
		}

		out := classifier.GenerateResult{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		}
		if len(resp.Choices) > 0 {
			choice := resp.Choices[0]
			content := string(choice.Message.Content)
			if strings.TrimSpace(content) == "" {
				content = choice.Message.reasoning()
			}
			out.Content = content
			// The finish reason is reported rather than acted on here: only
			// the caller knows whether what did arrive was a usable
			// document, and a reply cut off after the last answer is still
			// an answer.
			out.FinishReason = choice.FinishReason
		}
		return out, nil
	}
}

package inference

import (
	"bytes"
	"encoding/json"
	"strings"
)

// chatRequest is the /chat/completions request document. Temperature carries
// no omitempty: 0 is the meaningful default for a classifier and must reach
// the provider.
//
// ReasoningEffort does carry omitempty, and that is the whole of its default
// behaviour: an unconfigured client sends no reasoning field at all, so a
// provider that has never seen one is asked exactly what it was asked before.
type chatRequest struct {
	Model           string          `json:"model"`
	Messages        []chatMessage   `json:"messages"`
	Temperature     float64         `json:"temperature"`
	MaxTokens       int             `json:"max_tokens,omitempty"`
	ResponseFormat  *responseFormat `json:"response_format,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
}

// chatMessage is one turn of the prompt.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// responseFormat asks for constrained output. Type is "json_schema" or
// "json_object"; JSONSchema is set only for the former.
type responseFormat struct {
	Type       string          `json:"type"`
	JSONSchema *jsonSchemaSpec `json:"json_schema,omitempty"`
}

// jsonSchemaSpec is the strict-schema payload.
type jsonSchemaSpec struct {
	Name   string          `json:"name"`
	Strict bool            `json:"strict"`
	Schema json.RawMessage `json:"schema"`
}

// chatResponse is the slice of the response document this package reads.
type chatResponse struct {
	Choices []chatChoice `json:"choices"`
	Usage   chatUsage    `json:"usage"`
}

// chatChoice is one candidate completion.
//
// FinishReason is why the provider stopped generating. "length" is the one
// value this package acts on: it means the reply was cut off at the output
// token limit rather than finished, and a cut-off reply is not an answer.
type chatChoice struct {
	Message      chatResponseMessage `json:"message"`
	FinishReason string              `json:"finish_reason"`
}

// chatResponseMessage is the assistant turn. The reasoning fields are
// non-standard but common enough — a model that thinks before it answers puts
// the thinking in one of them, and several providers leave content empty while
// doing so. Which of the two names is used is provider-specific, so both are
// read.
type chatResponseMessage struct {
	Content          messageContent `json:"content"`
	Reasoning        messageContent `json:"reasoning"`
	ReasoningContent messageContent `json:"reasoning_content"`
}

// reasoning returns whichever reasoning field the provider populated.
func (m chatResponseMessage) reasoning() string {
	if strings.TrimSpace(string(m.Reasoning)) != "" {
		return string(m.Reasoning)
	}
	return string(m.ReasoningContent)
}

// truncated reports whether a choice was cut off at the output token limit.
// The comparison is case-insensitive because the value is echoed text, not a
// protocol constant; nothing else is accepted, because nothing else means the
// reply ran out of room.
func (c chatChoice) truncated() bool {
	return strings.EqualFold(strings.TrimSpace(c.FinishReason), finishReasonLength)
}

// finishReasonLength is the finish_reason a provider sends when the output
// token limit, not the model, ended the reply.
const finishReasonLength = "length"

// chatUsage is the token accounting. Absent fields stay zero, which is exactly
// what [classifier.ScoreResult] wants.
type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// messageContent is a message field that different providers render
// differently: a string, null, or an array of typed parts. Anything else
// decodes to the empty string rather than failing the whole call — one odd
// candidate should cost one score, not the request.
type messageContent string

// UnmarshalJSON implements [json.Unmarshaler]. It never returns an error.
func (m *messageContent) UnmarshalJSON(data []byte) error {
	*m = ""
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}

	var s string
	if err := json.Unmarshal(trimmed, &s); err == nil {
		*m = messageContent(s)
		return nil
	}

	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(trimmed, &parts); err == nil {
		var b strings.Builder
		for _, p := range parts {
			b.WriteString(p.Text)
		}
		*m = messageContent(b.String())
	}
	return nil
}

// outputLevel is how hard the client is currently willing to push for
// structured output. The levels are ordered from most to least constrained and
// the client only ever moves down them.
type outputLevel int32

const (
	// levelJSONSchema sends a strict json_schema response_format.
	levelJSONSchema outputLevel = iota
	// levelJSONObject sends response_format {"type":"json_object"}.
	levelJSONObject
	// levelPlain sends no response_format at all.
	levelPlain
)

// String names the level for log records.
func (l outputLevel) String() string {
	switch l {
	case levelJSONSchema:
		return "json_schema"
	case levelJSONObject:
		return "json_object"
	case levelPlain:
		return "plain"
	default:
		return "unknown"
	}
}

// outputLevel reports the level new calls should start at.
func (c *Client) outputLevel() outputLevel {
	return outputLevel(c.level.Load())
}

// downgrade records that everything at or above the given level is
// unsupported. It never raises the level back: a provider that rejected a
// response format once is not asked again for the life of the client, so a
// fanned-out evaluation pays the discovery cost once instead of once per call.
func (c *Client) downgrade(to outputLevel) {
	for {
		current := c.level.Load()
		if int32(to) <= current {
			return
		}
		if c.level.CompareAndSwap(current, int32(to)) {
			c.log.Debug("inference: structured output downgraded",
				"from", outputLevel(current).String(),
				"to", to.String())
			return
		}
	}
}

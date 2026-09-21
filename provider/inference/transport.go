package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	// completionsPath is appended to the configured base URL.
	completionsPath = "/chat/completions"
	// maxErrorBody is how much of a failing response is kept on an
	// [APIError] for diagnosis.
	maxErrorBody = 512
	// maxReadBody bounds how much of any response body is read at all, so a
	// misbehaving endpoint cannot stream unbounded memory into the process.
	maxReadBody = 1 << 20
	// baseBackoff is the first retry delay, doubled per attempt.
	baseBackoff = 500 * time.Millisecond
	// maxBackoff caps both the computed backoff and an honoured
	// Retry-After.
	maxBackoff = 30 * time.Second
	// maxBackoffShift bounds the doubling so the shift cannot overflow.
	maxBackoffShift = 16
)

// APIError is a non-2xx response from the endpoint.
type APIError struct {
	// StatusCode is the HTTP status.
	StatusCode int
	// Type and Code are the provider's own classification, when it sends
	// one.
	Type string
	Code string
	// Message is the provider's message, falling back to the raw body and
	// then to the HTTP status text.
	Message string
	// Body is the raw response body, truncated, with the API key removed.
	Body string

	// retryAfter carries a Retry-After header through to the backoff
	// calculation, which runs after the response has been closed.
	retryAfter    time.Duration
	hasRetryAfter bool
}

// Error implements error. It never contains the API key.
func (e *APIError) Error() string {
	var b strings.Builder
	b.WriteString("inference: http ")
	b.WriteString(strconv.Itoa(e.StatusCode))
	if e.Message != "" {
		b.WriteString(": ")
		b.WriteString(e.Message)
	}
	var extra []string
	if e.Type != "" {
		extra = append(extra, "type="+e.Type)
	}
	if e.Code != "" {
		extra = append(extra, "code="+e.Code)
	}
	if len(extra) > 0 {
		b.WriteString(" (")
		b.WriteString(strings.Join(extra, ", "))
		b.WriteString(")")
	}
	return b.String()
}

// nonRetryable marks an error that must not be retried even though it is not
// an [APIError] — a malformed success body, for instance.
type nonRetryable struct{ err error }

func (e *nonRetryable) Error() string { return e.err.Error() }
func (e *nonRetryable) Unwrap() error { return e.err }

// complete POSTs one chat completion, retrying transient failures.
func (c *Client) complete(ctx context.Context, body chatRequest) (*chatResponse, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("inference: encode request: %w", err)
	}
	endpoint := c.baseURL + completionsPath
	attempts := c.maxRetries + 1

	var last error
	for attempt := range attempts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resp, err := c.attempt(ctx, endpoint, payload, body.Model, attempt)
		if err == nil {
			return resp, nil
		}
		last = err
		if attempt == attempts-1 || !retryable(err) {
			return nil, err
		}
		if serr := c.sleep(ctx, backoffFor(attempt, err)); serr != nil {
			return nil, fmt.Errorf("inference: retry abandoned: %w (last attempt: %w)", serr, last)
		}
	}
	return nil, last
}

// attempt performs one HTTP round trip.
func (c *Client) attempt(ctx context.Context, endpoint string, payload []byte, model string, attempt int) (*chatResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, &nonRetryable{fmt.Errorf("inference: build request: %w", err)}
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c.referer != "" {
		req.Header.Set("HTTP-Referer", c.referer)
	}
	if c.title != "" {
		req.Header.Set("X-Title", c.title)
	}

	start := time.Now()
	resp, err := c.httpClient.Do(req)
	if err != nil {
		c.log.DebugContext(ctx, "inference: chat completion failed",
			"model", model,
			"attempt", attempt+1,
			"latency_ms", time.Since(start).Milliseconds())
		return nil, fmt.Errorf("inference: chat completion: %w", err)
	}
	defer drain(resp)

	c.log.DebugContext(ctx, "inference: chat completion",
		"model", model,
		"attempt", attempt+1,
		"status", resp.StatusCode,
		"latency_ms", time.Since(start).Milliseconds())

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, c.apiError(resp)
	}

	var out chatResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxReadBody)).Decode(&out); err != nil {
		return nil, &nonRetryable{fmt.Errorf("inference: decode response: %w", err)}
	}
	return &out, nil
}

// drain empties and closes a response body so the connection returns to the
// idle pool instead of being torn down.
func drain(resp *http.Response) {
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxReadBody))
	_ = resp.Body.Close()
}

// retryable reports whether an attempt is worth repeating: a 408, a 429, a
// 5xx, or a transport failure. Every 4xx, a cancelled context, and a malformed
// success body are final.
//
// A 501 is the one 5xx that is final too. "Not implemented" is a statement
// about the endpoint rather than about this moment, so repeating the call
// cannot change it — and because a 501 does send the client a level lower (see
// [unsupportedShape]), retrying it would multiply the whole negotiation by the
// retry count.
func retryable(err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var final *nonRetryable
	if errors.As(err, &final) {
		return false
	}
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case http.StatusRequestTimeout, http.StatusTooManyRequests:
			return true
		case http.StatusNotImplemented:
			return false
		}
		return apiErr.StatusCode >= 500
	}
	// Anything left is a transport error: DNS, dial, reset, truncated read.
	return true
}

// unsupportedShape reports whether an error plausibly means "this provider
// does not understand the request I sent" rather than something unrelated to
// the request's shape. Only such an error is worth retrying a step lower down
// the structured-output levels.
//
// The list is deliberately short. A 400 and a 422 are how a provider rejects a
// field it cannot honour, and a 501 is how it says the feature is not
// implemented. Every other status says something else entirely and must not
// send the client probing: a 404 is a wrong base URL or an unknown model, a
// 405 a wrong method, a 413 a prompt that is too long, a 409 or a 423 a
// conflict, and an auth failure, a timeout, a rate limit or a 5xx have nothing
// to do with the request document at all. Sending a downgrade probe for those
// costs two extra calls and answers a question nobody asked.
func unsupportedShape(err error) bool {
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		return false
	}
	switch apiErr.StatusCode {
	case http.StatusBadRequest, http.StatusUnprocessableEntity, http.StatusNotImplemented:
		return true
	}
	return false
}

// backoffFor computes the delay before the next attempt: an honoured
// Retry-After when the provider sent one, otherwise exponential backoff with
// equal jitter, both capped.
func backoffFor(attempt int, err error) time.Duration {
	var apiErr *APIError
	if errors.As(err, &apiErr) && apiErr.hasRetryAfter {
		return min(max(apiErr.retryAfter, 0), maxBackoff)
	}
	shift := min(attempt, maxBackoffShift)
	delay := min(baseBackoff<<shift, maxBackoff)
	half := delay / 2
	return half + time.Duration(rand.Int64N(int64(half)+1))
}

// retryAfterFrom reads a Retry-After header in either of its forms: a number
// of seconds, or an HTTP date.
func retryAfterFrom(h http.Header) (time.Duration, bool) {
	raw := strings.TrimSpace(h.Get("Retry-After"))
	if raw == "" {
		return 0, false
	}
	if secs, err := strconv.ParseFloat(raw, 64); err == nil {
		return max(time.Duration(secs*float64(time.Second)), 0), true
	}
	if when, err := http.ParseTime(raw); err == nil {
		return max(time.Until(when), 0), true
	}
	return 0, false
}

// errorEnvelope is the union of the error shapes seen in the wild. Every field
// is decoded loosely because the failing path is exactly where providers
// diverge most.
type errorEnvelope struct {
	Error   json.RawMessage `json:"error"`
	Message flexString      `json:"message"`
	Detail  flexString      `json:"detail"`
}

// errorDetail is the usual {"error":{...}} object.
type errorDetail struct {
	Message flexString `json:"message"`
	Type    flexString `json:"type"`
	Code    flexString `json:"code"`
}

// flexString decodes a JSON string, number, boolean or null into text, because
// "code" in particular comes back as a string from some endpoints and as a
// number from others.
type flexString string

// UnmarshalJSON implements [json.Unmarshaler]. It never returns an error.
func (f *flexString) UnmarshalJSON(data []byte) error {
	*f = ""
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var s string
	if err := json.Unmarshal(trimmed, &s); err == nil {
		*f = flexString(s)
		return nil
	}
	if trimmed[0] != '{' && trimmed[0] != '[' {
		*f = flexString(trimmed)
	}
	return nil
}

// apiError reads a non-2xx response into an [APIError].
func (c *Client) apiError(resp *http.Response) *APIError {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxReadBody))
	text := c.redact(string(raw))

	out := &APIError{
		StatusCode: resp.StatusCode,
		Body:       truncate(text, maxErrorBody),
	}

	var env errorEnvelope
	if err := json.Unmarshal(raw, &env); err == nil {
		var detail errorDetail
		if len(env.Error) > 0 {
			if err := json.Unmarshal(env.Error, &detail); err != nil {
				// {"error": "plain text"} and friends.
				var flat flexString
				_ = flat.UnmarshalJSON(env.Error)
				detail.Message = flat
			}
		}
		out.Message = string(detail.Message)
		out.Type = string(detail.Type)
		out.Code = string(detail.Code)
		if out.Message == "" {
			out.Message = string(env.Message)
		}
		if out.Message == "" {
			out.Message = string(env.Detail)
		}
	}
	if out.Message == "" {
		// A body that is a bare JSON string, e.g. "upstream unavailable".
		var bare string
		if err := json.Unmarshal(raw, &bare); err == nil {
			out.Message = bare
		}
	}
	if out.Message == "" {
		// A body that is not JSON at all, e.g. an nginx error page.
		out.Message = strings.TrimSpace(out.Body)
	}
	if out.Message == "" {
		out.Message = http.StatusText(resp.StatusCode)
	}

	out.Message = truncate(c.redact(out.Message), maxErrorBody)
	out.Type = c.redact(out.Type)
	out.Code = c.redact(out.Code)

	if d, ok := retryAfterFrom(resp.Header); ok {
		out.retryAfter, out.hasRetryAfter = d, true
	}
	return out
}

// truncate cuts s to at most limit bytes, never splitting a rune, and marks
// that it did so.
func truncate(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	cut := s[:limit]
	for len(cut) > 0 && !utf8.ValidString(cut) {
		cut = cut[:len(cut)-1]
	}
	return cut + "…(truncated)"
}

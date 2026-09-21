package config

import (
	"errors"
	"net/url"
	"strings"
)

// ValidateBaseURL reports why raw could never be called as an OpenAI-compatible
// API root, and nil when it could.
//
// The returned message is a fragment with no subject, so that each caller can
// name the thing at fault — the environment variable at startup, the body
// field in a request. It never repeats the value: a base URL may carry a
// credential in its userinfo or its query, and this check runs in places that
// do not yet know to scrub one.
func ValidateBaseURL(raw string) error {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return errors.New("is not a valid URL")
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return errors.New("must be an absolute http or https URL")
	}
	if u.Host == "" {
		return errors.New("must include a host")
	}
	return nil
}

// DisplayBaseURL renders a base URL for a response body or a log line.
//
// Operators are invited to point PERCEPTEA_INFERENCE_BASE_URL at a proxy or a
// gateway, and a gateway URL routinely carries the credential: in the userinfo
// (https://svc:sk-secret@gateway/v1) or in the query (?key=...). Neither is
// anybody else's business, so both are dropped, along with the fragment, and
// what is left still says where calls are going.
func DisplayBaseURL(raw string) string {
	trimmed := strings.TrimSpace(raw)
	u, err := url.Parse(trimmed)
	if err != nil {
		// Unparseable, so there is no structure to strip a credential from.
		// Saying nothing is better than guessing which half is the secret.
		return "[unparseable url]"
	}
	if u.User == nil && u.RawQuery == "" && !u.ForceQuery && u.Fragment == "" {
		// Nothing to strip: return the original text rather than url.URL's
		// re-encoding of it.
		return trimmed
	}
	stripped := *u
	stripped.User = nil
	stripped.RawQuery = ""
	stripped.ForceQuery = false
	stripped.Fragment = ""
	stripped.RawFragment = ""
	return stripped.String()
}

// CredentialsIn lists the parts of a base URL that are secrets rather than
// addressing: the userinfo, and the query, which is where a gateway key
// usually sits.
//
// [DisplayBaseURL] keeps these out of anything this service renders itself,
// but an error from somewhere else can carry the URL whole — net/http masks
// the password in a *url.Error and leaves the query alone — so what comes
// back here is fed to the same scrubbing a key gets.
//
// Two edges are accepted deliberately. Percent-encoding is not unpicked, so a
// credential that reaches a message in a different encoding from the one
// configured is not caught. And a query that holds no secret at all is
// redacted anyway, along with a userinfo username that happens to be an
// ordinary word: over-redacting a message is a cosmetic cost, and the other
// way round is a leak.
func CredentialsIn(raw string) []string {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return nil
	}
	var out []string
	if u.User != nil {
		if password, ok := u.User.Password(); ok && password != "" {
			out = append(out, password)
		}
		if name := u.User.Username(); name != "" {
			out = append(out, name)
		}
	}
	if u.RawQuery != "" {
		out = append(out, u.RawQuery)
	}
	return out
}

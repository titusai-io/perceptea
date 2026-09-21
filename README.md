# Perceptea

A Go REST classifier API: post a **state** and a set of typed **questions**, get
typed **answers** with probability distributions.

It answers with typed decisions a program can branch on, rather than prose:
the answer space is declared up front and the model only ever scores
candidates from it.

The whole thing is standard library only — no third-party modules, in the
service or in its tests.

```
POST /api/evaluate      classify a state against declared questions
GET  /api/health        liveness, active provider and model
GET  /api/providers     the built-in provider presets
```

## How the parallel sampler works

The usual way to make a model classify something is to ask it for one large
JSON document describing every answer. Models drop keys, invent labels and
emit invalid JSON. Perceptea never asks for a document.

Instead the answer space is **declared up front**, and every candidate answer
in it becomes one tiny, constrained call:

```
statement = The correct which team should handle this is "billing" (Charges, refunds, invoices).
reply     = {"p": 0.87}
```

1. **One call per candidate.** A choice with 4 options is 4 calls; a score with
   4 levels is 4 calls; a noul is 1. Questions are evaluated in the same wave,
   bounded by `PERCEPTEA_MAX_CONCURRENCY`.
2. **Independent probabilities.** Each `p` is estimated on its own, against the
   same state, with no knowledge of its rivals — that is what makes them
   comparable rather than a single sampled guess.
3. **Logit, then softmax.** Each `p` is clamped into `(0.001, 0.999)`, mapped
   to `log(p / (1 - p))`, and the logits of one question's candidates are
   softmaxed into a distribution that sums to 1.
4. **Read off the answer.** A **choice** takes the argmax (the earliest
   declared option wins an exact tie). A **score** takes the expected value
   over the level indices, so a rating may land between two declared levels. A
   **noul** skips the normalisation entirely and reports its bare probability.

`confidence`, reported for choice and score answers, is half the winning
probability plus half of its margin over the runner-up, offset so that a
two-way coin flip scores 0.5.

Each call is small and constrained, so the failure mode of a big structured
generation — a dropped key, an invented label — cannot happen. The cost is
that the number of calls grows with the number of declared candidates.

## Quick start

```bash
cp .env.example .env          # then put a key in it
export OPENAI_API_KEY=sk-...  # or just this
go run ./cmd/perceptea
```

```
time=2026-09-20T21:12:58.122-04:00 level=INFO msg="perceptea listening" addr=[::]:8080
  provider=openai base_url=https://api.openai.com/v1 model=gpt-4o-mini
  api_key_configured=true allow_request_credentials=true request_timeout=1m0s
  max_concurrency=8
```

Then one line per request, and nothing else — never the key, never the state,
never the questions:

```
level=INFO msg=request method=POST path=/api/evaluate status=200
  duration=713.204ms questions=3 mode=parallel
```

Point it somewhere else without touching the code:

```bash
PERCEPTEA_PROVIDER=openrouter OPENROUTER_API_KEY=sk-or-... go run ./cmd/perceptea
PERCEPTEA_BASE_URL=http://localhost:11434/v1 PERCEPTEA_API_KEY=ollama go run ./cmd/perceptea
go run ./cmd/perceptea -addr 127.0.0.1:9000
```

A local model server usually wants a throwaway key rather than no key —
`ollama`, `EMPTY`, `lm-studio`. Short as they are, they are still scrubbed out
of responses and log lines.

## Configuration

Everything is an environment variable; `-addr` is the one flag. A `.env` file
in the working directory is read at startup, and never overrides a variable
that is already set. A **missing API key is not a startup error** — the server
comes up and rejects the requests that would need one with a 401.

| Variable | Default | What it does |
|---|---|---|
| `PERCEPTEA_ADDR` | `:8080` | Listen address. `-addr` overrides it. |
| `PERCEPTEA_PROVIDER` | `openai` | Which preset to use (see below). |
| `PERCEPTEA_BASE_URL` | the preset's | Override the endpoint: a proxy, a gateway, a local server. Must be an absolute `http` or `https` URL. A credential in it — userinfo, or a `?key=` — never reaches a response or a log line. |
| `PERCEPTEA_API_KEY` | — | The key, whatever the provider. Checked first. |
| *the preset's key* | — | `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPINFRA_API_KEY` or `ZAI_API_KEY`, for the active provider only. |
| `PERCEPTEA_MODEL` | the preset's | Override the model id. |
| `PERCEPTEA_TEMPERATURE` | `0` | Sampling temperature. 0 is the only reproducible setting. |
| `PERCEPTEA_MAX_CONCURRENCY` | `8` | Scoring calls in flight per evaluation. |
| `PERCEPTEA_REQUEST_TIMEOUT` | `60s` | Deadline for one request, end to end. |
| `PERCEPTEA_MAX_BODY_BYTES` | `2097152` | Request body limit, 2 MiB. A state large enough to exceed it will not fit a model's context either. |
| `PERCEPTEA_ALLOW_REQUEST_CREDENTIALS` | `true` | Whether a request body may carry `api_key` and `base_url`. **On, the service is an open proxy**: any caller can name an endpoint and have the service call it, on the server's key if they send none of their own. Right on a laptop; set it to `false` anywhere else. |
| `PERCEPTEA_ALLOWED_BASE_URLS` | — | Comma-separated URL prefixes a request body's `base_url` must start with; empty means any. Bounds the row above without switching it off. Never applies to `PERCEPTEA_BASE_URL`. |
| `PERCEPTEA_LOG_LEVEL` | `info` | `debug`, `info`, `warn`, `error`. |
| `PERCEPTEA_LOG_FORMAT` | `text` | `text` or `json`. |

An unknown provider, an unparseable number or duration, a base URL that could
never be called, or a bad log level stops the process with an error naming the
variable.

A prefix in `PERCEPTEA_ALLOWED_BASE_URLS` is matched as text against the URL
with its scheme and host lower-cased, so end each one at a host or a path
separator: `https://api.example` would also match
`https://api.example.attacker.test`.

### Timeouts that are not configurable

`PERCEPTEA_REQUEST_TIMEOUT` bounds one evaluation. Four more are fixed, and
the first of them is visible to callers:

| Timeout | Value | What it bounds |
|---|---|---|
| read | 30s | Headers **and** body. A body still arriving after 30s gets a 504. |
| read header | 10s | The request line and headers on their own. |
| write | request timeout + 30s | The response, with room for a whole evaluation before it. |
| idle | 120s | A kept-alive connection between requests. |
| shutdown grace | request timeout + 5s, at least 30s | How long a `SIGTERM` waits for requests in flight. Never shorter than a request is allowed to be. |

### Providers

| Name | Base URL | Default model | Key variable |
|---|---|---|---|
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1` | `z-ai/glm-5.3-flash` | `OPENROUTER_API_KEY` |
| `deepinfra` | `https://api.deepinfra.com/v1/openai` | `zai-org/GLM-5.3-Flash` | `DEEPINFRA_API_KEY` |
| `zai` | `https://api.z.ai/api/paas/v4` | `glm-5.3-flash` | `ZAI_API_KEY` |

> **The DeepInfra model id is a best-effort guess** at that provider's naming
> and has not been verified against their catalogue. If it 404s, set
> `PERCEPTEA_MODEL` to the id DeepInfra actually publishes — that is what the
> variable is for. The same goes for any preset whose default model is retired.

A preset is only a convenience. Any endpoint that speaks the OpenAI chat
completions API works with `PERCEPTEA_BASE_URL`.

## API

### `POST /api/evaluate`

```bash
curl -s localhost:8080/api/evaluate \
  -H 'content-type: application/json' \
  -d '{
    "state": "Charged twice again!! Second month in a row.",
    "mode": "parallel",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Charges, refunds, invoices",
          "technical": "Bugs or product issues",
          "other": "Doesn'\''t fit"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["Low", "Medium", "High", "Critical"]
      },
      "angry": {
        "type": "noul",
        "instructions": "Strong frustration or anger?"
      }
    },
    "model": "gpt-4o-mini"
  }' | jq
```

```json
{
  "model": "gpt-4o-mini",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.82,
      "probabilities": { "billing": 0.71, "technical": 0.12, "other": 0.17 }
    },
    "urgency": {
      "type": "score",
      "score": 2.4,
      "confidence": 0.75,
      "legend": { "0": "Low", "1": "Medium", "2": "High", "3": "Critical" },
      "probabilities": { "0": 0.05, "1": 0.15, "2": 0.45, "3": 0.35 }
    },
    "angry": { "type": "noul", "noul": 0.88 }
  },
  "usage": { "input_tokens": 1840, "output_tokens": 96 },
  "meta": { "mode": "parallel", "latency_ms": 620, "parallel_calls": 9 }
}
```

The service answers in compact JSON; the examples here are piped through `jq`.
Answers come back in the order the questions were declared, and a choice's
probabilities in the order its options were declared.

**Request fields**

| Field | Required | Notes |
|---|---|---|
| `state` | yes | A string, object or array — whatever the questions are about. An explicit `null` counts as absent. Objects keep their key order into the prompt. |
| `questions` | yes | At least one. See the question types below. |
| `model` | no | Falls back to `PERCEPTEA_MODEL`, then the preset's default. |
| `temperature` | no | Falls back to `PERCEPTEA_TEMPERATURE`. An explicit `0` is honoured. |
| `mode` | no | `parallel` (default) or `oneshot`. |
| `api_key` | no | Only when `PERCEPTEA_ALLOW_REQUEST_CREDENTIALS` is on; otherwise ignored silently. |
| `base_url` | no | Same, and it must also pass `PERCEPTEA_ALLOWED_BASE_URLS` when that is set. |

A field the service does not know is a `400 invalid_json` naming it, and so is
anything after the JSON object. Silently ignoring `apikey` would mean spending
the *server's* credential on a request that meant to bring its own. Field
names are matched case-insensitively, so `Api_Key` is understood rather than
rejected. Inside a question, unknown keys are still ignored.

`mode: "oneshot"` asks the model for one JSON document covering every question,
the way most classifiers do it. It exists for comparison and is markedly less
reliable. It also needs a backend that can do plain chat completions: the
provider shipped here can, so only a `classifier.Scorer` of your own can run
into that refusal.

### `GET /api/health`

```bash
curl -s localhost:8080/api/health
```

```json
{"ok":true,"service":"perceptea","provider":"openai","model":"gpt-4o-mini","api_key_configured":true}
```

`api_key_configured` says whether the *server* holds a key. Neither the key
nor a credential embedded in a base URL is ever in a response, a log line or
an error message: `/api/providers` reports the endpoint with its userinfo and
query stripped, and both keys — the server's and one sent in a request body —
are scrubbed out of every message on its way to a caller or a log.

### `GET /api/providers`

```bash
curl -s localhost:8080/api/providers | jq
```

```json
{
  "active": {
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "api_key_configured": true,
    "allow_request_credentials": true
  },
  "providers": [
    {
      "name": "openai",
      "base_url": "https://api.openai.com/v1",
      "default_model": "gpt-4o-mini",
      "env_key": "OPENAI_API_KEY",
      "active": true
    },
    {
      "name": "openrouter",
      "base_url": "https://openrouter.ai/api/v1",
      "default_model": "z-ai/glm-5.3-flash",
      "env_key": "OPENROUTER_API_KEY",
      "active": false
    },
    {
      "name": "deepinfra",
      "base_url": "https://api.deepinfra.com/v1/openai",
      "default_model": "zai-org/GLM-5.3-Flash",
      "env_key": "DEEPINFRA_API_KEY",
      "active": false
    },
    {
      "name": "zai",
      "base_url": "https://api.z.ai/api/paas/v4",
      "default_model": "glm-5.3-flash",
      "env_key": "ZAI_API_KEY",
      "active": false
    }
  ]
}
```

`env_key` is the *name* of a variable, never its value.

### Errors

Every failure but one answers with the same body, so a client can branch on
`code` and never on prose:

```json
{"error":"missing \"state\"","code":"invalid_request"}
```

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_json` | The body is not valid JSON, a question is malformed, a field is not one this endpoint takes, or something follows the JSON object. |
| 400 | `invalid_request` | No `state`, no `questions`, a question the classifier rejects, or a `base_url` that is unusable or not allowed. |
| 400 | `unsupported_mode` | A mode that is not `parallel` or `oneshot`. (The provider shipped here can do both, so the "this provider cannot" half needs a backend of your own.) |
| 401 | `missing_api_key` | No key could be resolved for the request. |
| 404 | `not_found` | No such endpoint. |
| 405 | `method_not_allowed` | Wrong method for the route. The `Allow` header names the right one. |
| 413 | `payload_too_large` | The body is over `PERCEPTEA_MAX_BODY_BYTES`. |
| 429 | `upstream_rate_limited` | The provider rate limited us. Reported even when the wait for it ran into the request deadline. |
| 499 | — | The caller hung up, or the body stopped arriving. No body is sent. |
| 500 | `internal` | A bug, a panic included. The detail goes to the log, not to the caller. |
| 502 | `upstream_error` | The provider failed, or could not be reached. |
| 504 | `timeout` | The evaluation outran `PERCEPTEA_REQUEST_TIMEOUT`, or the body did not arrive before the read timeout. |

A failure that is about the connection rather than the document — a body that
timed out, a caller that went away — says nothing more on the wire: the
underlying error names this server's own socket, so it goes to the log line
instead.

## Question types

Each question is one key in the `questions` object; the key is the question's
name, and answers come back under the same key.

### `choice` — pick one declared option

```json
{ "type": "choice",
  "instructions": "Which team should handle this?",
  "criteria": { "billing": "Charges, refunds, invoices", "technical": "Bugs" } }
```

`criteria` maps **1 to 255** option keys to a description shown to the model.
The answer carries `choice` (the winning key), `confidence` and the full
`probabilities` distribution. Declaration order decides ties.

### `score` — rate on a declared scale

```json
{ "type": "score", "instructions": "How urgent?",
  "criteria": ["Low", "Medium", "High", "Critical"] }
```

`criteria` is an ordered array of **2 to 10** level labels, whose indices are
the scale. The answer carries `score` — the expected value over those indices,
so `2.4` means "between High and Critical, nearer High" — plus `confidence`,
the `legend` mapping indices to labels, and the `probabilities`.

### `noul` — a single proposition

```json
{ "type": "noul", "instructions": "Strong frustration or anger?" }
```

No criteria. The answer is `noul`: the bare probability that the proposition
holds, with no separate confidence, because the probability *is* the
confidence. `"type": "boolean"` is accepted as a synonym on input and always
reported back as `noul`.

## Using it as a library

The HTTP layer is a thin shell. The classifier and the provider client are
ordinary packages:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/provider/openai"
)

func main() {
	client, err := openai.New(openai.Config{
		APIKey:  os.Getenv("OPENAI_API_KEY"),
		BaseURL: "https://api.openai.com/v1",
		Model:   "gpt-4o-mini",
	})
	if err != nil {
		panic(err)
	}

	questions := classifier.NewOrderedMap[classifier.Question]()
	options := classifier.NewOrderedMap[string]()
	options.Set("billing", "Charges, refunds, invoices")
	options.Set("technical", "Bugs or product issues")
	questions.Set("department", classifier.Question{
		Type:         classifier.TypeChoice,
		Instructions: "Which team should handle this?",
		Options:      *options,
	})
	questions.Set("angry", classifier.Question{
		Type:         classifier.TypeNoul,
		Instructions: "Strong frustration or anger?",
	})

	resp, err := classifier.New(client, classifier.WithMaxConcurrency(8)).
		Evaluate(context.Background(), classifier.Request{
			State:     classifier.StringState("Charged twice again!! Second month in a row."),
			Questions: *questions,
			Model:     "gpt-4o-mini",
		})
	if err != nil {
		panic(err)
	}

	out, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(out))
}
```

`OrderedMap` is a JSON object that remembers its key order, because the order
of a choice's options is part of the contract: it fixes the order statements
are built in and decides an exact tie.

Anything that implements `classifier.Scorer` — a single
`Score(ctx, ScoreRequest) (ScoreResult, error)` method — can stand in for the
provider, in a test or against a model of your own.

## Testing

```bash
go test ./...            # everything
go test -race ./...      # what CI runs
go vet ./...
gofmt -l .               # silence means formatted
```

No test in this repository makes a network call. The HTTP layer builds its
evaluator through a seam (`api.Options.NewEvaluator`), so its tests inject a
fake and assert on what the server would have asked for; the provider package
tests its client against `httptest` servers.

CI runs `gofmt`, `go mod tidy` (with `git diff --exit-code` behind it, so the
zero-dependency claim at the top of this file is checked rather than trusted),
`go vet`, `go build` and `go test -race` on every push and pull request.

## Limitations

- **The probabilities are normalised opinions, not calibrated ones.** A
  distribution that reads `0.71 / 0.12 / 0.17` says the model preferred the
  first option by roughly that much on this state. It is not a frequency, and
  nothing here has been fitted to outcomes. Compare them across candidates,
  not across states, and do not feed them into anything that assumes a
  calibrated prior.
- **Cost and latency scale with the number of declared candidates.** A request
  with three questions and nine candidates is nine model calls. Ten questions
  of ten options each is a hundred. `PERCEPTEA_MAX_CONCURRENCY` bounds how
  many run at once, not how many run.
- **`PERCEPTEA_ALLOW_REQUEST_CREDENTIALS` turns the service into an open
  proxy.** With it on, any caller can supply `base_url` and `api_key` and have
  the service make the call for them — to any host it can reach, including
  ones only it can reach. That is exactly what you want on a laptop, where
  pointing at a local model server is the point, and exactly what you do not
  want anywhere else: set it to `false` outside development, or list the
  endpoints you meant in `PERCEPTEA_ALLOWED_BASE_URLS`. The service has no
  authentication of its own — put it behind something that does.
- **A purpose-built classifier model would be faster and cheaper.** This is
  ordinary chat models used as micro-scorers; its latency and calibration are
  those of the model you point it at, and one call per candidate is a worse
  deal than a model that scores the whole answer space in a single pass.
- **`oneshot` is a comparison baseline**, not a supported path. Use `parallel`.

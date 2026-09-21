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
GET  /api/health        liveness, plus the endpoint and model in use
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
cp .env.example .env             # then put a key in it
export PERCEPTEA_API_KEY=sk-...  # or just this
go run ./cmd/perceptea
```

```
time=2026-09-20T21:12:58.122-04:00 level=INFO msg="perceptea listening" addr=[::]:8080
  inference_base_url=https://api.deepinfra.com/v1/openai
  model=Qwen/Qwen3.8-Flash
  api_key_configured=true allow_request_credentials=true request_timeout=1m0s
  max_concurrency=8
```

Then one line per request, and nothing else — never the key, never the state,
never the questions:

```
level=INFO msg=request method=POST path=/api/evaluate status=200
  duration=713.204ms questions=3 mode=parallel
```

Point it somewhere else without touching the code — an endpoint and the id of
a model that endpoint serves:

```bash
PERCEPTEA_INFERENCE_BASE_URL=https://openrouter.ai/api/v1 \
  PERCEPTEA_MODEL=meta-llama/llama-3.1-8b-instruct \
  PERCEPTEA_API_KEY=sk-or-... go run ./cmd/perceptea

PERCEPTEA_INFERENCE_BASE_URL=http://localhost:11434/v1 \
  PERCEPTEA_MODEL=llama3.1 \
  PERCEPTEA_API_KEY=ollama go run ./cmd/perceptea

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
| `PERCEPTEA_INFERENCE_BASE_URL` | `https://api.deepinfra.com/v1/openai` | The API root of the service that runs the model — see below. Must be an absolute `http` or `https` URL. A credential in it — userinfo, or a `?key=` — never reaches a response or a log line. |
| `PERCEPTEA_API_KEY` | — | The key sent to that endpoint as a bearer token. The only variable a key is read from. |
| `PERCEPTEA_MODEL` | `Qwen/Qwen3.8-Flash` | The model id to score with. It has to be one the endpoint above serves, and it should be one that does not reason — see [Choosing a model](#choosing-a-model). |
| `PERCEPTEA_REASONING_EFFORT` | — | Sent to the provider as `reasoning_effort`: `none`, `low`, `medium` or `high`. Unset sends no reasoning field at all. See [Choosing a model](#choosing-a-model). |
| `PERCEPTEA_TEMPERATURE` | `0` | Sampling temperature. 0 is the only reproducible setting. |
| `PERCEPTEA_MAX_CONCURRENCY` | `8` | Scoring calls in flight per evaluation. |
| `PERCEPTEA_REQUEST_TIMEOUT` | `60s` | Deadline for one request, end to end. |
| `PERCEPTEA_MAX_BODY_BYTES` | `2097152` | Request body limit, 2 MiB. A state large enough to exceed it will not fit a model's context either. |
| `PERCEPTEA_ALLOW_REQUEST_CREDENTIALS` | `true` | Whether a request body may carry `api_key` and `inference_base_url`. **On, the service is an open proxy**: any caller can name an endpoint and have the service call it, on the server's key if they send none of their own. Right on a laptop; set it to `false` anywhere else. |
| `PERCEPTEA_LOG_LEVEL` | `info` | `debug`, `info`, `warn`, `error`. |
| `PERCEPTEA_LOG_FORMAT` | `text` | `text` or `json`. |

An unparseable number or duration, a base URL that could never be called, or a
bad log level stops the process with an error naming the variable.

### What the inference base URL is

`PERCEPTEA_INFERENCE_BASE_URL` is the **API root of the service that actually
runs the model** — the hosted endpoint you have a key for, or the local model
server on your machine. Perceptea appends the path to it:

```
<inference base URL> + /chat/completions
https://api.deepinfra.com/v1/openai  →  https://api.deepinfra.com/v1/openai/chat/completions
http://localhost:11434/v1            →  http://localhost:11434/v1/chat/completions
```

That call is made **once per declared candidate answer** — four options is
four calls, each one scoring a single statement — so the endpoint you name
here is where the whole fan-out of an evaluation lands, and the model id in
`PERCEPTEA_MODEL` has to be one it serves. Include the version segment the
endpoint publishes (`/v1`, or whatever it uses) and no trailing slash.

### Endpoints you can point it at

Reference only — **this table is documentation, not configuration.** There is
nothing to select: set the two variables to a row you like, or to anything
else that speaks the same chat completions API.

| Where the model runs | `PERCEPTEA_INFERENCE_BASE_URL` | An id for `PERCEPTEA_MODEL` |
|---|---|---|
| DeepInfra | `https://api.deepinfra.com/v1/openai` | `Qwen/Qwen3.8-Flash` |
| OpenRouter | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.1-8b-instruct` |
| Z.ai | `https://api.z.ai/api/paas/v4` | `glm-5.3-flash` (a reasoning model — read the next section first) |
| Ollama, locally | `http://localhost:11434/v1` | whatever you have pulled, e.g. `llama3.1` |
| vLLM, locally | `http://localhost:8000/v1` | the id the server was started with |

One model is published under a different id by each of them — the first two
rows are the same weights — which is why the model is a setting of its own
and not something derived from the endpoint. A hosted endpoint wants its own
key in `PERCEPTEA_API_KEY`; a local one usually wants a throwaway string
rather than nothing at all.

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

## Choosing a model

Most services pick a model by capability. This one should not, because the
shape of the work is unusual and that shape decides almost everything.

### What this workload actually looks like

**One call per declared candidate answer.** A request with a choice of 4, a
score of 4 and a noul is **9 calls**, not one. Every one of them re-sends the
whole state, and every one of them asks for the same ten characters back:

```
9 calls × (system prompt + state + one statement)   ← the bill
9 × {"p": 0.87}                                     ← the output
```

So **input tokens dominate and output is about ten tokens per call.** A
provider's output price, the number most model comparisons lead with, is
nearly irrelevant here. Input price and latency are what you are buying, and
latency matters twice over because the calls fan out: the request takes as
long as the slowest wave of `PERCEPTEA_MAX_CONCURRENCY` calls, not as long as
one.

### Reasoning models are the trap

A scoring call sets `max_tokens: 32`. That is room for `{"p": 0.87}` and
nothing else, and it is deliberate: it is what stops a chatty model turning
one score into an essay.

A model that thinks before it answers spends that budget on thinking tokens.
The reply is cut off before the JSON, and because the cap is the same on
every call, **it happens to every candidate in the request at once.**

Unguarded, that is the worst failure a classifier can have. An unreadable
reply degrades to the neutral 0.5, so every candidate scores 0.5, the softmax
turns nine identical scores into a flat distribution, and the service answers
200 with a well-formed, confident-looking document that is pure noise — and
nothing in the log says otherwise.

So it is guarded: a reply cut off before it said anything readable is a
**502 `upstream_error`** whose message says the reply was cut off at the
output token limit, that this is what a reasoning model does under a small
limit, and what to change. A reply that did get its number out before the cap
is still accepted, and a reply that finished normally and said nothing
readable is still the one-off it always was, worth one 0.5 and no more.

**How to check before you point at one.** DeepInfra tags its models, and the
tags are on the model list and on each model's page. The three that matter:

- `non-reasoning` — safe, pick one of these.
- `can-disable-reasoning` — usable with `PERCEPTEA_REASONING_EFFORT=none`.
- `reasoning` with no `can-disable-reasoning` — the trap. It will be
  truncated on every scoring call, and there is nothing this service can send
  to stop it.

Other providers publish the same thing under other names; if you cannot find
out, send one request and see whether you get a 502.

**Check whether the id is deprecated, too.** A deprecated model on DeepInfra
carries a `replaced_by`, and a request naming it is served by the replacement
— which may not have the same capabilities. `meta-llama/Meta-Llama-3.1-8B-Instruct`
is tagged `structured-output` and is deprecated in favour of
`…-Instruct-Turbo`, which is not, so a request naming the first is answered by
the second with `json_schema response format is not supported for model` and a
405. The service steps down to a format the model does accept, so it works —
but the schema, the strongest guarantee on offer, is quietly not in play. Read
`deprecated` and `replaced_by` on the model you pick, and check the tags of
whatever actually serves it.

### What matters, in order

1. **Does it reason?** A yes is disqualifying unless the thinking can be
   turned off. Everything else is a preference; this one is a constraint.
2. **Does it support a JSON-schema response format?** The scorer asks for a
   strict `json_schema` first, steps down to `json_object`, then to plain
   text, and salvages a number out of prose if it has to. All three paths
   work — the negotiation is remembered per client, so it costs one probe —
   but the strict path is the one where the reply cannot be anything but a
   probability. DeepInfra tags these `structured-output`.
3. **Input price.** Nine calls of a few hundred tokens each, per request.
4. **Latency.** Small models answer in a fraction of the time a large one
   takes, and the fan-out multiplies the difference.
5. **General capability, last.** Each call asks one question: how likely is
   this one statement, given this one state. That is a judgement, not a
   reasoning task — there is no chain to follow and no document to keep
   coherent, which is the whole point of the parallel sampler. A small
   instruction-tuned model does this well.

### Good choices on DeepInfra

Prices are US dollars per million tokens, from DeepInfra's own model list.
Remember the shape of the bill: the input column is the one you are paying.

| Model | In $/M | Out $/M | Notes |
|---|---|---|---|
| **`Qwen/Qwen3.8-Flash`** | **0.113** | **0.382** | **The default.** Non-reasoning, structured output. Current-generation and needs no switch thrown. |
| `mistralai/Mistral-Nemo-Instruct-2407` | 0.019 | 0.030 | Non-reasoning, structured output, tools. The cheapest of the set; a small model, so a fine swap when the questions are easy. |
| `mistralai/Mistral-Small-24B-Instruct-2501` | 0.050 | 0.080 | Non-reasoning, structured output, built for low latency. |
| `Qwen/Qwen3-32B` | 0.080 | 0.280 | Non-reasoning, structured output. The cheapest current Qwen. |
| `google/gemma-3-27b-it` | 0.080 | 0.160 | Non-reasoning, structured output, tools. |
| `deepseek-ai/DeepSeek-V4-Flash` | 0.090 | 0.180 | `can-disable-reasoning` — set the effort to `none`, and see the caveat below. |
| `Qwen/Qwen3.8-27B` | 0.200 | 2.500 | `can-disable-reasoning`. Four times the input price of the default for the same job; see the caveat below. |
| `zai-org/GLM-5.3-Flash` | 0.150 | 0.500 | **Reasoning, and it cannot be disabled.** Every scoring reply is truncated at 32 tokens. Not usable for `parallel` mode as it stands. |

**Prefer `non-reasoning` over `can-disable-reasoning`.** A model that never
reasons cannot be misconfigured into truncating. A model that merely *can* be
told not to depends on the switch reaching it, and providers do not agree on
what the switch is: this service sends `reasoning_effort`, while much of the
Qwen line conventionally takes `chat_template_kwargs: {"enable_thinking":
false}`. If `reasoning_effort` is not mapped onto that, the setting does
nothing and every call truncates. The `non-reasoning` rows above have no such
dependency.

Note also that most of the current Qwen generation — `Qwen3.5-27B`,
`Qwen3.6-27B`, `Qwen3.5-4B`, `Qwen3.5-2B` and others — forces reasoning with
no way off. Within a family, the `Flash` and `Instruct` members are usually
the ones that do not.

Tags and prices move; check the list rather than this table when it matters.

### What it costs

Take a 300-token state and the 9-candidate request above. Each call carries
the system prompt, the state and one statement — call it 300 tokens — so the
request is roughly **2,700 input tokens and 90 output tokens**.

At the default model's prices that is 2,700 × $0.113/M + 90 × $0.382/M ≈
**$0.00034**, about a third of a tenth of a cent. A thousand requests is
thirty-four cents. Even the most expensive row above stays under a tenth of a
cent per request.

So cost is rarely the thing to optimise. **Reliability is**: a model that
returns a readable probability every time is worth far more than one that
saves you a hundredth of a cent and truncates.

### `PERCEPTEA_REASONING_EFFORT`

```bash
PERCEPTEA_REASONING_EFFORT=none go run ./cmd/perceptea
```

Unset — the default — sends **no reasoning field at all**, which is exactly
what a provider that has never seen one expects. Set to `none`, `low`,
`medium` or `high`, it is sent as `reasoning_effort` in every scoring and
generate request body — the name the OpenAI-compatible protocol settled on,
and the one DeepInfra accepts. Anything else stops the process at startup,
naming the variable and listing the four values.

It is server-side only: a request body cannot carry one, because the effort
describes the model the operator chose rather than the question being asked.

`chat_template_kwargs: {"enable_thinking": false}` is a second, widely
implemented way to ask for the same thing, and **this service does not send
it.** It is provider-specific where `reasoning_effort` is not, and an
endpoint that validates its request bodies strictly answers an unknown field
with a 400 — which would break every request against a provider that has
never heard of it, to help with one that has. If your model only understands
that switch, set it where the model is served, or use one that does not need
it.

### One honest caveat

None of these models is calibrated for this task. A reply of `0.87` means the
model was willing to write 0.87; it is not a frequency, and nothing here has
been fitted to outcomes. **Compare the probabilities across the candidates of
one question, not across states or across models.** A distribution of
`0.71 / 0.12 / 0.17` says the first option was preferred by roughly that much
on this state, and that is all it says.

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
    "model": "Qwen/Qwen3.8-Flash"
  }' | jq
```

```json
{
  "model": "Qwen/Qwen3.8-Flash",
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
| `model` | no | Falls back to `PERCEPTEA_MODEL`. |
| `temperature` | no | Falls back to `PERCEPTEA_TEMPERATURE`. An explicit `0` is honoured. |
| `mode` | no | `parallel` (default) or `oneshot`. |
| `api_key` | no | Only when `PERCEPTEA_ALLOW_REQUEST_CREDENTIALS` is on; otherwise ignored silently. |
| `inference_base_url` | no | Same, and it means the same as the variable of that name: the API root this request's scoring calls go to. An unusable one is a `400`. |

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
{"ok":true,"service":"perceptea","inference_base_url":"https://api.deepinfra.com/v1/openai","model":"Qwen/Qwen3.8-Flash","api_key_configured":true}
```

It reports what this process would call: the endpoint, the model, and whether
the *server* holds a key of its own — never the key itself. Neither the key
nor a credential embedded in a base URL is ever in a response, a log line or
an error message: `inference_base_url` is rendered with its userinfo and query
stripped, and both keys — the server's and one sent in a request body — are
scrubbed out of every message on its way to a caller or a log.

### Errors

Every failure but one answers with the same body, so a client can branch on
`code` and never on prose:

```json
{"error":"missing \"state\"","code":"invalid_request"}
```

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_json` | The body is not valid JSON, a question is malformed, a field is not one this endpoint takes, or something follows the JSON object. |
| 400 | `invalid_request` | No `state`, no `questions`, a question the classifier rejects, or an `inference_base_url` that could never be called. |
| 400 | `unsupported_mode` | A mode that is not `parallel` or `oneshot`. (The provider shipped here can do both, so the "this provider cannot" half needs a backend of your own.) |
| 401 | `missing_api_key` | No key could be resolved for the request. |
| 404 | `not_found` | No such endpoint. |
| 405 | `method_not_allowed` | Wrong method for the route. The `Allow` header names the right one. |
| 413 | `payload_too_large` | The body is over `PERCEPTEA_MAX_BODY_BYTES`. |
| 429 | `upstream_rate_limited` | The provider rate limited us. Reported even when the wait for it ran into the request deadline. |
| 499 | — | The caller hung up, or the body stopped arriving. No body is sent. |
| 500 | `internal` | A bug, a panic included. The detail goes to the log, not to the caller. |
| 502 | `upstream_error` | The provider failed, could not be reached, or cut its reply off at the output token limit before saying anything readable — see [Choosing a model](#choosing-a-model). The message says which. |
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
	"github.com/titusai-io/perceptea/provider/inference"
)

func main() {
	client, err := inference.New(inference.Config{
		APIKey:  os.Getenv("PERCEPTEA_API_KEY"),
		BaseURL: "https://api.deepinfra.com/v1/openai",
		Model:   "Qwen/Qwen3.8-Flash",
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
			Model:     "Qwen/Qwen3.8-Flash",
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
  proxy.** With it on, any caller can supply `inference_base_url` and
  `api_key` and have the service make the call for them — to any host it can
  reach, including ones only it can reach. That is exactly what you want on a
  laptop, where pointing at a local model server is the point, and exactly
  what you do not want anywhere else: set it to `false` outside development.
  It is a boolean and there is no middle setting — a caller either names the
  endpoint or they do not. The service has no authentication of its own — put
  it behind something that does.
- **A purpose-built classifier model would be faster and cheaper.** This is
  ordinary chat models used as micro-scorers; its latency and calibration are
  those of the model you point it at, and one call per candidate is a worse
  deal than a model that scores the whole answer space in a single pass.
- **A reasoning model cannot be used for `parallel` mode** unless its
  thinking can be turned off, because a scoring reply is capped at 32 tokens
  and thinking tokens exhaust it before the answer. The request fails with a
  502 that says so, which is the best available outcome, not a good one. See
  [Choosing a model](#choosing-a-model).
- **`oneshot` is a comparison baseline**, not a supported path. Use `parallel`.

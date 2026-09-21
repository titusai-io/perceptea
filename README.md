# Perceptea

A Go REST classifier API: post a **state** and a set of typed **questions**, get
typed **answers** with probability distributions.

It answers with typed decisions a program can branch on, rather than prose:
the answer space is declared up front and the model only ever scores
candidates from it.

The whole thing is standard library only — no third-party modules, in the
service or in its tests.

```
POST /api/evaluate        classify a state against declared questions
POST /api/evaluate/batch  classify many states against one question set
GET  /api/health          liveness, plus the endpoint and model in use
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
2. **Independent probabilities, of a candidate that knows the field.** Each
   `p` comes from its own call, against the same state, and nothing one call
   returns can move another — that independence is what keeps a weak candidate
   from dragging a strong one, and what makes the numbers comparable rather
   than a single sampled guess. Each call is *shown* the question's other
   candidates, because a call asked "is `d` correct?" with no idea that `a`,
   `b` and `c` exist cannot compare, and comparing is most of what such a
   question asks. Showing them is measured, not assumed — see
   [The candidate list](#the-candidate-list) — and it is free: the list is the
   same for every candidate of the question, so it rides in the cached prefix.
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
time=2026-09-20T21:12:58.122-04:00 level=INFO msg="perceptea listening" addr=[::]:5301
  inference_base_url=https://api.deepinfra.com/v1/openai
  model=mistralai/Mistral-Small-24B-Instruct-2501
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

## Running it in a container

```bash
cp .env.example .env          # then put PERCEPTEA_API_KEY in it
docker compose up --build
curl localhost:5301/api/health
```

Compose reads `.env` itself and passes the settings in as environment
variables, so the image carries no configuration and no `.env` of its own —
`.dockerignore` keeps both out of the build context. `PERCEPTEA_API_KEY` is
the one variable with no default: leave it unset and compose refuses to start
with a message saying so, rather than bringing up a service that 401s on
every request.

The image is a static binary on `distroless/static`, about 9 MB, running as
`nonroot` with no shell, no package manager and a read-only root filesystem.
Since there is no shell there is also no `curl`, so the healthcheck is the
binary probing itself:

```bash
perceptea -healthcheck      # exits 0 when /api/health answers 200
```

The build runs `go vet` and the tests inside the builder stage, so an image
that exists is an image whose tests passed. `docker build --build-arg
VERSION=v1.2.3` stamps the version into the startup log line.

### A fully local setup

To run against a model on your own machine, with nothing leaving it:

```bash
docker compose --profile local-model up --build
docker compose --profile local-model exec ollama ollama pull qwen2.5:7b
```

Then in `.env`:

```bash
PERCEPTEA_INFERENCE_BASE_URL=http://ollama:11434/v1
PERCEPTEA_MODEL=qwen2.5:7b
PERCEPTEA_API_KEY=ollama      # a token is required; any value will do
```

Pick one that does not reason — see [Choosing a model](#choosing-a-model).

## Configuration

Everything is an environment variable; `-addr` is the one flag. A `.env` file
in the working directory is read at startup, and never overrides a variable
that is already set. A **missing API key is not a startup error** — the server
comes up and rejects the requests that would need one with a 401.

| Variable | Default | What it does |
|---|---|---|
| `PERCEPTEA_ADDR` | `:5301` | Listen address. `-addr` overrides it. |
| `PERCEPTEA_INFERENCE_BASE_URL` | `https://api.deepinfra.com/v1/openai` | The API root of the service that runs the model — see below. Must be an absolute `http` or `https` URL. A credential in it — userinfo, or a `?key=` — never reaches a response or a log line. |
| `PERCEPTEA_API_KEY` | — | The key sent to that endpoint as a bearer token. The only variable a key is read from. |
| `PERCEPTEA_MODEL` | `mistralai/Mistral-Small-24B-Instruct-2501` | The model id to score with. It has to be one the endpoint above serves, and it should be one that does not reason — see [Choosing a model](#choosing-a-model). |
| `PERCEPTEA_REASONING_EFFORT` | — | Sent to the provider as `reasoning_effort`: `none`, `low`, `medium` or `high`. Unset sends no reasoning field at all. See [Choosing a model](#choosing-a-model). |
| `PERCEPTEA_SCORER` | `chat` | How a probability is obtained: `chat` asks the model to write one, `logprob` reads it out of the first token's distribution. See [Two scorers](#two-scorers). |
| `PERCEPTEA_TEMPERATURE` | `0` | Sampling temperature. 0 is the only reproducible setting. |
| `PERCEPTEA_MAX_CONCURRENCY` | `8` | Scoring calls in flight per request. For `/api/evaluate` that is per evaluation; for `/api/evaluate/batch` it is the budget for the whole job, shared across every item, which is the point of that endpoint. |
| `PERCEPTEA_MAX_BATCH_ITEMS` | `100` | The most states one `/api/evaluate/batch` request may carry. A batch fans out into items × candidates calls, so this is what stops a single request committing to an unbounded amount of provider spend. Over it is a `400`. |
| `PERCEPTEA_REQUEST_TIMEOUT` | `60s` | Deadline for one request, end to end. A batch is one request, so a large one needs a larger deadline. |
| `PERCEPTEA_MAX_BODY_BYTES` | `2097152` | Request body limit, 2 MiB, shared by both endpoints. A single state large enough to exceed it will not fit a model's context either. It is the batch endpoint that can reach it honestly — see [`POST /api/evaluate/batch`](#post-apievaluatebatch) for the arithmetic. |
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
| DeepInfra | `https://api.deepinfra.com/v1/openai` | `mistralai/Mistral-Small-24B-Instruct-2501` |
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
9 calls × (instruction + state + one statement)   ← the bill
9 × {"p": 0.87}                                   ← the output
```

So **input tokens dominate and output is about ten tokens per call.** A
provider's output price, the number most model comparisons lead with, is
nearly irrelevant here. Input price and latency are what you are buying, and
latency matters twice over because the calls fan out: the request takes as
long as the slowest wave of `PERCEPTEA_MAX_CONCURRENCY` calls, not as long as
one.

### The prompt is cut so its prefix can be cached

Look at that bill again. Only the last few words of each call differ — the
state is the same state nine times over. So the prompt is sent as two
messages, cut exactly where the repetition stops:

```
message 1  the estimator instruction, the question's examples, the question's
           candidate list, the STATE
           ← identical for every candidate of the question, byte for byte
message 2  the one STATEMENT being judged, and the question to answer
           ← a line or two, and the only part that changes
```

Most OpenAI-compatible endpoints cache a prompt **prefix** automatically and
bill a matching one at a fraction of the input price. Nothing has to be asked
for and no extra field is sent — a match is a match. What it needs is that the
repeated part comes first and is exactly the same bytes, which is why nothing
that varies by candidate is allowed into the first message.

For the 9-candidate request that is 3 distinct prefixes instead of 9 whole
prompts: the state is charged at full price once per question and at the
cached rate for every candidate after the first. The saving grows with the
state, which is the direction real states grow in — and with a long state and
a wide choice, the prefix is nearly the whole bill.

It is not free everywhere. An endpoint that does not cache simply sends the
same tokens it always did, so the layout costs nothing where it gains nothing.

### The candidate list

Message 1 carries one more thing, for any question with more than one
candidate: all of them.

```
The candidates for this question, exactly one of which is correct:
- The correct which team should handle this is "billing" (Charges, refunds, invoices).
- The correct which team should handle this is "technical" (Bugs).
- The correct which team should handle this is "sales" (Pricing and plans).
```

Without it, the call judging `sales` has never been told that `billing` and
`technical` were on the table. A question whose entire content is *which of
these* is then being put to a model that cannot see *these*.

It changes the numbers. Measured on a held-out set of 764 questions:

| Model | accuracy | ECE | Brier |
|---|---|---|---|
| `Mistral-Small-24B`, without the list | 0.654 | 0.170 | 0.486 |
| `Mistral-Small-24B`, with it | **0.723** | **0.146** | **0.420** |
| `Llama-3.3-70B`, without the list | 0.711 | **0.149** | 0.422 |
| `Llama-3.3-70B`, with it | **0.731** | 0.166 | **0.410** |

Accuracy and Brier score improve on both models. Expected calibration error
improves on one and worsens on the other by about as much, so the list earns
credit for more answers being right and for better probabilities behind them
— not for confidence that tracks the outcome more closely.

The reason to believe it is where the movement lands, not how big it is. On
the 70B model the two widest-option sources move (0.431 → 0.500 and 0.526 →
0.574) and the binary and few-option sources do not move at all. An effect
that appears exactly where comparison is the work, and nowhere it is not, is
an effect; the same shift spread evenly over everything would have been a
run.

And it is free, which is the other half of why it is there. The list is the
same bytes for every candidate of its question — that is what qualifies it
for message 1 — so it is no extra call, and after the first candidate it is
no extra uncached token either. A question with one candidate, a noul or a
one-option choice, renders no list at all: there is nothing to choose among,
and its prompt is byte for byte the prompt it was before.

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
| **`mistralai/Mistral-Small-24B-Instruct-2501`** | **0.050** | **0.080** | **The default.** Non-reasoning, structured output, built for low latency. |
| `mistralai/Mistral-Nemo-Instruct-2407` | 0.019 | 0.030 | Non-reasoning, structured output, tools. The cheapest of the set; a small model, so a fine swap when the questions are easy. |
| `Qwen/Qwen3.8-Flash` | 0.113 | 0.382 | Non-reasoning, structured output. Its endpoint rejects a strict `json_schema`, so it runs a level down — see the note below the table. |
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

**A `structured-output` tag is not a promise the strict schema will be
accepted.** It says the model can be constrained; the endpoint in front of it
still has to accept the exact `response_format` sent, and some reject a strict
`json_schema` while accepting `json_object` — sometimes reporting it as a 405,
sometimes as a 500 wrapping their own upstream's 400. The service reads the
message rather than the status, steps down, and remembers the level that
answered, so this costs one extra call per client and then nothing. It is
worth knowing it happened, because the strict schema is the one setting that
makes a non-probability reply impossible: look for `structured output
downgraded` at debug level.

### What it costs

Take a 300-token state and the 9-candidate request above. Each call carries
the instruction, the state and one statement — call it 300 tokens — so the
request is roughly **2,700 input tokens and 90 output tokens**.

At the default model's prices that is 2,700 × $0.050/M + 90 × $0.080/M ≈
**$0.00014**, about a seventh of a tenth of a cent. A thousand requests is
fourteen cents. Even the most expensive row above stays under a tenth of a
cent per request. That is the price with nothing cached; on an endpoint that
does cache a matching prefix, most of those 2,700 input tokens are repeats of
three prefixes and are billed at the cached rate.

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

### Two scorers

`PERCEPTEA_SCORER` picks how a probability is obtained. `chat`, the default,
asks the model to write one and parses `{"p": 0.87}` out of the reply.
`logprob` asks the same statement as a one-word Yes or No, requests
`logprobs`, and computes the answer from the first token's distribution:

```
P = exp(l_yes) / (exp(l_yes) + exp(l_no))
```

**Why you would switch.** A written probability is quantised by the model's
writing habits: models emit 0.8, 0.9 and 0.95 and almost never 0.87, so the
shape of a distribution is partly the shape of that habit. A logprob is
continuous, and it is the estimate itself rather than a description of one.
It also decodes one token instead of ten, which is most of the latency on a
reply this short.

**What it needs.** An endpoint *and* a model that accept `logprobs` and
`top_logprobs` on a chat completion and return the top-k distribution. Not
every endpoint does, some cap `top_logprobs` below the 20 asked for, and a
gateway may strip the fields. Check yours before flipping the switch.

**How it fails: loudly.** No logprobs is a `502 upstream_error` on
`/api/evaluate`, naming the setting and the model — and on
`/api/evaluate/batch` it is the same message on the failed item, with
`error_code` `upstream_error`, because a batch is served whatever became of
its items. It does not fall back to the chat scorer and it does not score a
neutral 0.5 — for the same reason a truncated reply does not. A model that
answers a Yes-or-No question with neither word fails the same way, and the
error prints the tokens it offered instead. Both faults repeat on every
candidate of every request, so a neutral score would turn the whole answer
into a confident-looking uniform distribution.

Two warnings. The scorers are **different estimators of the same quantity**,
so thresholds and calibration do not transfer — re-run the benchmark when you
switch. And a reasoning model is worse here than for the chat scorer: its one
token is a thinking token, so pair `logprob` with
`PERCEPTEA_REASONING_EFFORT=none` or a model that does not reason.

## API

### `POST /api/evaluate`

```bash
curl -s localhost:5301/api/evaluate \
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
    "model": "mistralai/Mistral-Small-24B-Instruct-2501"
  }' | jq
```

```json
{
  "model": "mistralai/Mistral-Small-24B-Instruct-2501",
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

### `POST /api/evaluate/batch`

Many states, one question set, one job.

Classifying a hundred states over a hundred requests means a hundred
independent fan-outs, each entitled to `PERCEPTEA_MAX_CONCURRENCY` calls of
its own, all competing for the same provider. A batch is one fan-out:
**every candidate of every item shares one concurrency budget**, so the load
the provider sees is the limit you configured rather than the limit times the
number of callers.

```bash
curl -s localhost:5301/api/evaluate/batch \
  -H 'content-type: application/json' \
  -d '{
    "items": [
      {"id": "t-1", "state": "Charged twice again!! Second month in a row."},
      {"id": "t-2", "state": "Thanks, all sorted now."},
      {"id": "t-3", "state": "The export button does nothing."}
    ],
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
      "angry": {"type": "noul", "instructions": "Strong frustration or anger?"}
    },
    "mode": "parallel"
  }'
```

```json
{
  "model": "mistralai/Mistral-Small-24B-Instruct-2501",
  "results": [
    {
      "id": "t-1",
      "index": 0,
      "answers": {
        "department": {"type": "choice", "choice": "billing", "confidence": 0.82,
                       "probabilities": {"billing": 0.71, "technical": 0.12, "other": 0.17}},
        "angry": {"type": "noul", "noul": 0.88}
      },
      "usage": {"input_tokens": 1180, "output_tokens": 32}
    },
    {
      "id": "t-2",
      "index": 1,
      "answers": {
        "department": {"type": "choice", "choice": "other", "confidence": 0.44,
                       "probabilities": {"billing": 0.21, "technical": 0.24, "other": 0.55}},
        "angry": {"type": "noul", "noul": 0.03}
      },
      "usage": {"input_tokens": 1104, "output_tokens": 32}
    },
    {
      "id": "t-3",
      "index": 2,
      "answers": {},
      "usage": {"input_tokens": 552, "output_tokens": 16},
      "error": "upstream provider error (status 429): rate limit exceeded",
      "error_code": "upstream_rate_limited"
    }
  ],
  "usage": {"input_tokens": 2836, "output_tokens": 80},
  "meta": {"mode": "parallel", "latency_ms": 2140, "parallel_calls": 12,
           "items": 3, "succeeded": 2, "failed": 1}
}
```

| Field | Required | What it is |
|---|---|---|
| `items` | yes | The states to classify. Each is `{"id": "...", "state": ...}`; `id` is yours and is echoed back, and `state` takes the same shapes it does on `/api/evaluate`. |
| `questions` | yes | One question set, shared by every item and validated once. |
| `model`, `temperature`, `mode`, `api_key`, `inference_base_url` | no | Exactly as on `/api/evaluate`, with the same rules. |

**Results come back in request order**, each carrying its `index` and whatever
`id` you sent, so they can be reassociated either way.

**A failed item does not fail the batch.** The items are independent, so one
state the provider refused gets its own `error` and the rest keep their
answers — the opposite of a single evaluation, where the first error cancels
the whole wave because every candidate there feeds one normalisation. A batch
in which *every* item failed is still a `200`: the request was served, the
items were not, and the results say which. What answers from the error table
instead is anything that invalidates the whole request: no `items`, no
`questions`, a question set that does not validate, an unknown mode, too many
items, a body over the size limit, a key that could not be resolved (`401`),
the evaluation outrunning `PERCEPTEA_REQUEST_TIMEOUT` (`504`), and the caller
hanging up (`499`).

**A failed item carries `error_code` beside its `error`**, from the same
vocabulary the error table uses — `upstream_rate_limited`, `upstream_error`,
`timeout`, `invalid_request`, `unsupported_mode` — so a per-item failure can
be branched on the way a per-request one can, and a rate limit worth backing
off from is tellable from a provider that is simply unwell. The `error`
itself is the message `/api/evaluate` would have put in an error body for the
same fault: a failure whose own text is the diagnosis says so, and one that
never reached the provider says only that, because its text names the
endpoint this server calls and the caller may not be allowed to know it. The
detail goes to the log line instead.

`meta.succeeded` and `meta.failed` let you tell a wholly successful batch from
a partly successful one without walking the results. `usage` totals every
item's, failed ones included: a call that failed late still cost what it cost.

#### How big a batch can be

Two limits apply, and they are different questions.

`PERCEPTEA_MAX_BATCH_ITEMS` (default `100`) is about spend: a batch of *n*
items against a question set with *c* candidates issues *n × c* calls under
one request, so the ceiling is what stops one caller committing the server's
key to an unbounded amount of it. The example above is 3 items × 4 candidates
— three options plus a noul — so 12 were planned and `parallel_calls` reports
12: all of them had been issued by the time the rate limit came back. The
counter increments when a call is *issued*, not when it returns, so the
rate-limited call counts itself, and so does the third item's fourth
candidate, which was already in flight when its sibling failed and was
cancelled there. `usage` counts what came back instead, which is why the
failed item still reports the two candidates that answered before the rate
limit did. Had the rate limit arrived sooner, a candidate still queued behind
the concurrency budget would have been abandoned rather than issued and
`parallel_calls` would be lower.

`PERCEPTEA_MAX_BODY_BYTES` (default 2 MiB) is about the wire, and it is shared
with `/api/evaluate` rather than raised for batches. The arithmetic: the
question set is sent once, so 2 MiB across 100 items leaves roughly **21 KB
per state**. Support tickets, reviews and log lines — what this service is
for — run a few hundred bytes to a couple of kilobytes, so a full batch of
them is 50–200 KB and nowhere near the limit. It binds only on states
averaging over about 21 KB, which is some three and a half thousand words
each; at that size the context window and the fan-out cost are the real
constraints, not the body limit. So the default stands: raise
`PERCEPTEA_MAX_BODY_BYTES` if you batch documents rather than messages, and
leave it alone otherwise.

The two limits answer differently, because they are found at different
points. Too many items is a `400 invalid_request` naming
`PERCEPTEA_MAX_BATCH_ITEMS`, the ceiling and the count you sent: the body was
read and understood, and only then was it too big a job. A body over the byte
limit is a `413 payload_too_large` naming the limit in bytes, because the
read is cut short before there is a document to count items in — the item
ceiling is this endpoint's, but the byte limit is shared with
`/api/evaluate` and answers the same way there.

### `GET /api/health`

```bash
curl -s localhost:5301/api/health
```

```json
{"ok":true,"service":"perceptea","inference_base_url":"https://api.deepinfra.com/v1/openai","model":"mistralai/Mistral-Small-24B-Instruct-2501","api_key_configured":true}
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
| 400 | `invalid_request` | No `state`, no `questions`, no `items`, more items than `PERCEPTEA_MAX_BATCH_ITEMS`, a question the classifier rejects, or an `inference_base_url` that could never be called. |
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

The table is for failures of the *request*. On `/api/evaluate/batch` an item
that failed is not one of those: it answers `200` with the reason in that
item's `error` and the code in its `error_code`. Both come from this same
table — an item is told what a request would have been told, minus the
status, which a served request has no second one of. The message is
scrubbed of credentials like any other on its way out, and a failure this
server cannot vouch for is reported as the same opaque upstream error a
`502` carries, with its text kept for the log.

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

### `examples` — teach a question by demonstration

Any question may carry `examples`: worked answers, each a state and the answer
that was correct for it. They are optional, and a question that declares none
produces exactly the prompt it would have produced without the field.

```json
{ "type": "choice",
  "instructions": "Which team should handle this?",
  "criteria": { "billing": "Charges, refunds, invoices", "technical": "Bugs" },
  "examples": [
    { "state": "I was charged twice for the Pro plan.", "answer": "billing" },
    { "state": "The dashboard 500s on every load.",     "answer": "technical" }
  ] }
```

`answer` is written in the question type's own terms, and is checked against
the criteria before any call is made:

| Type | `answer` | Rejected when |
|---|---|---|
| `choice` | an option key, as a string — `"billing"` | it is not one of the declared keys |
| `score` | a level index, as a whole number — `2` | it is outside the declared scale |
| `noul` / `boolean` | `true` or `false` | — (only the rule below) |

**Every type rejects an example with no `answer`**, and `"answer": null`
counts as none: `example 0 is missing "answer"`. The two are one omission and
get one message. It is checked before the type-specific rule above rather
than left to it, because only `choice` would have caught it — `null` decodes
into a number and into a boolean without complaint and leaves the zero value,
so a `score` example would teach level `0` and a `noul` example would teach
`false`, neither of which the request said. `perceptea-bench` applies the
same rule to a dataset line, where a fabricated label would be scored against
rather than shown to the model.

`state` takes the same shapes a request's own state does — a string, an object
or an array — and is required on every example.

Use them where the criteria alone leave a judgement call open: a borderline
case, a label whose name reads more broadly than you mean it, two options a
model keeps confusing. Two or three do most of the work.

They are **shown once per question, not once per candidate**: they render into
the part of the prompt that every candidate shares, so on an endpoint that
caches a matching prefix they are paid for in full once and at the cached rate
thereafter. They are still tokens, so keep each example's state short — an
example is not the place for a full transcript.

Examples belong to the parallel sampler, which is the default. `"mode":
"oneshot"` builds one document from the instructions alone and does not show
them.

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
		Model:   "mistralai/Mistral-Small-24B-Instruct-2501",
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
			Model:     "mistralai/Mistral-Small-24B-Instruct-2501",
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
`go vet`, `go build` and `go test -race` on every pull request, including
from a fork, and on every push to `main`.

## Measuring whether the probabilities mean anything

Everything above is a claim about accuracy. `perceptea-bench` is how you check
one:

```bash
go run ./cmd/perceptea-bench -dataset cmd/perceptea-bench/example.jsonl
```

It reads a labelled dataset, runs it through the classifier against your
configured endpoint, and reports **Brier score**, **expected calibration
error** over ten bins with the per-bin reliability table, **accuracy** for
choice questions, **mean absolute error** for score questions, and
**coverage** — how many cases ran and how many failed, because a benchmark
that quietly evaluated 40 of 100 and reported a lovely number is worse than
none.

The dataset is JSON Lines, one labelled case per line, with the question in
exactly the shape the API takes so a case can be lifted out of a request body:

```json
{"id":"refund-01","name":"intent","state":"I want my money back.",
 "question":{"type":"choice","instructions":"Which intent is this?",
             "criteria":{"refund":"Wants money back","support":"Wants help"}},
 "answer":"refund"}
```

`answer` follows the question's type, the same way a worked example does: an
option key, a level index, or true and false. A line that will not parse names
its line number rather than being skipped.

`-json` emits the same numbers as a document, with no timestamp and every
statistic rounded, so two runs diff cleanly — which is the point. Reach for
this before and after changing a model, a scorer or a prompt; without it, "the
new one is better" is an opinion.

**Interrupting a run does not throw it away.** Ctrl-C stops the dispatch,
prints the report for the attempts that did finish, and exits non-zero saying
how many there were: a hundred cases stopped at sixty measured sixty cases,
and those cost real calls. The non-zero exit is what stops a script filing a
partial document as a complete one.

It needs a real key and makes real calls, so it is a tool rather than a test.
`cmd/perceptea-bench/example.jsonl` is a runnable seven-case dataset covering
all three question types. Everything else it needs comes from the same
environment the server reads — the endpoint, the key, the model, the
temperature, `PERCEPTEA_SCORER`, `PERCEPTEA_MAX_CONCURRENCY` and
`PERCEPTEA_REQUEST_TIMEOUT` — so a benchmark measures the configuration you
are actually serving. A credential carried in `PERCEPTEA_INFERENCE_BASE_URL`
is struck out of any failure the report records, because the `-json` document
is one you are being told to store and diff.

## Contributing

Pull requests are welcome. [`CONTRIBUTING.md`](CONTRIBUTING.md) covers the
development setup, the four checks CI runs, and the testing rules — the one
worth knowing in advance is that **every commit must be signed off**
(`git commit -s`), which a check on the pull request enforces.

Notable changes are recorded in [`CHANGELOG.md`](CHANGELOG.md), and
contributors in [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

## License

Perceptea is licensed under the [GNU Affero General Public License v3.0](LICENSE).

The AGPL's network clause is the part that matters for a service like this
one: running a modified Perceptea and letting other people reach it over a
network counts as distribution, so they are entitled to the source of your
modified version. Using it unmodified, or modifying it for yourself and
nobody else, carries no such obligation.

## Limitations

- **Nothing here is fitted to outcomes, so measure before you trust a
  threshold.** The probabilities come from the model you point at, normalised
  across a question's candidates. Whether they are well calibrated is a
  property of that model on your data, and it is a question with an answer:
  `perceptea-bench` reports the Brier score and the calibration error on a
  labelled set of your own. Run it before you pick the number you branch on.
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
- **A reasoning model cannot be used for `parallel` mode** unless its
  thinking can be turned off, because a scoring reply is capped at 32 tokens
  and thinking tokens exhaust it before the answer. The request fails with a
  502 that says so, which is the best available outcome, not a good one. See
  [Choosing a model](#choosing-a-model).
- **`oneshot` is a comparison baseline**, not a supported path. Use `parallel`.

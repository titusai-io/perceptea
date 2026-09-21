# Changelog

All notable changes to Perceptea are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and Perceptea adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Perceptea is pre-1.0: breaking changes may land in minor releases, and are
always called out below.

## [Unreleased]

### Added

- **`PERCEPTEA_SOFTMAX_TEMPERATURE`, the divisor a question's candidate
  logits are softmaxed at.** It is not `PERCEPTEA_TEMPERATURE`, which is the
  sampling temperature the provider is sent; this one is never sent anywhere
  and only decides how sharply the probabilities that came back are
  normalised into a distribution. It **cannot change an answer**: dividing
  every logit of a question by the same positive number is monotone, so no
  argmax and no ranking can move, at any temperature. What moves is
  calibration. Fitted by minimising the negative log likelihood of the true
  label over a held-out set of 764 questions on
  `mistralai/Mistral-Small-24B-Instruct-2501`, the best temperature is 5.04,
  the same on either half of a random split; cross-fitted, expected
  calibration error falls from 0.158 to 0.103 and Brier from 0.430 to 0.417
  with accuracy unchanged at 0.712, as it has to be. The **default is 1**,
  the identity, because that value is fitted to one model on one task and no
  existing deployment's numbers should move without being asked to; the
  README's [The softmax temperature](README.md#the-softmax-temperature) is
  the procedure for fitting your own. `0`, a negative, a NaN or a word stops
  the process at startup naming the variable, rather than being floored to
  0.05 by the transform without a word. `classifier.WithSoftmaxTemperature`
  is the library form, `cmd/perceptea-bench` honours it so a benchmark
  measures the configuration being served, and the startup line reports it.

- **Every scoring call is shown its question's candidate list.** The call
  judging one option used to be given no hint that the others existed, which
  is most of what a "which of these?" question is. The whole list now renders
  once per question — never once per candidate — into the shared half of the
  prompt, between the worked examples and the state, so it is no extra call
  and, on an endpoint that caches a matching prefix, no extra uncached token
  after the first candidate. Measured on a held-out set of 764 questions:
  accuracy 0.654 → 0.723 and Brier 0.486 → 0.420 on `Mistral-Small-24B`,
  0.711 → 0.731 and 0.422 → 0.410 on `Llama-3.3-70B`, with the gain landing
  on the multi-option questions and the binary ones unmoved; expected
  calibration error improved on the first model and worsened on the second,
  so the claim is accuracy and Brier and not calibration. A question with one
  candidate — a noul, a one-option choice — has nothing to choose among,
  renders no list, and produces the prompt it produced before, byte for byte.
  `classifier.ScoreRequest` gains a `Candidates` field beside `Examples`,
  carrying the list already rendered for the same reason `Examples` is, and
  `classifier.CandidatesBlock` is what renders it. Both scorers place it, and
  they share the code that does, so the two layouts cannot drift apart.
- **`cmd/perceptea-bench`, a calibration benchmark.** Reads a labelled JSON
  Lines dataset and reports Brier score, expected calibration error over ten
  bins with the reliability table, accuracy, mean absolute error, and how many
  cases actually ran. It exists because everything else here is a claim about
  accuracy: without it, "this model is better calibrated" or "the new prompt
  is no worse" cannot be checked. `-json` output carries no timestamp and is
  rounded, so two runs diff cleanly. Interrupting a run prints the report for
  the attempts that finished and exits non-zero saying how many there were,
  rather than discarding minutes of paid-for calls; a credential carried in
  `PERCEPTEA_INFERENCE_BASE_URL` is struck out of every failure the report
  records, because that document is one you are told to store and diff.
- **`PERCEPTEA_SCORER=chat|logprob`.** The default `chat` asks the model to
  write a probability. `logprob` asks a one-word Yes or No and computes the
  answer from the first token's distribution, which is continuous where a
  written number clusters on 0.8, 0.9 and 0.95, and decodes one token instead
  of ten. It needs an endpoint and a model that return logprobs, and when they
  do not it is a 502 naming the setting rather than a neutral score — the two
  scorers are different estimators, so thresholds do not transfer between
  them.

- **Licensed under the GNU Affero General Public License v3.0.**

- **The `classifier` package**: a reusable, standard-library-only classifier.
  Declared questions are answered by scoring each candidate independently and
  normalising the results through a logit transform and a softmax. Three
  question types: `choice` (1–255 options, argmax with the full distribution),
  `score` (2–10 levels, the expected value over their indices, so a rating can
  land between two declared levels), and `noul` (a single proposition, whose
  answer is the probability itself). Its one outward dependency is the
  `Scorer` interface, so the mathematics is testable with no network and a
  different backend can be substituted without touching it.
- **Declaration order is part of the contract.** The order a request declares
  a choice's options in fixes the order statements are built and scored in,
  and decides an exact tie in favour of the earliest. Criteria, questions,
  answers, legends and probability maps all decode into an order-preserving
  map. The state renders into the prompt from its raw request bytes, so a
  caller's key order is never silently sorted.
- **The `provider/inference` package**: a client for any OpenAI-compatible
  `/chat/completions` endpoint. It negotiates structured output down a ladder
  — strict JSON schema, JSON object, then plain text — remembers the level
  that answered rather than the level that failed, and retries 408, 429, 5xx
  and transport failures with backoff that honours `Retry-After`.
- **`POST /api/evaluate`**, with `GET /api/health` alongside it. Per-request
  deadlines and body limits, bounded concurrency, panic recovery, graceful
  shutdown, and one structured log line per request that carries neither the
  caller's data nor any credential.
- **`PERCEPTEA_REASONING_EFFORT`** sends `reasoning_effort` on every call when
  set, and nothing at all when unset.
- **A container image and a compose environment.** A static binary on
  `distroless/static`, roughly 9 MB, running as `nonroot` with a read-only
  root filesystem. Since the image has no shell, `perceptea -healthcheck`
  probes the running instance from inside it.
- **The scoring prompt is laid out so its prefix can be cached.** A scoring
  call is sent as two messages: the estimator instruction, the question's
  worked examples, its candidate list and the state in the first, and the one
  statement being judged in the second. The first is byte-identical for every
  candidate of a question — a choice of 4, a score of 4 and a noul are 9
  calls and 3 prefixes — so an endpoint that caches a matching prompt prefix
  bills the repeats at the cached rate. Nothing that varies by candidate may
  appear in it, and a test asserts the bytes rather than the intent, because
  a stray index or count would defeat the caching with no symptom except the
  bill. No cache-control field is sent: the match is what does it, and a
  field an endpoint has never heard of is one more thing for it to reject.
- **`examples`: optional worked answers on any question.** Each is a state
  and the answer that was correct for it, written in the question type's own
  terms — an option key, a level index, or `true`/`false`. They render into
  the shared prefix as one labelled block, so they are shown once per
  question rather than once per candidate. A question that declares none
  produces exactly the prompt it produced before the field existed.
- **Examples are validated before any call is made.** A choice example must
  name a declared option key, a score example a level inside the declared
  scale, and every example needs a state; each is a `400` naming the question
  and the `examples` field. An example is read on every call of the wave, so
  a wrong one is wrong many times over.
- **`POST /api/evaluate/batch`**: many states, one shared question set, one
  fan-out. The point is the concurrency budget: `PERCEPTEA_MAX_CONCURRENCY`
  bounds every candidate of every item together rather than each item
  separately, so classifying a hundred states no longer means a hundred
  independent fan-outs competing for the same provider. The shared questions
  are validated once for the whole batch rather than once per state.
- **A failed item does not fail the batch.** Items are independent, so each
  result carries either its answers or its own `error`, and the rest keep
  theirs — the opposite of a single evaluation, where the first error cancels
  the wave because every candidate feeds one normalisation. A batch in which
  every item failed is still a `200`: the request was served even though the
  items were not, and only a whole-request fault answers from the error
  table. Results come back in request order with their `index` and the
  caller's own `id`, and `meta` counts the items, the successes and the
  failures so the two cases can be told apart without walking the results.
  A per-item failure goes through the same classification a whole-request one
  does: the `error` is the message `/api/evaluate` would have sent for the
  same fault, scrubbed of credentials, and `error_code` beside it carries the
  code from the error table — `upstream_rate_limited`, `upstream_error`,
  `timeout`, `invalid_request`, `unsupported_mode` — so an item can be
  branched on the way a request can. A failure the server cannot vouch for is
  reported as the opaque upstream error rather than quoted, because its text
  names the endpoint this server calls.
- **`PERCEPTEA_MAX_BATCH_ITEMS`** (default `100`) bounds one batch. A batch
  fans out into items × candidates calls, so the ceiling is what stops a
  single request committing to an unbounded amount of provider spend; over it
  is a `400` naming the setting and both numbers. The body limit is
  deliberately not raised alongside it: 2 MiB across 100 items is about 21 KB
  per state, far more than the messages this service classifies, and an
  operator batching documents rather than messages raises
  `PERCEPTEA_MAX_BODY_BYTES` themselves.
- **`Evaluator.EvaluateBatch`** in the `classifier` package, on the same
  machinery as `Evaluate` rather than a second copy of it: both build groups
  of scorer calls and hand them to one wave, so a batch of one is
  indistinguishable from a single evaluation and a test says so.
- **`requests/deepinfra-requests.http`**: runnable examples of every question
  type, the options, worked examples, batches — including a partial failure —
  and every failure mode.

### Changed

- **A reported probability is no longer ever an exact `0` or an exact `1`.**
  Probabilities, `noul` and `confidence` are rounded to six decimal places
  instead of three; a `score` keeps two, because a score is a position on a
  declared scale and not a probability. The per-candidate scores entering the
  softmax are clamped into `(0.001, 0.999)`, so no candidate is ever
  certainly right or certainly wrong, and the most extreme distribution the
  service can reach is 0.999999 against 0.000001 — which three decimals
  published as an exact `1` and `0`, a claim the estimator cannot make. It
  was also a dead end: renormalising or re-tempering a published distribution
  needs `log(p)`. Six is the smallest precision that holds; five still
  reports that extreme as `1`. Finer rounding also tightens the
  sum-to-1 claim the README makes — the published numbers now total 1 to
  within half of the last place per candidate, at most 1.3e-4 for a 255-way
  choice against 0.13 before. Two things are unchanged and deliberate: a
  `noul` may still be `0` or `1`, because it is the model's own probability
  with no softmax between it and the wire, and a probability below 5e-7 still
  rounds to `0` — reachable only where a model scored three or more
  candidates of the same question at 0.999 while scoring another at 0.001,
  and no fixed precision survives that.

### Security

- A credential is never disclosed. The API key, and any userinfo or query
  string in the inference base URL, are stripped from the health payload, the
  startup line, and every error message on its way to a caller or a log.
- A failure about the connection rather than the document — a body that timed
  out, a caller that went away — says nothing on the wire, because the
  underlying error names this server's own socket.
- `PERCEPTEA_ALLOW_REQUEST_CREDENTIALS` is on by default so that a caller can
  point one request at a local model server. That makes the service a proxy to
  any host it can reach, so it is documented as something to turn off outside
  development, and the service has no authentication of its own.

### Notes

- **A reply cut off before it said anything readable is a `502`, not a
  probability.** A scoring call allows 32 output tokens, which a model that
  reasons spends on thinking — and because the cap is the same on every call,
  it would otherwise hit every candidate at once and turn the whole answer
  into a well-formed, confident-looking uniform distribution. The error names
  the cause and the two ways out. A reply that produced its number before the
  cap is still accepted, and one that finished normally without a number still
  counts for a single neutral 0.5.
- **The default model does not reason.** See
  [Choosing a model](README.md#choosing-a-model) for what to weigh when
  changing it: whether it reasons is a constraint, not a preference, and a
  `structured-output` tag is not a promise that the endpoint in front of the
  model accepts a strict schema.

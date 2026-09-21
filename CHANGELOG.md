# Changelog

All notable changes to Perceptea are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and Perceptea adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Perceptea is pre-1.0: breaking changes may land in minor releases, and are
always called out below.

## [Unreleased]

### Added

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
- **`requests/deepinfra-requests.http`**: runnable examples of every question
  type, the options, and every failure mode.

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

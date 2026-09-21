# Contributing to Perceptea

Thanks for your interest. This document covers what you need to know before
opening a pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [The Checks That Have to Pass](#the-checks-that-have-to-pass)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)

## Code of Conduct

This project and everyone participating in it is governed by our community
standards. Please be respectful, constructive, and inclusive in all
interactions.

## How Can I Contribute?

### Reporting Bugs

Open an issue describing what you did, what you expected, and what happened.
For a classification that came back wrong, the useful details are the request
body, the model and endpoint you pointed at, and the response — including
`meta`, which says how many calls were made and in which mode.

Please redact your API key. The service is careful never to log or return
one; an issue report is the one place it can still escape.

### Suggesting Enhancements

Open an issue. Say what you are trying to do and why the current surface makes
it hard — a description of the problem is more useful than a description of
the fix, and usually leads to a better one.

### Pull Requests

1. Fork the repository and branch from `main`.
2. Make your change, with tests.
3. Make sure [the checks](#the-checks-that-have-to-pass) pass locally.
4. [Sign off](#developer-certificate-of-origin-dco) every commit.
5. Open the pull request and describe what changed and how to verify it.

Small, focused pull requests are reviewed faster than large ones. If you are
planning something substantial, open an issue first so the design can be
agreed before you write it.

## Development Setup

### Prerequisites

- Go 1.27 or newer
- An API key for any OpenAI-compatible inference endpoint, to run the service
  against a real model. **You do not need one to work on the code**: no test
  makes a network call.

### Quick Start

```bash
# Clone your fork
git clone https://github.com/<you>/perceptea.git
cd perceptea

# Run the tests — no key, no network
go test ./...

# Run the service
cp .env.example .env     # then put PERCEPTEA_API_KEY in it
go run ./cmd/perceptea

curl localhost:5301/api/health
```

`requests/deepinfra-requests.http` has runnable examples of every question
type and every failure mode. Open it in VS Code with the REST Client
extension, or in a JetBrains IDE, and send them.

### In a container

```bash
docker compose up --build
```

See [Running it in a container](README.md#running-it-in-a-container).

## The Checks That Have to Pass

CI runs these on every push and pull request, and they are the same four you
can run locally:

```bash
gofmt -l .                              # silence means formatted
go vet ./...
go test -race ./...
go mod tidy && git diff --exit-code go.mod go.sum
```

That last one is not a formality. **Perceptea has no third-party
dependencies, in the service or in its tests, and CI enforces it.** A pull
request that adds one will fail, and the discussion to have first is whether
the dependency is worth losing that property for. Most of what a dependency
would give you here is a few dozen lines of standard library.

## Coding Standards

- `gofmt` decides formatting. There is nothing to argue about.
- Exported identifiers have doc comments. Unexported ones have them when the
  reason for the code is not obvious from the code.
- **Comments explain why, not what.** The code already says what it does. A
  comment earns its place by recording the reason a decision was made, the
  failure it prevents, or the surprise it is defending against.
- Errors say what could not be done and name the input that caused it.
  Messages that reach a caller never include a credential, a file path, or
  this server's own address.
- Keep the layering. `classifier` holds the orchestration and the
  mathematics and makes no network calls; everything network-shaped sits
  behind the `Scorer` and `Generator` interfaces. If a change puts an HTTP
  concern into `classifier`, or classification logic into `api`, it is
  probably in the wrong package.

## Testing

**Every behaviour needs a test that fails without the change.** This is the
one rule worth stating at length, because the project has already found three
tests that could not fail:

- One asserted that a markdown fence was stripped, but a fallback path
  recovered the same number either way, so the assertion held with the
  stripping removed.
- One reached its expected value down two independent code paths, so
  deleting either guard left it green.
- One needed two conditions to occur together, and no test produced both.

Before you open a pull request, **break the thing your test guards and watch
it go red**, then restore it and watch it go green. A test that passes
against a broken implementation is worse than no test, because it is a claim
nobody will re-examine.

Two more rules that follow from how this service is built:

- **No test may make a network call.** The HTTP layer builds its evaluator
  through a seam (`api.Options.NewEvaluator`) so tests inject a fake; the
  provider package tests its client against `httptest`. A test that needs a
  key is a test that will not run in CI.
- **Expected numbers are derived, not recorded.** Work an expected
  probability out from the formula and paste it in as a literal. A golden
  captured from the code under test agrees with that code by construction and
  proves nothing.

## Commit Messages

Write a short subject line in the imperative mood, then a body explaining why
the change is being made. The subject says what changed; the body is where
the reasoning goes, and it is the part that is still useful in a year.

```text
Fail loudly when a reply is cut off

A scoring call allows 32 output tokens. A model that thinks before
answering spends that budget on thinking, so the reply ends before the
JSON — and because the cap is the same on every call, it happens to every
candidate at once.
```

Conventional Commit prefixes (`feat:`, `fix:`, `docs:`) are welcome but not
required.

## Developer Certificate of Origin (DCO)

Perceptea uses the [Developer Certificate of Origin
1.1](https://developercertificate.org/) (DCO) to confirm that contributors
have the right to submit the code they are offering.

To sign off your work, pass the `-s` / `--signoff` flag when committing:

```bash
git commit -s -m "Add a thing"
```

This appends a `Signed-off-by:` line to your commit message using your
configured Git name and email address:

```text
Signed-off-by: Jane Contributor <jane@example.com>
```

**Every commit in a pull request must carry this line before it can be
merged**, and a check on the pull request reports whether they all do. If you
forgot to sign off an earlier commit, you can amend it:

```bash
# The most recent commit
git commit --amend --no-edit -s

# Several commits, in an interactive rebase
git rebase -i --signoff main
```

Then force-push your branch.

Signing off is a statement about provenance, not an assignment of copyright.
It says you wrote the change, or otherwise have the right to submit it under
the project's license, and that you are happy for it to be distributed as
part of the project.

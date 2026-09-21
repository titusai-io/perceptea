## What & why

<!-- What does this change, and what problem does it solve? Link any issue. -->

## How to verify

<!-- Steps a reviewer can follow, or the commands you ran. -->

```bash
go test -race ./...
```

## Checklist

<!-- Paths below are plain names on purpose: a relative link in a pull
     request body resolves against the repository root, not against this
     template, so it would arrive broken. -->

- [ ] Every commit is signed off (`git commit -s`) — see the DCO section of `CONTRIBUTING.md`
- [ ] `gofmt -l .` is clean and `go vet ./...` passes
- [ ] `go test -race ./...` passes
- [ ] No third-party dependency was added (`go mod tidy` leaves `go.mod` unchanged)
- [ ] New behaviour has a test, and **I broke the behaviour and watched that test fail**
- [ ] No test makes a network call
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]`

### If this changes the HTTP surface

- [ ] `README.md` is updated — the endpoint list, the configuration table, or the examples
- [ ] `requests/deepinfra-requests.http` still reflects reality
- [ ] Any new failure returns a machine-readable `code`, and it is in the README's error table

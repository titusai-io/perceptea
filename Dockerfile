# Build the binary. CGO is off so the result is static and can run on an
# image with no libc at all.
FROM golang:1.27 AS build

WORKDIR /src

# go.mod on its own first, so the dependency layer is only rebuilt when it
# changes. There are no third-party modules, which is what `go mod download`
# confirms here rather than assumes.
COPY go.mod ./
RUN go mod download

COPY . .

# Run the tests in the image that builds the binary, so a green local run and
# a green image cannot disagree. -race needs cgo, so it stays out of here and
# in CI.
RUN go vet ./... && go test ./...

ARG VERSION=dev
RUN CGO_ENABLED=0 go build \
    -trimpath \
    -ldflags="-s -w -X main.version=${VERSION}" \
    -o /out/perceptea ./cmd/perceptea

# Distroless: no shell, no package manager, nothing to exec into. It carries
# the CA certificates the HTTPS call to the inference endpoint needs, and
# nonroot gives an unprivileged uid without having to create one.
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=build /out/perceptea /perceptea

# The service reads no files: configuration is the environment, and a .env is
# a development convenience deliberately left out of the image.
ENV PERCEPTEA_ADDR=:8080
EXPOSE 8080

# The image has no curl and no shell, so the binary probes itself.
HEALTHCHECK --interval=30s --timeout=5s --start-period=2s --retries=3 \
    CMD ["/perceptea", "-healthcheck"]

USER nonroot:nonroot
ENTRYPOINT ["/perceptea"]

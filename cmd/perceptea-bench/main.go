// Command perceptea-bench measures whether Perceptea's probabilities mean
// anything.
//
// It reads a file of labelled cases, evaluates each one through the same
// classifier path the service uses, and reports the expected calibration
// error with the reliability table behind it, a Brier score per question
// type, accuracy for choice questions, mean absolute error for score
// questions, and how much of the dataset actually ran.
//
// It makes real calls to a real endpoint and needs a real key, so it is a
// tool and not a test. Everything it needs beyond the dataset comes from the
// environment that configures the server itself — the endpoint, the model,
// the key, both temperatures, the scorer, PERCEPTEA_MAX_CONCURRENCY and
// PERCEPTEA_REQUEST_TIMEOUT — so a benchmark run measures the configuration
// you are actually serving. Both temperatures, because there are two and
// they are unrelated: PERCEPTEA_TEMPERATURE is the sampling one the provider
// is told, and PERCEPTEA_SOFTMAX_TEMPERATURE is what the answers are
// normalised through afterwards — the setting this tool exists to fit. A .env file in the working directory is read
// first, as it is for the server.
//
// Interrupting a run prints the report for the attempts that did finish and
// exits non-zero, saying how many of them there were. A run of a hundred
// cases stopped at sixty measured sixty cases, and those are worth having;
// the non-zero exit is what stops a script filing a partial document as a
// complete one.
//
// Usage:
//
//	perceptea-bench -dataset cases.jsonl [-json] [-repeat n] [-limit n]
//
// The dataset is JSON Lines, one labelled case per line. example.jsonl beside
// this file is a runnable one; a line looks like:
//
//	{"id":"refund-01","name":"intent","state":"I want my money back.",
//	 "question":{"type":"choice","instructions":"Which intent is this?",
//	             "criteria":{"refund":"asking for money back","support":"needs help"}},
//	 "answer":"refund"}
//
// The question is exactly the shape a request body declares, so a dataset can
// be lifted straight out of one. The answer follows the question's type: an
// option key for a choice, a level index for a score, true or false for a
// noul. A line that will not parse stops the run and names its line number.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/titusai-io/perceptea/calibration"
	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
	"github.com/titusai-io/perceptea/provider/inference"
)

// dotEnvPath is the optional file read before the environment is examined, as
// the server reads it.
const dotEnvPath = ".env"

// maxRetries is how often a failed provider call is retried. A benchmark run
// makes a great many calls, so a generous number multiplies badly; two clears
// a blip, and anything worse than a blip is a coverage number the report is
// supposed to show rather than hide.
const maxRetries = 2

// attributionURL and attributionTitle identify this tool to providers that
// record them. They duplicate the constants the API server uses, which are
// unexported there.
const (
	attributionURL   = "https://github.com/titusai-io/perceptea"
	attributionTitle = "Perceptea benchmark"
)

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "perceptea-bench: %v\n", err)
		os.Exit(1)
	}
}

// buildEvaluator builds the thing a run evaluates against.
//
// It is a parameter of [runWith] rather than a call inside it so that
// everything around it — the flags, the dataset, the runner's wiring, the
// report, the interrupt and the exit status — can be exercised without a
// provider and without a socket. Only this one function reaches the network,
// and no test supplies the one below.
type buildEvaluator func(cfg config.Config, logger *slog.Logger) (calibration.Evaluator, error)

// run performs one benchmark against the configured provider and writes the
// report to stdout.
func run(args []string, stdout, stderr io.Writer) error {
	return runWith(context.Background(), args, stdout, stderr, liveEvaluator)
}

// liveEvaluator wires the real provider client into the classifier, exactly
// as the server does, so that a benchmark measures the configuration being
// served rather than one of its own.
func liveEvaluator(cfg config.Config, logger *slog.Logger) (calibration.Evaluator, error) {
	client, err := inference.New(inference.Config{
		APIKey:          cfg.APIKey,
		BaseURL:         cfg.BaseURL,
		Model:           cfg.Model,
		ReasoningEffort: cfg.ReasoningEffort,
		Scorer:          cfg.Scorer,
		MaxRetries:      maxRetries,
		Referer:         attributionURL,
		Title:           attributionTitle,
		Logger:          logger,
	})
	if err != nil {
		return nil, err
	}
	return benchEvaluator(client, cfg.MaxConcurrency, cfg.SoftmaxTemperature), nil
}

// benchEvaluator puts the configured ceiling on the scorer, where every call
// passes, and takes the evaluator's own ceiling off.
//
// PERCEPTEA_MAX_CONCURRENCY is a bound on provider calls, and the throttle is
// the only place that sees all of them: an evaluator's limit is per
// evaluation, so leaving it at its default would hold a single case below the
// configured number however high that number was set. With the throttle in
// place it is the only bound, which is the point.
// PERCEPTEA_SOFTMAX_TEMPERATURE is passed straight through, and has to be:
// it is what the reported probabilities are normalised at, so a benchmark
// run without it would measure the calibration of a configuration nobody is
// serving. It is the one setting this tool exists to fit — sweep it, and read
// the negative log likelihood and the calibration error off the report.
func benchEvaluator(scorer classifier.Scorer, maxConcurrency int, softmaxTemperature float64) *classifier.Evaluator {
	return classifier.New(
		calibration.Throttle(scorer, maxConcurrency),
		classifier.WithMaxConcurrency(0),
		classifier.WithSoftmaxTemperature(softmaxTemperature),
	)
}

// runWith performs one benchmark against whatever build produces, stopping
// when parent is done or a signal arrives, whichever comes first.
//
// The parent context is a parameter so that the interrupted path can be
// tested without sending this process a signal: a stray SIGINT in a test
// binary is a killed test run, and the behaviour being checked — a partial
// report, printed — is the same whichever way the cancellation arrived.
func runWith(parent context.Context, args []string, stdout, stderr io.Writer, build buildEvaluator) error {
	flags := flag.NewFlagSet("perceptea-bench", flag.ContinueOnError)
	flags.SetOutput(stderr)
	dataset := flags.String("dataset", "", "path to the labelled JSON Lines dataset (required)")
	asJSON := flags.Bool("json", false, "write the report as a JSON document instead of text")
	repeat := flags.Int("repeat", 1, "evaluate every case this many times, to measure spread at a non-zero temperature")
	limit := flags.Int("limit", 0, "evaluate only the first n cases; 0 means all of them")
	if err := flags.Parse(args); err != nil {
		return err
	}
	// Nothing else is a flag. The endpoint, model, key, temperature and
	// concurrency are already carried by the environment, and a second way to
	// set them is a second thing to get out of step with what the server does.
	if *dataset == "" {
		return errors.New("no dataset: pass -dataset with a path to a JSON Lines file of labelled cases")
	}
	if *repeat < 1 {
		return fmt.Errorf("-repeat must be at least 1, got %d", *repeat)
	}
	if *limit < 0 {
		return fmt.Errorf("-limit must not be negative, got %d", *limit)
	}

	cases, err := calibration.LoadDataset(*dataset)
	if err != nil {
		return err
	}
	cases = truncate(cases, *limit)

	// A .env file is a convenience, not a configuration source of record: a
	// real environment variable always wins, and a missing file is normal.
	if err := config.LoadDotEnv(dotEnvPath); err != nil {
		fmt.Fprintf(stderr, "perceptea-bench: ignoring %s: %v\n", dotEnvPath, err)
	}
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	// Warnings and errors only. The report is the output; a debug record per
	// provider call would bury it.
	logger := slog.New(slog.NewTextHandler(stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))

	evaluator, err := build(cfg, logger)
	if err != nil {
		return err
	}

	runner := newRunner(cfg, evaluator, *repeat)

	// A benchmark is long enough that somebody will interrupt one. Stopping
	// the dispatch on a signal is what turns that into a report of what did
	// run, rather than minutes of real calls thrown away.
	ctx, stop := signal.NotifyContext(parent, syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// An interrupted run returns its outcomes *alongside* the cancellation,
	// and those outcomes are the whole reason they were collected: a hundred
	// cases measured before the Ctrl-C are a hundred measurements, and
	// discarding them prints nothing after minutes of paid-for calls. So the
	// report is written either way and the interruption is reported after
	// it, on stderr, with a non-zero exit — the run did not finish, and a
	// script must not read a partial document as a complete one.
	result, err := runner.Run(ctx, cases)
	if err != nil && !errors.Is(err, context.Canceled) {
		return err
	}
	interrupted := err
	result.Dataset = *dataset

	report := calibration.Summarise(result)
	if err := write(stdout, report, *asJSON); err != nil {
		return err
	}
	if interrupted != nil {
		return fmt.Errorf("interrupted: %d of %d attempts ran; the report above covers those only",
			report.Coverage.Attempted, len(cases)*max(*repeat, 1))
	}
	return emptyRunError(report)
}

// truncate keeps the first limit cases, so that a dataset can be tried
// against a provider without paying for all of it. A limit of zero or one
// past the end keeps every case.
func truncate(cases []calibration.Case, limit int) []calibration.Case {
	if limit <= 0 || limit >= len(cases) {
		return cases
	}
	return cases[:limit]
}

// newRunner wires the run from the same configuration the server reads, so
// that a benchmark measures what is actually being served.
func newRunner(cfg config.Config, evaluator calibration.Evaluator, repeat int) calibration.Runner {
	return calibration.Runner{
		Evaluator:   evaluator,
		Model:       cfg.Model,
		Temperature: cfg.Temperature,
		Repeat:      repeat,
		// Cases in flight, which is not the same ceiling as the one on
		// provider calls: each case fans out on its own, and the throttle
		// inside the evaluator is what bounds the calls.
		Concurrency: cfg.MaxConcurrency,
		// The per-attempt deadline, the same one PERCEPTEA_REQUEST_TIMEOUT
		// puts on a served request. Without it a provider that accepts a
		// connection and then says nothing stops the whole benchmark.
		Timeout: cfg.RequestTimeout,
		Scrub:   scrubber(cfg),
	}
}

// redacted is what a credential is replaced by in a recorded failure.
const redacted = "[redacted]"

// shortestCredential is the length below which a secret is left alone. Local
// model servers conventionally take a throwaway key — "EMPTY", "ollama" — and
// redacting a one or two character secret would strike out every stray letter
// in a message and leave it unreadable. It matches the server's own
// threshold, for the same reasons.
const shortestCredential = 4

// scrubber returns the function a run rewrites its recorded failures with.
//
// The report is a document people store and diff, and a failed provider call
// quotes the URL it was calling. PERCEPTEA_INFERENCE_BASE_URL is routinely a
// gateway carrying a credential — in the userinfo, where net/http masks the
// password and prints the username in full, or in the query, where it masks
// nothing — so config.CredentialsIn says which parts of it are secrets and
// they are struck out of anything the run records. The API key travels in a
// header rather than the URL, and is included anyway: a provider that echoes
// a rejected key into its error message has put it somewhere this cannot see
// coming.
//
// Nil is returned when there is nothing to remove, so the ordinary case
// copies no strings at all.
func scrubber(cfg config.Config) func(string) string {
	var secrets []string
	for _, secret := range append(config.CredentialsIn(cfg.BaseURL), cfg.APIKey) {
		if len(secret) >= shortestCredential {
			secrets = append(secrets, secret)
		}
	}
	if len(secrets) == 0 {
		return nil
	}
	return func(s string) string {
		for _, secret := range secrets {
			s = strings.ReplaceAll(s, secret, redacted)
		}
		return s
	}
}

// emptyRunError reports a run that measured nothing, which is not a result
// however many numbers were printed above it.
func emptyRunError(report calibration.Report) error {
	if report.Coverage.Evaluated > 0 {
		return nil
	}
	return fmt.Errorf("no case produced an answer: all %d attempts failed", report.Coverage.Attempted)
}

// write emits the report in the requested form.
func write(w io.Writer, report calibration.Report, asJSON bool) error {
	if !asJSON {
		_, err := io.WriteString(w, report.Text())
		return err
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

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
// the key, the temperature and PERCEPTEA_MAX_CONCURRENCY — so a benchmark run
// measures the configuration you are actually serving. A .env file in the
// working directory is read first, as it is for the server.
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

// run performs one benchmark and writes the report to stdout.
func run(args []string, stdout, stderr io.Writer) error {
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
	if *limit > 0 && *limit < len(cases) {
		cases = cases[:*limit]
	}

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

	client, err := inference.New(inference.Config{
		APIKey:          cfg.APIKey,
		BaseURL:         cfg.BaseURL,
		Model:           cfg.Model,
		ReasoningEffort: cfg.ReasoningEffort,
		MaxRetries:      maxRetries,
		Referer:         attributionURL,
		Title:           attributionTitle,
		Logger:          logger,
	})
	if err != nil {
		return err
	}

	// PERCEPTEA_MAX_CONCURRENCY bounds provider calls, so it is applied to
	// the scorer, where every call passes. The evaluator's own limit is then
	// switched off: leaving both on would bound each case separately and let
	// the run as a whole exceed the configured number several times over.
	evaluator := classifier.New(
		calibration.Throttle(client, cfg.MaxConcurrency),
		classifier.WithMaxConcurrency(0),
	)

	runner := calibration.Runner{
		Evaluator:   evaluator,
		Model:       cfg.Model,
		Temperature: cfg.Temperature,
		Repeat:      *repeat,
		Concurrency: cfg.MaxConcurrency,
		Timeout:     cfg.RequestTimeout,
	}

	// A benchmark is long enough that somebody will interrupt one. Stopping
	// the dispatch on a signal is what turns that into a clean exit rather
	// than a half-written report.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	result, err := runner.Run(ctx, cases)
	if err != nil {
		return err
	}
	result.Dataset = *dataset

	report := calibration.Summarise(result)
	if err := write(stdout, report, *asJSON); err != nil {
		return err
	}
	// Every number above was computed from nothing, which is not a result.
	if report.Coverage.Evaluated == 0 {
		return fmt.Errorf("no case produced an answer: all %d attempts failed", report.Coverage.Attempted)
	}
	return nil
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

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/titusai-io/perceptea/calibration"
	"github.com/titusai-io/perceptea/classifier"
	"github.com/titusai-io/perceptea/internal/config"
)

// No test here reaches a network. The one function that would —
// liveEvaluator — is a parameter of runWith, so everything around it can be
// driven with a fake: the flags, the dataset, the runner's wiring, the
// report, the interrupt and the exit status. liveEvaluator itself is checked
// only for what it decides before any call is made.

// runErr runs the command with args and returns the error it reported,
// failing the test if it reported none.
func runErr(t *testing.T, args ...string) string {
	t.Helper()
	var stdout, stderr bytes.Buffer
	err := run(args, &stdout, &stderr)
	if err == nil {
		t.Fatalf("run(%v) succeeded; want an error. stdout:\n%s", args, stdout.String())
	}
	return err.Error()
}

func TestRunRequiresADataset(t *testing.T) {
	msg := runErr(t)

	if !strings.Contains(msg, "-dataset") {
		t.Errorf("error = %q, want it to name -dataset", msg)
	}
}

func TestRunRejectsAnImpossibleRepeatOrLimit(t *testing.T) {
	// A dataset that does not exist, deliberately: the flags are checked
	// before anything is opened, so a run that got past them fails with a
	// different message instead of reaching a provider from a unit test.
	absent := filepath.Join(t.TempDir(), "absent.jsonl")
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"repeat below one", []string{"-dataset", absent, "-repeat", "0"}, "-repeat"},
		{"negative limit", []string{"-dataset", absent, "-limit", "-3"}, "-limit"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if msg := runErr(t, c.args...); !strings.Contains(msg, c.want) {
				t.Errorf("error = %q, want it to name %s", msg, c.want)
			}
		})
	}
}

func TestRunNamesADatasetItCannotRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.jsonl")

	msg := runErr(t, "-dataset", path)

	if !strings.Contains(msg, path) {
		t.Errorf("error = %q, want it to name %q", msg, path)
	}
}

// TestExampleDatasetParses keeps the format the command documents and the
// format the loader accepts from drifting apart: the example beside main.go
// is the only thing most people will read before writing their own.
func TestExampleDatasetParses(t *testing.T) {
	cases, err := calibration.LoadDataset("example.jsonl")
	if err != nil {
		t.Fatalf("LoadDataset: %v", err)
	}

	if len(cases) != 7 {
		t.Fatalf("parsed %d cases, want 7", len(cases))
	}
	seen := map[classifier.QuestionType]int{}
	for _, c := range cases {
		seen[c.Question.Type]++
	}
	// All three question types, so the example shows how each is labelled.
	for _, want := range []classifier.QuestionType{classifier.TypeChoice, classifier.TypeScore, classifier.TypeNoul} {
		if seen[want] == 0 {
			t.Errorf("the example dataset has no %q case", want)
		}
	}
}

// ------------------------------------------------ the run, without a provider --

// fakeEvaluator answers every request the same way and counts the cases it
// saw. onCase runs before each answer, so a test can interrupt a run at a
// known point.
type fakeEvaluator struct {
	err    error
	onCase func(n int)

	mu    sync.Mutex
	calls int
}

func (f *fakeEvaluator) Evaluate(ctx context.Context, req classifier.Request) (classifier.Response, error) {
	f.mu.Lock()
	f.calls++
	n := f.calls
	f.mu.Unlock()
	if f.onCase != nil {
		f.onCase(n)
	}
	if err := ctx.Err(); err != nil {
		return classifier.Response{}, err
	}
	if f.err != nil {
		return classifier.Response{}, f.err
	}
	answers := classifier.NewOrderedMap[classifier.Answer]()
	for _, name := range req.Questions.Keys() {
		answers.Set(name, classifier.Answer{Type: classifier.TypeNoul, Noul: 0.75})
	}
	return classifier.Response{Answers: *answers}, nil
}

func (f *fakeEvaluator) seen() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

// datasetFile writes n noul cases and returns the path.
func datasetFile(t *testing.T, n int) string {
	t.Helper()
	lines := make([]string, 0, n)
	for i := range n {
		lines = append(lines, fmt.Sprintf(
			`{"id":"c%d","name":"holds","state":"s%d","question":{"type":"noul","instructions":"It holds."},"answer":true}`,
			i, i))
	}
	path := filepath.Join(t.TempDir(), "cases.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatalf("writing the dataset: %v", err)
	}
	return path
}

// benchEnv pins the configuration a run reads, so that a stray
// PERCEPTEA_ variable in the developer's environment cannot change what these
// tests measure. One case at a time, so an interrupt lands at a known point.
func benchEnv(t *testing.T) {
	t.Helper()
	for name, value := range map[string]string{
		config.EnvAPIKey:           "sk-test-0123456789",
		config.EnvInferenceBaseURL: "https://inference.example/v1",
		config.EnvModel:            "probe-1",
		config.EnvMaxConcurrency:   "1",
		config.EnvRequestTimeout:   "30s",
		config.EnvTemperature:      "0",
		config.EnvScorer:           "chat",
	} {
		t.Setenv(name, value)
	}
}

// TestRunLimitEvaluatesOnlyTheFirstCases is what -limit is for: trying a
// dataset against a provider without paying for all of it.
func TestRunLimitEvaluatesOnlyTheFirstCases(t *testing.T) {
	benchEnv(t)
	const (
		cases = 7
		limit = 3
	)
	fake := &fakeEvaluator{}
	var stdout, stderr bytes.Buffer

	err := runWith(context.Background(), []string{
		"-dataset", datasetFile(t, cases), "-limit", strconv.Itoa(limit), "-json",
	}, &stdout, &stderr, evaluatorFor(fake))
	if err != nil {
		t.Fatalf("runWith: %v (stderr: %s)", err, stderr.String())
	}

	if fake.seen() != limit {
		t.Errorf("the provider saw %d cases, want %d: -limit did not truncate", fake.seen(), limit)
	}
	report := decodeReport(t, stdout.Bytes())
	if report.Coverage.Cases != limit {
		t.Errorf("report covers %d cases, want %d", report.Coverage.Cases, limit)
	}
}

// A limit at or past the end keeps the whole dataset, so -limit 0 — the
// default — is not a run of nothing.
func TestTruncateKeepsEverythingWithoutALimit(t *testing.T) {
	cases := make([]calibration.Case, 5)
	for _, limit := range []int{0, 5, 9} {
		if got := truncate(cases, limit); len(got) != len(cases) {
			t.Errorf("truncate(5 cases, %d) kept %d, want 5", limit, len(got))
		}
	}
	if got := truncate(cases, 2); len(got) != 2 {
		t.Errorf("truncate(5 cases, 2) kept %d, want 2", len(got))
	}
}

// TestRunFailsWhenNothingWasMeasured is the difference between a benchmark
// and a well-formatted page of zeroes.
func TestRunFailsWhenNothingWasMeasured(t *testing.T) {
	benchEnv(t)
	fake := &fakeEvaluator{err: errors.New("upstream said no")}
	var stdout, stderr bytes.Buffer

	err := runWith(context.Background(), []string{
		"-dataset", datasetFile(t, 4), "-json",
	}, &stdout, &stderr, evaluatorFor(fake))

	if err == nil {
		t.Fatalf("runWith succeeded on a run where every case failed. stdout:\n%s", stdout.String())
	}
	for _, want := range []string{"no case produced an answer", "4"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error = %q, want it to mention %q", err, want)
		}
	}
	// The coverage section is the one part of the report that is still worth
	// reading, so it is still printed.
	report := decodeReport(t, stdout.Bytes())
	if report.Coverage.Failed != 4 {
		t.Errorf("report = %d failures, want 4", report.Coverage.Failed)
	}
}

// A run that did measure something exits zero, so the check above cannot be
// passing by rejecting everything.
func TestRunSucceedsWhenSomethingWasMeasured(t *testing.T) {
	benchEnv(t)
	var stdout, stderr bytes.Buffer

	err := runWith(context.Background(), []string{
		"-dataset", datasetFile(t, 2), "-json",
	}, &stdout, &stderr, evaluatorFor(&fakeEvaluator{}))
	if err != nil {
		t.Fatalf("runWith: %v (stderr: %s)", err, stderr.String())
	}
	if report := decodeReport(t, stdout.Bytes()); report.Coverage.Evaluated != 2 {
		t.Errorf("report evaluated %d, want 2", report.Coverage.Evaluated)
	}
}

// TestRunInterruptedPrintsWhatItMeasured is the defect this was written for:
// a Ctrl-C threw away every measurement, printed nothing, and exited 1 after
// minutes of real calls.
func TestRunInterruptedPrintsWhatItMeasured(t *testing.T) {
	benchEnv(t)
	const cases = 10
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	fake := &fakeEvaluator{onCase: func(n int) {
		if n == 3 {
			cancel()
		}
	}}
	var stdout, stderr bytes.Buffer

	err := runWith(ctx, []string{"-dataset", datasetFile(t, cases), "-json"}, &stdout, &stderr, evaluatorFor(fake))

	if err == nil {
		t.Fatal("an interrupted run reported success")
	}
	if !strings.Contains(err.Error(), "interrupted") {
		t.Errorf("error = %q, want it to say the run was interrupted", err)
	}
	report := decodeReport(t, stdout.Bytes())
	if report.Coverage.Evaluated == 0 {
		t.Fatalf("the partial report measured nothing:\n%s", stdout.String())
	}
	if report.Coverage.Attempted >= cases {
		t.Errorf("report covers %d attempts of %d: the run was not stopped",
			report.Coverage.Attempted, cases)
	}
}

// decodeReport reads the -json document the command wrote.
func decodeReport(t *testing.T, out []byte) calibration.Report {
	t.Helper()
	var report calibration.Report
	if err := json.Unmarshal(out, &report); err != nil {
		t.Fatalf("the report is not a JSON document (%v):\n%s", err, out)
	}
	return report
}

// evaluatorFor hands the same fake back whatever the configuration says.
func evaluatorFor(e calibration.Evaluator) buildEvaluator {
	return func(config.Config, *slog.Logger) (calibration.Evaluator, error) { return e, nil }
}

// ------------------------------------------------------------- the wiring --

// TestNewRunnerCarriesTheConfiguration checks the settings a benchmark is
// supposed to inherit from the server's own environment. Each expected value
// differs from every other and from every zero value, so no field can be
// reading another's.
func TestNewRunnerCarriesTheConfiguration(t *testing.T) {
	cfg := config.Config{
		Model:          "probe-7",
		Temperature:    0.3,
		MaxConcurrency: 6,
		RequestTimeout: 17 * time.Second,
		BaseURL:        "https://gw.example/v1?key=QUERYSECRET-zzz999",
	}

	runner := newRunner(cfg, &fakeEvaluator{}, 4)

	if runner.Model != cfg.Model {
		t.Errorf("Model = %q, want %q", runner.Model, cfg.Model)
	}
	if runner.Temperature != cfg.Temperature {
		t.Errorf("Temperature = %v, want %v", runner.Temperature, cfg.Temperature)
	}
	if runner.Repeat != 4 {
		t.Errorf("Repeat = %d, want 4", runner.Repeat)
	}
	if runner.Concurrency != cfg.MaxConcurrency {
		t.Errorf("Concurrency = %d, want %d", runner.Concurrency, cfg.MaxConcurrency)
	}
	if runner.Timeout != cfg.RequestTimeout {
		t.Errorf("Timeout = %v, want %v: %s is the per-attempt deadline",
			runner.Timeout, cfg.RequestTimeout, config.EnvRequestTimeout)
	}
	if runner.Scrub == nil {
		t.Fatal("Scrub is nil, so a failure would carry the base URL's credential into the report")
	}
	if got := runner.Scrub("calling https://gw.example/v1?key=QUERYSECRET-zzz999"); strings.Contains(got, "QUERYSECRET") {
		t.Errorf("Scrub left the credential in %q", got)
	}
}

// TestScrubberRemovesWhatCredentialsInFinds is the defect: the -json document
// the README tells people to store and diff carried the gateway key out of
// the base URL, in full.
func TestScrubberRemovesWhatCredentialsInFinds(t *testing.T) {
	tests := []struct {
		name     string
		cfg      config.Config
		text     string
		unwanted []string
	}{
		{
			name:     "a key in the query",
			cfg:      config.Config{BaseURL: "http://llm.internal.corp:8443/v1?api-key=QUERYSECRET-zzz999"},
			text:     `inference: chat completion: Post "http://llm.internal.corp:8443/v1?api-key=QUERYSECRET-zzz999/chat/completions": refused`,
			unwanted: []string{"QUERYSECRET-zzz999"},
		},
		{
			// net/http masks the password in a *url.Error and prints the
			// username, and CredentialsIn treats the username as a secret.
			name:     "userinfo, whose username net/http does not mask",
			cfg:      config.Config{BaseURL: "https://USERSECRET-abc:PASSSECRET-def@gw.example/v1"},
			text:     `inference: chat completion: Post "https://USERSECRET-abc:xxxxx@gw.example/v1/chat/completions": refused`,
			unwanted: []string{"USERSECRET-abc"},
		},
		{
			name:     "a key the provider echoed back",
			cfg:      config.Config{BaseURL: "https://gw.example/v1", APIKey: "sk-live-0123456789"},
			text:     "inference: http 401: rejected key sk-live-0123456789",
			unwanted: []string{"sk-live-0123456789"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scrub := scrubber(tt.cfg)
			if scrub == nil {
				t.Fatal("scrubber returned nil, so nothing would be removed")
			}
			got := scrub(tt.text)
			for _, secret := range tt.unwanted {
				if strings.Contains(got, secret) {
					t.Errorf("the credential %q survived into %q", secret, got)
				}
			}
			if !strings.Contains(got, redacted) {
				t.Errorf("nothing was redacted in %q", got)
			}
		})
	}
}

// A base URL with nothing to hide costs nothing, and a throwaway local key is
// too short to strike out without making the message unreadable.
func TestScrubberIsNilWhenThereIsNothingToRemove(t *testing.T) {
	for _, cfg := range []config.Config{
		{BaseURL: "https://api.example/v1"},
		{BaseURL: "https://api.example/v1", APIKey: "abc"},
	} {
		if scrubber(cfg) != nil {
			t.Errorf("scrubber(%+v) is not nil", cfg)
		}
	}
}

// TestBenchEvaluatorLetsTheThrottleBeTheOnlyBound: the evaluator's own
// per-evaluation ceiling is switched off, so a single case may use the whole
// of PERCEPTEA_MAX_CONCURRENCY. Left at its default it would hold one case to
// classifier.DefaultMaxConcurrency however high the throttle was set.
func TestBenchEvaluatorLetsTheThrottleBeTheOnlyBound(t *testing.T) {
	// Above the classifier's own default, so the two limits cannot be
	// mistaken for each other.
	const limit = classifier.DefaultMaxConcurrency + 4
	scorer := &countingScorer{started: make(chan struct{}, limit), release: make(chan struct{})}
	evaluator := benchEvaluator(scorer, limit)

	// One case, one choice question with `limit` options, so the whole
	// fan-out belongs to a single evaluation.
	options := classifier.NewOrderedMap[string]()
	for i := range limit {
		options.Set(fmt.Sprintf("o%d", i), fmt.Sprintf("option %d", i))
	}
	questions := classifier.NewOrderedMap[classifier.Question]()
	questions.Set("pick", classifier.Question{
		Type:         classifier.TypeChoice,
		Instructions: "Which one?",
		Options:      *options,
	})

	done := make(chan error, 1)
	go func() {
		_, err := evaluator.Evaluate(context.Background(), classifier.Request{
			State:     classifier.State{},
			Questions: *questions,
		})
		done <- err
	}()

	// A blocking receive per expected call: it cannot pass early, and the
	// (limit)th is the one that would never arrive if the evaluator kept its
	// own ceiling of classifier.DefaultMaxConcurrency.
	timeout := time.After(5 * time.Second)
	for i := range limit {
		select {
		case <-scorer.started:
		case <-timeout:
			t.Fatalf("only %d of %d candidates were in flight: the evaluator is bounding them at %d",
				i, limit, classifier.DefaultMaxConcurrency)
		}
	}
	close(scorer.release)
	<-done
}

// countingScorer reports every call that starts and holds it until released.
type countingScorer struct {
	started chan struct{}
	release chan struct{}
}

func (c *countingScorer) Score(context.Context, classifier.ScoreRequest) (classifier.ScoreResult, error) {
	c.started <- struct{}{}
	<-c.release
	return classifier.ScoreResult{Probability: 0.5}, nil
}

// TestLiveEvaluatorHonoursTheConfiguredScorer: a benchmark exists to say
// whether a configuration's probabilities mean anything, and the scorer is
// half of that configuration. inference.New rejects a scorer it does not
// know, which is what makes the wiring observable without a call.
func TestLiveEvaluatorHonoursTheConfiguredScorer(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	base := config.Config{
		APIKey:         "sk-test-0123456789",
		BaseURL:        "https://inference.example/v1",
		Model:          "probe-1",
		MaxConcurrency: 4,
	}

	for _, scorer := range []string{config.ScorerChat, config.ScorerLogprob} {
		cfg := base
		cfg.Scorer = scorer
		if _, err := liveEvaluator(cfg, logger); err != nil {
			t.Errorf("liveEvaluator with %s=%q: %v", config.EnvScorer, scorer, err)
		}
	}

	cfg := base
	cfg.Scorer = "not-a-scorer"
	if _, err := liveEvaluator(cfg, logger); err == nil {
		t.Errorf("liveEvaluator accepted %s=%q, so the configured scorer is not reaching the client",
			config.EnvScorer, cfg.Scorer)
	}
}

func TestWriteEmitsTextByDefaultAndJSONOnRequest(t *testing.T) {
	report := calibration.Report{
		Dataset:     "example.jsonl",
		Model:       "some-model",
		Coverage:    calibration.Coverage{Cases: 1, Repeat: 1, Attempted: 1, Evaluated: 1},
		Calibration: calibration.Calibrate([]calibration.Prediction{{P: 0.75, Holds: true}}),
	}

	var text bytes.Buffer
	if err := write(&text, report, false); err != nil {
		t.Fatalf("write text: %v", err)
	}
	if !strings.Contains(text.String(), "perceptea calibration benchmark") {
		t.Errorf("text output = %q", text.String())
	}

	var doc bytes.Buffer
	if err := write(&doc, report, true); err != nil {
		t.Fatalf("write json: %v", err)
	}
	var back calibration.Report
	if err := json.Unmarshal(doc.Bytes(), &back); err != nil {
		t.Fatalf("the -json output is not a JSON document: %v", err)
	}
	if back.Dataset != report.Dataset || back.Calibration.ECE != report.Calibration.ECE {
		t.Errorf("decoded report = %+v, want the same numbers as %+v", back, report)
	}
}

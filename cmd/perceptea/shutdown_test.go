package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// serveEnv re-executes this test binary as the server itself: with it set,
// TestMain runs main() instead of the suite. It is the only way to see what
// main() does with a signal and what exit code it leaves behind.
const serveEnv = "PERCEPTEA_TEST_SERVE"

func TestMain(m *testing.M) {
	if os.Getenv(serveEnv) == "1" {
		main()
		// main() returns only on a clean stop; a failure exits 1 from inside.
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// TestRunShutsDownGracefully holds a request open across a SIGTERM: the
// server must finish answering it, and then exit 0. A Close() in place of
// Shutdown() cuts the connection and this goes red; an error return on the
// shutdown path shows up as a non-zero exit.
func TestRunShutsDownGracefully(t *testing.T) {
	if _, err := os.Stat(os.Args[0]); err != nil {
		t.Skipf("the test binary is not on disk: %v", err)
	}

	child := exec.Command(os.Args[0])
	child.Env = []string{
		serveEnv + "=1",
		"PATH=" + os.Getenv("PATH"),
		"HOME=" + os.Getenv("HOME"),
		"PERCEPTEA_ADDR=127.0.0.1:0",
		"PERCEPTEA_LOG_FORMAT=json",
		// A long request timeout is the case the two-minute cap broke.
		"PERCEPTEA_REQUEST_TIMEOUT=10m",
		// Credential-bearing, and never called: the request below is refused
		// before any evaluator is built, so no test here touches a socket it
		// did not open itself.
		"PERCEPTEA_BASE_URL=https://svc:sk-secret-0123456789@127.0.0.1:1/v1?key=abc123",
		"PERCEPTEA_API_KEY=sk-child-0123456789",
	}
	stderr, err := child.StderrPipe()
	if err != nil {
		t.Fatalf("wiring stderr: %v", err)
	}
	if err := child.Start(); err != nil {
		t.Fatalf("starting the server: %v", err)
	}

	var (
		mu    sync.Mutex
		lines []string
	)
	addrCh := make(chan string, 1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		sc := bufio.NewScanner(stderr)
		for sc.Scan() {
			line := sc.Text()
			mu.Lock()
			lines = append(lines, line)
			mu.Unlock()
			var entry map[string]any
			if err := json.Unmarshal([]byte(line), &entry); err != nil {
				continue
			}
			if entry["msg"] == "perceptea listening" {
				addr, _ := entry["addr"].(string)
				select {
				case addrCh <- addr:
				default:
				}
			}
		}
	}()
	t.Cleanup(func() {
		_ = child.Process.Kill()
		<-done
	})

	var addr string
	select {
	case addr = <-addrCh:
	case <-time.After(20 * time.Second):
		t.Fatal("the server never said what it was listening on")
	}
	if addr == "" {
		t.Fatal("the startup line carries no address")
	}

	// The startup line is the other place a base URL reaches a log.
	mu.Lock()
	startup := strings.Join(lines, "\n")
	mu.Unlock()
	for _, secret := range []string{"sk-secret-0123456789", "sk-child-0123456789", "key=abc123"} {
		if strings.Contains(startup, secret) {
			t.Errorf("the server's own log contains %q:\n%s", secret, startup)
		}
	}

	// Open a request and stop half way through its body, so that the server
	// is mid-request when the signal arrives.
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dialling the server: %v", err)
	}
	defer conn.Close()

	// A body the server refuses on its own, so draining it needs no provider.
	const tail = `}`
	head := `{"state":"in flight"`
	body := head + tail
	fmt.Fprintf(conn, "POST /api/evaluate HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s",
		addr, len(body), head)

	// Give the server time to reach the body read, then ask it to stop.
	time.Sleep(200 * time.Millisecond)
	if err := child.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatalf("signalling the server: %v", err)
	}
	time.Sleep(200 * time.Millisecond)

	// Finish the request the server is draining.
	if _, err := fmt.Fprint(conn, tail); err != nil {
		t.Fatalf("the connection was cut before the request could finish: %v", err)
	}
	_ = conn.SetReadDeadline(time.Now().Add(20 * time.Second))
	answer, err := readAll(conn)
	if err != nil {
		t.Fatalf("reading the answer to an in-flight request: %v", err)
	}
	// The body names no questions, so it is refused on its own terms — what
	// matters is that it was answered at all rather than cut off.
	if !strings.HasPrefix(answer, "HTTP/1.1 400") {
		t.Fatalf("the in-flight request was not answered: %q", answer)
	}
	if !strings.Contains(answer, `"code":"invalid_request"`) {
		t.Errorf("the drained request got no JSON body: %q", answer)
	}

	waitErr := make(chan error, 1)
	go func() { waitErr <- child.Wait() }()
	select {
	case err := <-waitErr:
		if err != nil {
			mu.Lock()
			out := strings.Join(lines, "\n")
			mu.Unlock()
			t.Fatalf("the server exited with %v, want 0:\n%s", err, out)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("the server did not exit after SIGTERM")
	}

	mu.Lock()
	out := strings.Join(lines, "\n")
	logged := append([]string(nil), lines...)
	mu.Unlock()
	for _, want := range []string{"shutting down", "stopped"} {
		if !strings.Contains(out, want) {
			t.Errorf("the log has no %q line:\n%s", want, out)
		}
	}
	// The grace must clear the configured request timeout rather than a cap
	// below it: at two minutes this request would have been cut off.
	grace, ok := fieldOf(logged, "shutting down", "grace")
	if !ok {
		t.Fatalf("no grace in the shutdown line:\n%s", out)
	}
	if want := float64(10*time.Minute + shutdownSlack); grace != want {
		t.Errorf("grace = %v, want %v", time.Duration(grace.(float64)), time.Duration(want))
	}
}

// fieldOf pulls one field out of the first JSON log line with the given msg.
func fieldOf(lines []string, msg, field string) (any, bool) {
	for _, line := range lines {
		var entry map[string]any
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			continue
		}
		if entry["msg"] != msg {
			continue
		}
		v, ok := entry[field]
		return v, ok
	}
	return nil, false
}

// readAll reads until the peer closes or the deadline passes.
func readAll(conn net.Conn) (string, error) {
	var b strings.Builder
	buf := make([]byte, 4096)
	for {
		n, err := conn.Read(buf)
		b.Write(buf[:n])
		if err != nil {
			if b.Len() > 0 {
				return b.String(), nil
			}
			return "", err
		}
		// One response is all this test sends for, and the connection is
		// marked close, so a complete body is the end of it.
		if strings.Contains(b.String(), "\r\n\r\n") && strings.Contains(b.String(), "}") {
			return b.String(), nil
		}
	}
}

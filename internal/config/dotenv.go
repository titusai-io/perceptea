package config

import (
	"bufio"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strings"
)

// LoadDotEnv reads a .env file into the process environment.
//
// It is deliberately small and forgiving, so that no dependency is needed for
// a development convenience: blank lines and # comments are skipped, an
// optional "export " prefix is allowed, a value may be wrapped in matching
// single or double quotes (escapes are expanded only inside double quotes),
// either form of value may be followed by a # comment, and a line that is not
// KEY=VALUE is ignored rather than rejected.
//
// A variable that is already set in the environment is never overwritten, so
// a real environment always beats the file. A missing file is not an error.
func LoadDotEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		key, value, ok := parseDotEnvLine(sc.Text())
		if !ok {
			continue
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		if err := os.Setenv(key, value); err != nil {
			return fmt.Errorf("%s: setting %s: %w", path, key, err)
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

// parseDotEnvLine splits one line into a key and a value, reporting false for
// a line that carries no assignment.
func parseDotEnvLine(raw string) (key, value string, ok bool) {
	line := strings.TrimSpace(raw)
	if line == "" || strings.HasPrefix(line, "#") {
		return "", "", false
	}
	if rest, found := strings.CutPrefix(line, "export"); found && strings.IndexAny(rest, " \t") == 0 {
		line = strings.TrimSpace(rest)
	}
	name, val, found := strings.Cut(line, "=")
	if !found {
		return "", "", false
	}
	key = strings.TrimSpace(name)
	if !validKey(key) {
		return "", "", false
	}
	value = strings.TrimSpace(val)
	// A quoted value is settled by its closing quote, not by the end of the
	// line: KEY="sk-abc def" # comment is a quoted value with a comment after
	// it. Matching the last character instead kept the quotes, and the key
	// went out as Bearer "sk-abc def".
	if len(value) >= 2 {
		if q := value[0]; q == '"' || q == '\'' {
			if end := closingQuote(value[1:], q); end >= 0 {
				inner, rest := value[1:1+end], strings.TrimSpace(value[2+end:])
				// Anything but a comment after the closing quote means this
				// was never a quoted value; fall through and keep it verbatim,
				// as an unterminated quote already does.
				if rest == "" || strings.HasPrefix(rest, "#") {
					if q == '"' {
						inner = expandEscapes(inner)
					}
					return key, inner, true
				}
			}
		}
	}
	// An unquoted value ends at an inline comment: whitespace, then '#'. A
	// value that starts with '#' is not a comment, having survived the check
	// at the top of this function.
	for i := 1; i < len(value); i++ {
		if value[i] == '#' && (value[i-1] == ' ' || value[i-1] == '\t') {
			value = strings.TrimSpace(value[:i])
			break
		}
	}
	return key, value, true
}

// closingQuote returns the index in s of the quote that closes a value opened
// with q, or -1 when there is none. Inside a double-quoted value a backslash
// escapes the next byte, so that KEY="a \" b" closes at the last quote and not
// at the escaped one.
func closingQuote(s string, q byte) int {
	for i := 0; i < len(s); i++ {
		if q == '"' && s[i] == '\\' {
			i++
			continue
		}
		if s[i] == q {
			return i
		}
	}
	return -1
}

// validKey reports whether s looks like a shell variable name.
func validKey(s string) bool {
	if s == "" || (s[0] >= '0' && s[0] <= '9') {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '_':
		default:
			return false
		}
	}
	return true
}

// expandEscapes expands the handful of escapes a double-quoted value may use.
func expandEscapes(s string) string {
	if !strings.ContainsRune(s, '\\') {
		return s
	}
	return strings.NewReplacer(`\n`, "\n", `\r`, "\r", `\t`, "\t", `\"`, `"`, `\\`, `\`).Replace(s)
}

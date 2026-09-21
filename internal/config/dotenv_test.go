package config

import (
	"os"
	"path/filepath"
	"testing"
)

// writeDotEnv puts contents in a temporary .env and returns its path.
func writeDotEnv(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), ".env")
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("writing %s: %v", path, err)
	}
	return path
}

// unsetAfter makes sure a variable the test introduced does not leak into the
// rest of the run.
func unsetAfter(t *testing.T, keys ...string) {
	t.Helper()
	t.Cleanup(func() {
		for _, k := range keys {
			os.Unsetenv(k)
		}
	})
}

func TestLoadDotEnvParsing(t *testing.T) {
	const body = `
# a comment line
   # an indented comment

DOTENV_PLAIN=plain
DOTENV_SPACED  =   spaced value
export DOTENV_EXPORTED=exported
export	DOTENV_TAB_EXPORTED=tabbed
DOTENV_DQUOTED="  padded  "
DOTENV_SQUOTED='  single  '
DOTENV_HASH_IN_QUOTES="value # not a comment"
DOTENV_TRAILING_COMMENT=value # a comment
DOTENV_QUOTED_COMMENT="sk-abc def" # the key
DOTENV_TAB_COMMENT=value	# a comment after a tab
DOTENV_EMPTY=
DOTENV_EQUALS=a=b=c
DOTENV_ESCAPES="line1\nline2\ttabbed"
DOTENV_UNBALANCED="only-leading
not an assignment line
=novalue
9STARTS_WITH_DIGIT=nope
`
	keys := []string{
		"DOTENV_PLAIN", "DOTENV_SPACED", "DOTENV_EXPORTED", "DOTENV_TAB_EXPORTED",
		"DOTENV_DQUOTED", "DOTENV_SQUOTED", "DOTENV_HASH_IN_QUOTES",
		"DOTENV_TRAILING_COMMENT", "DOTENV_QUOTED_COMMENT", "DOTENV_TAB_COMMENT",
		"DOTENV_EMPTY", "DOTENV_EQUALS", "DOTENV_ESCAPES",
		"DOTENV_UNBALANCED", "9STARTS_WITH_DIGIT",
	}
	unsetAfter(t, keys...)

	if err := LoadDotEnv(writeDotEnv(t, body)); err != nil {
		t.Fatalf("LoadDotEnv: %v", err)
	}

	want := map[string]string{
		"DOTENV_PLAIN":            "plain",
		"DOTENV_SPACED":           "spaced value",
		"DOTENV_EXPORTED":         "exported",
		"DOTENV_TAB_EXPORTED":     "tabbed",
		"DOTENV_DQUOTED":          "  padded  ",
		"DOTENV_SQUOTED":          "  single  ",
		"DOTENV_HASH_IN_QUOTES":   "value # not a comment",
		"DOTENV_TRAILING_COMMENT": "value",
		"DOTENV_QUOTED_COMMENT":   "sk-abc def",
		"DOTENV_TAB_COMMENT":      "value",
		"DOTENV_EMPTY":            "",
		"DOTENV_EQUALS":           "a=b=c",
		"DOTENV_ESCAPES":          "line1\nline2\ttabbed",
		"DOTENV_UNBALANCED":       `"only-leading`,
	}
	for k, w := range want {
		got, ok := os.LookupEnv(k)
		if !ok {
			t.Errorf("%s was not set", k)
			continue
		}
		if got != w {
			t.Errorf("%s = %q, want %q", k, got, w)
		}
	}
	if _, ok := os.LookupEnv("9STARTS_WITH_DIGIT"); ok {
		t.Error("a key starting with a digit was accepted")
	}
}

func TestLoadDotEnvNeverOverwrites(t *testing.T) {
	t.Setenv("DOTENV_EXISTING", "from-the-environment")
	t.Setenv("DOTENV_EXISTING_EMPTY", "")
	unsetAfter(t, "DOTENV_NEW")

	body := "DOTENV_EXISTING=from-the-file\nDOTENV_EXISTING_EMPTY=from-the-file\nDOTENV_NEW=from-the-file\n"
	if err := LoadDotEnv(writeDotEnv(t, body)); err != nil {
		t.Fatalf("LoadDotEnv: %v", err)
	}

	if got := os.Getenv("DOTENV_EXISTING"); got != "from-the-environment" {
		t.Errorf("DOTENV_EXISTING = %q, want the environment to win", got)
	}
	if got := os.Getenv("DOTENV_EXISTING_EMPTY"); got != "" {
		t.Errorf("DOTENV_EXISTING_EMPTY = %q, want a set-but-empty variable to win too", got)
	}
	if got := os.Getenv("DOTENV_NEW"); got != "from-the-file" {
		t.Errorf("DOTENV_NEW = %q, want the file value", got)
	}
}

func TestLoadDotEnvMissingFileIsNotAnError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "definitely-absent", ".env")
	if err := LoadDotEnv(path); err != nil {
		t.Errorf("LoadDotEnv(%q) = %v, want nil for a missing file", path, err)
	}
}

func TestLoadDotEnvUnreadableFileIsAnError(t *testing.T) {
	// A directory stands in for anything that exists but cannot be read as a
	// file; on some systems opening it succeeds and reading it fails, so both
	// outcomes are acceptable as long as a value is not invented.
	dir := t.TempDir()
	if err := LoadDotEnv(dir); err == nil {
		t.Errorf("LoadDotEnv(%q) = nil, want an error for a directory", dir)
	}
}

func TestLoadDotEnvFeedsLoad(t *testing.T) {
	unsetAfter(t, "PERCEPTEA_MODEL", "PERCEPTEA_LOG_FORMAT")
	os.Unsetenv("PERCEPTEA_MODEL")
	os.Unsetenv("PERCEPTEA_LOG_FORMAT")

	body := "PERCEPTEA_MODEL=from-dotenv\nPERCEPTEA_LOG_FORMAT=json\n"
	if err := LoadDotEnv(writeDotEnv(t, body)); err != nil {
		t.Fatalf("LoadDotEnv: %v", err)
	}
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load(): %v", err)
	}
	if cfg.Model != "from-dotenv" || cfg.LogFormat != "json" {
		t.Errorf("Load() = %+v, want the .env values", cfg)
	}
}

func TestParseDotEnvLine(t *testing.T) {
	tests := []struct {
		line      string
		key, want string
		ok        bool
	}{
		{"", "", "", false},
		{"   ", "", "", false},
		{"# comment", "", "", false},
		{"#KEY=value", "", "", false},
		{"KEY=value", "KEY", "value", true},
		{"exported=1", "exported", "1", true}, // not the export prefix
		{"export=1", "export", "1", true},     // a variable actually named export
		{"export KEY=value", "KEY", "value", true},
		{"KEY='#1'", "KEY", "#1", true},
		{"KEY=#1", "KEY", "#1", true}, // only " #" starts a comment
		{"KEY=a b # c", "KEY", "a b", true},
		{"KEY=\"a\" ", "KEY", "a", true},
		{"KEY", "", "", false},
		{"KEY WITH SPACE=value", "", "", false},
		{"KEY-WITH-DASH=value", "", "", false},
		{"_UNDERSCORE=ok", "_UNDERSCORE", "ok", true},
		{"K2=ok", "K2", "ok", true},
		{`KEY="a\\b"`, "KEY", `a\b`, true},
		{`KEY='a\nb'`, "KEY", `a\nb`, true}, // single quotes keep escapes
		{`KEY="unterminated`, "KEY", `"unterminated`, true},

		// A quoted value followed by a comment kept its quotes, so the key
		// went out as Bearer "sk-abc def" and every call 401'd with nothing
		// in the log to explain it.
		{`KEY="sk-abc def" # comment`, "KEY", "sk-abc def", true},
		{`KEY='sk-abc def' # comment`, "KEY", "sk-abc def", true},
		{"KEY=\"sk-abc def\"\t# comment", "KEY", "sk-abc def", true},
		{`KEY="" # empty`, "KEY", "", true},
		{`KEY="a"#immediate`, "KEY", "a", true},
		{`KEY="a \" b" # escaped quote inside`, "KEY", `a " b`, true},
		{`KEY="a" trailing junk`, "KEY", `"a" trailing junk`, true},

		// A tab before the # starts a comment, as a space already did.
		{"KEY=value\t# comment", "KEY", "value", true},
		{"KEY=a b\t\t# comment", "KEY", "a b", true},
		{"KEY=va#lue", "KEY", "va#lue", true}, // no whitespace, no comment
	}
	for _, tt := range tests {
		key, value, ok := parseDotEnvLine(tt.line)
		if ok != tt.ok || key != tt.key || value != tt.want {
			t.Errorf("parseDotEnvLine(%q) = (%q, %q, %v), want (%q, %q, %v)",
				tt.line, key, value, ok, tt.key, tt.want, tt.ok)
		}
	}
}

package api

import (
	"net/http"

	"github.com/titusai-io/perceptea/internal/config"
)

// serviceName is the identifier /api/health reports.
const serviceName = "perceptea"

// healthResponse is the body of GET /api/health. It says whether a key is
// configured, never what it is, and it carries no base URL: a base URL can
// hold a credential of its own, and anything here that grew one would have to
// go through [config.DisplayBaseURL] first.
type healthResponse struct {
	OK               bool   `json:"ok"`
	Service          string `json:"service"`
	Provider         string `json:"provider"`
	Model            string `json:"model"`
	APIKeyConfigured bool   `json:"api_key_configured"`
}

// handleHealth reports that the process is up and what it would call.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.writeJSON(w, http.StatusOK, healthResponse{
		OK:               true,
		Service:          serviceName,
		Provider:         s.cfg.Provider,
		Model:            s.cfg.Model,
		APIKeyConfigured: s.cfg.APIKeyConfigured(),
	})
}

// providerInfo is one preset as GET /api/providers reports it: names and
// URLs only, never a key.
type providerInfo struct {
	Name         string `json:"name"`
	BaseURL      string `json:"base_url"`
	DefaultModel string `json:"default_model"`
	EnvKey       string `json:"env_key"`
	Active       bool   `json:"active"`
}

// activeInfo describes what this server will actually call.
type activeInfo struct {
	Provider                string `json:"provider"`
	BaseURL                 string `json:"base_url"`
	Model                   string `json:"model"`
	APIKeyConfigured        bool   `json:"api_key_configured"`
	AllowRequestCredentials bool   `json:"allow_request_credentials"`
}

// providersResponse is the body of GET /api/providers.
type providersResponse struct {
	Active    activeInfo     `json:"active"`
	Providers []providerInfo `json:"providers"`
}

// handleProviders lists the presets and marks the one in use.
func (s *Server) handleProviders(w http.ResponseWriter, r *http.Request) {
	presets := config.Providers()
	out := make([]providerInfo, 0, len(presets))
	for _, p := range presets {
		out = append(out, providerInfo{
			Name:         p.Name,
			BaseURL:      p.BaseURL,
			DefaultModel: p.DefaultModel,
			EnvKey:       p.EnvKey,
			Active:       p.Name == s.cfg.Provider,
		})
	}
	s.writeJSON(w, http.StatusOK, providersResponse{
		Active: activeInfo{
			Provider: s.cfg.Provider,
			// Rendered for display: an operator pointing this at a gateway
			// may have put the credential in the URL, and this endpoint
			// answers anyone who can reach the port.
			BaseURL:                 config.DisplayBaseURL(s.cfg.BaseURL),
			Model:                   s.cfg.Model,
			APIKeyConfigured:        s.cfg.APIKeyConfigured(),
			AllowRequestCredentials: s.cfg.AllowRequestCredentials,
		},
		Providers: out,
	})
}

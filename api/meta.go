package api

import (
	"net/http"

	"github.com/titusai-io/perceptea/internal/config"
)

// serviceName is the identifier /api/health reports.
const serviceName = "perceptea"

// healthResponse is the body of GET /api/health. It says whether a key is
// configured, never what it is, and the inference base URL it carries has
// been through [config.DisplayBaseURL]: an operator pointing this at a
// gateway may have put a credential in the URL, and this endpoint answers
// anyone who can reach the port.
type healthResponse struct {
	OK               bool   `json:"ok"`
	Service          string `json:"service"`
	InferenceBaseURL string `json:"inference_base_url"`
	Model            string `json:"model"`
	APIKeyConfigured bool   `json:"api_key_configured"`
}

// handleHealth reports that the process is up and what it would call.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.writeJSON(w, http.StatusOK, healthResponse{
		OK:               true,
		Service:          serviceName,
		InferenceBaseURL: config.DisplayBaseURL(s.cfg.BaseURL),
		Model:            s.cfg.Model,
		APIKeyConfigured: s.cfg.APIKeyConfigured(),
	})
}

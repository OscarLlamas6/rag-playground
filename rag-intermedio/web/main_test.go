package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

func get(t *testing.T, h http.Handler, path string) (int, string) {
	t.Helper()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
	body, _ := io.ReadAll(rec.Result().Body)
	return rec.Code, string(body)
}

func TestServesTheInterface(t *testing.T) {
	backend, _ := url.Parse("http://127.0.0.1:1")
	h := newHandler(backend)
	for path, want := range map[string]string{"/": "RAG intermedio", "/app.js": "fetch", "/style.css": "--"} {
		code, body := get(t, h, path)
		if code != http.StatusOK || !strings.Contains(body, want) {
			t.Errorf("%s: código %d, no contiene %q", path, code, want)
		}
	}
}

func TestProxiesApiToBackend(t *testing.T) {
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"ruta":"` + r.URL.Path + `"}`))
	}))
	defer fake.Close()
	backend, _ := url.Parse(fake.URL)
	code, body := get(t, newHandler(backend), "/api/status")
	if code != http.StatusOK || body != `{"ruta":"/api/status"}` {
		t.Fatalf("proxy: código %d, cuerpo %q", code, body)
	}
}

func TestBackendDownGivesReadableJSON(t *testing.T) {
	backend, _ := url.Parse("http://127.0.0.1:1") // puerto sin nadie escuchando
	code, body := get(t, newHandler(backend), "/api/status")
	if code != http.StatusBadGateway || !strings.Contains(body, `"detail"`) || !strings.Contains(body, "no responde") {
		t.Fatalf("esperaba 502 con detail, obtuve %d %q", code, body)
	}
}

func TestHealthz(t *testing.T) {
	backend, _ := url.Parse("http://127.0.0.1:1")
	if code, body := get(t, newHandler(backend), "/healthz"); code != 200 || !strings.Contains(body, "ok") {
		t.Fatalf("healthz: %d %q", code, body)
	}
}

func TestProxyPassesRangeRequestsThrough(t *testing.T) {
	// Un backend que responde como FastAPI: ServeContent atiende Range con 206 + Content-Range + Accept-Ranges.
	data := strings.Repeat("0123456789", 100)
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/mpeg")
		http.ServeContent(w, r, "x.mp3", time.Time{}, strings.NewReader(data))
	}))
	defer fake.Close()
	backend, _ := url.Parse(fake.URL)
	req := httptest.NewRequest(http.MethodGet, "/api/media/x.mp3", nil)
	req.Header.Set("Range", "bytes=10-19")
	rec := httptest.NewRecorder()
	newHandler(backend).ServeHTTP(rec, req)
	body, _ := io.ReadAll(rec.Result().Body)
	if rec.Code != http.StatusPartialContent || string(body) != data[10:20] {
		t.Fatalf("esperaba 206 con 10 bytes, obtuve %d %q", rec.Code, body)
	}
	if got := rec.Header().Get("Content-Range"); got != "bytes 10-19/1000" {
		t.Errorf("Content-Range = %q", got)
	}
	if got := rec.Header().Get("Accept-Ranges"); got != "bytes" {
		t.Errorf("Accept-Ranges = %q", got)
	}
}

func TestProxyTimeoutFromEnv(t *testing.T) {
	t.Setenv("PROXY_TIMEOUT_SECONDS", "")
	if proxyTimeout() != 900*time.Second {
		t.Errorf("por defecto debe ser 900 s, es %v", proxyTimeout())
	}
	t.Setenv("PROXY_TIMEOUT_SECONDS", "42")
	if proxyTimeout() != 42*time.Second {
		t.Errorf("con 42 debe ser 42 s, es %v", proxyTimeout())
	}
	t.Setenv("PROXY_TIMEOUT_SECONDS", "abc")
	if proxyTimeout() != 900*time.Second {
		t.Errorf("un valor inválido debe usar 900 s")
	}
}

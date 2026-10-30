package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
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
	for path, want := range map[string]string{"/": "RAG simple", "/app.js": "fetch", "/style.css": "--"} {
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

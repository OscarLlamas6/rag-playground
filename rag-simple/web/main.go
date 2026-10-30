// Servidor web del RAG simple.
//
// Hace dos cosas, nada más:
//  1. Sirve la interfaz (HTML, CSS y JS embebidos en el binario con go:embed).
//  2. Reenvía todo lo que empiece por /api/ al backend de Python (FastAPI) con un proxy inverso.
//
// Así el navegador habla con un único origen (sin CORS) y el binario se despliega solo.
//
//	BACKEND_URL  dirección del backend      (por defecto http://127.0.0.1:8000)
//	WEB_ADDR     dirección donde escucha    (por defecto :8080)
package main

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"
)

//go:embed static
var staticFiles embed.FS

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// newHandler arma las rutas. Recibe el backend como parámetro para poder probarlo con un servidor falso.
func newHandler(backend *url.URL) http.Handler {
	proxy := httputil.NewSingleHostReverseProxy(backend)
	proxy.Transport = &http.Transport{
		DialContext:           (&net.Dialer{Timeout: 5 * time.Second}).DialContext,
		ResponseHeaderTimeout: 180 * time.Second, // una pregunta puede esperar al modelo de lenguaje
	}
	// Si el backend está apagado, la interfaz recibe un JSON con el mismo formato de error que FastAPI.
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		log.Printf("proxy: %v", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{
			"detail": fmt.Sprintf("El backend no responde en %s. ¿Está corriendo? (cd backend && uv run uvicorn app.main:app --port 8000)", backend),
		})
	}

	static, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.Handle("/api/", proxy)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.Handle("/", http.FileServer(http.FS(static)))
	return mux
}

func main() {
	backend, err := url.Parse(env("BACKEND_URL", "http://127.0.0.1:8000"))
	if err != nil || backend.Host == "" {
		log.Fatalf("BACKEND_URL no es válida: %q", os.Getenv("BACKEND_URL"))
	}
	addr := env("WEB_ADDR", ":8080")
	server := &http.Server{Addr: addr, Handler: newHandler(backend), ReadHeaderTimeout: 10 * time.Second}
	shown := addr
	if strings.HasPrefix(addr, ":") {
		shown = "localhost" + addr
	}
	log.Printf("RAG simple · interfaz en http://%s · backend en %s", shown, backend)
	log.Fatal(server.ListenAndServe())
}

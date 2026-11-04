// Servidor web del RAG avanzado.
//
// Hace dos cosas, nada más:
//  1. Sirve la interfaz (HTML, CSS y JS embebidos en el binario con go:embed).
//  2. Reenvía todo lo que empiece por /api/ al backend de Python (FastAPI) con un proxy inverso.
//
// Así el navegador habla con un único origen (sin CORS) y el binario se despliega solo.
//
//	BACKEND_URL  dirección del backend      (por defecto http://127.0.0.1:8020)
//	WEB_ADDR     dirección donde escucha    (por defecto :8082)
//	PROXY_TIMEOUT_SECONDS  cuánto espera una respuesta del backend (por defecto 900: la primera ingesta de PDF, audio y
//	             video transcribe y puede tardar minutos)
//
// El proxy deja pasar las cabeceras Range / Content-Range / Accept-Ranges tal cual: el navegador necesita saltar a un
// instante de un audio o un video sin descargarlo entero (GET /api/media/{nombre} responde 206).
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
	"strconv"
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

// proxyTimeout lee PROXY_TIMEOUT_SECONDS; si no es un número positivo, usa 900 s.
func proxyTimeout() time.Duration {
	if n, err := strconv.Atoi(os.Getenv("PROXY_TIMEOUT_SECONDS")); err == nil && n > 0 {
		return time.Duration(n) * time.Second
	}
	return 900 * time.Second
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
		ResponseHeaderTimeout: proxyTimeout(), // una pregunta puede esperar al modelo; una ingesta, a Docling y a Whisper
	}
	// Si el backend está apagado, la interfaz recibe un JSON con el mismo formato de error que FastAPI.
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		log.Printf("proxy: %v", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{
			"detail": fmt.Sprintf("El backend no responde en %s. ¿Está corriendo? (cd backend && uv run uvicorn app.main:app --port 8020)", backend),
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
	backend, err := url.Parse(env("BACKEND_URL", "http://127.0.0.1:8020"))
	if err != nil || backend.Host == "" {
		log.Fatalf("BACKEND_URL no es válida: %q", os.Getenv("BACKEND_URL"))
	}
	addr := env("WEB_ADDR", ":8082")
	server := &http.Server{Addr: addr, Handler: newHandler(backend), ReadHeaderTimeout: 10 * time.Second}
	shown := addr
	if strings.HasPrefix(addr, ":") {
		shown = "localhost" + addr
	}
	log.Printf("RAG avanzado · interfaz en http://%s · backend en %s", shown, backend)
	log.Fatal(server.ListenAndServe())
}

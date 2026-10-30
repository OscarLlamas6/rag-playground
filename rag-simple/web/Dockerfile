# syntax=docker/dockerfile:1
# Interfaz de rag-simple: un binario de Go que sirve la UI y reenvía /api al backend.
FROM golang:1.25-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY main.go ./
COPY static ./static
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/web .

FROM alpine:3.20
RUN adduser -D -u 10001 app
COPY --from=build /out/web /usr/local/bin/web
USER app
ENV BACKEND_URL=http://backend:8000 WEB_ADDR=:8080
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=3 CMD wget -qO- http://127.0.0.1:8080/healthz || exit 1
ENTRYPOINT ["web"]

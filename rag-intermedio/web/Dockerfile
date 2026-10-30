# syntax=docker/dockerfile:1
# Interfaz de rag-intermedio: un binario de Go que sirve la UI y reenvía /api al backend.
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
ENV BACKEND_URL=http://backend:8010 WEB_ADDR=:8081
EXPOSE 8081
HEALTHCHECK --interval=10s --timeout=3s --retries=3 CMD wget -qO- http://127.0.0.1:8081/healthz || exit 1
ENTRYPOINT ["web"]

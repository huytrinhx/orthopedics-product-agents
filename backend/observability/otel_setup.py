"""OpenTelemetry instrumentation, exported to any OTLP collector
(OTEL_EXPORTER_OTLP_ENDPOINT — local, self-hosted, or SaaS). Covers
service-level health: request latency, error rates, dependency calls
(Postgres, AuraDB). Call configure_otel() once at startup in
backend/api/main.py.
"""


def configure_otel(app, otlp_endpoint: str) -> None:
    raise NotImplementedError

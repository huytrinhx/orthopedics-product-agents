// Infrastructure as Code replacement for railway.toml (Config as Code is
// deprecated -- existing railway.toml/railway.json keep working only until
// 2026-12-01, see https://docs.railway.com/infrastructure-as-code).
//
// Built from `railway config pull` (the live service's actual settings --
// source binding, replicas, and every current env var, each wrapped in
// preserve() so real secret values never land in git) plus the fields
// railway.toml declares that pull doesn't surface, because they're
// currently applied via Config as Code rather than the dashboard: build
// (matches railway.toml's [build]), healthcheck/healthcheckTimeout, the
// preDeploy alembic migration, and restartPolicyType.
//
// NOT machine-generated via `railway config migrate`: that command's
// dry-run output (verified against CLI v5.43.1, both --lang ts and --lang
// py) silently drops restartPolicyType and, more importantly,
// releaseCommand entirely -- the alembic migration that must run before
// every deploy. Field names/shapes below are confirmed against the
// installed `railway` package's own type definitions
// (node_modules/railway/dist/index-C3uk0ruc.d.ts's DeployConfig/BuildConfig/
// IntentServiceConfig), not just doc prose, after `railway config plan`
// showed the migrate tool's gaps would have silently regressed both.
import { defineRailway, github, preserve, project, service } from "railway/iac";

export default defineRailway(() => {
  const orthopedicsProductAgents = service("orthopedics-product-agents", {
    source: github("huytrinhx/orthopedics-product-agents", { checkSuites: false }),
    build: {
      builder: "DOCKERFILE",
      dockerfilePath: "Dockerfile",
    },
    start: "",
    replicas: { "us-east4-eqdc4a": 1 },
    healthcheck: "/health",
    healthcheckTimeout: 100,
    preDeploy: "cd backend && alembic upgrade head",
    deploy: {
      restartPolicyType: "ON_FAILURE",
    },
    env: {
      ADMIN_EMAILS: preserve(),
      DATABASE_URL: preserve(),
      FRONTEND_PUBLIC_URL: preserve(),
      GOOGLE_CLIENT_ID: preserve(),
      GOOGLE_CLIENT_SECRET: preserve(),
      GOOGLE_REDIRECT_URI: preserve(),
      INGEST_DATA_DIR: preserve(),
      JWT_SECRET: preserve(),
      LANGFUSE_BASE_URL: preserve(),
      LANGFUSE_PUBLIC_KEY: preserve(),
      LANGFUSE_SECRET_KEY: preserve(),
      LANGFUSE_TRACING_ENVIRONMENT: preserve(),
      NEO4J_PASSWORD: preserve(),
      NEO4J_URI: preserve(),
      NEO4J_USER: preserve(),
      OPENAI_API_KEY: preserve(),
      OPENAI_CHAT_MODEL: preserve(),
      OPENAI_EMBEDDING_MODEL: preserve(),
    },
  });

  return project("ortho-mate", {
    resources: [orthopedicsProductAgents],
  });
});

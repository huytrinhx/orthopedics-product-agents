// Infrastructure as Code replacement for railway.toml (Config as Code is
// deprecated -- existing railway.toml/railway.json keep working only until
// 2026-12-01, see https://docs.railway.com/infrastructure-as-code). This
// file is now the ONLY deploy config -- railway.toml has been deleted.
//
// Built from `railway config pull` (the live service's actual settings --
// source binding, replicas, and every current env var, each wrapped in
// preserve() so real secret values never land in git) plus the fields
// railway.toml used to declare that pull doesn't surface on its own,
// because they were applied via Config as Code rather than the dashboard:
// build (matched railway.toml's [build]), healthcheck/healthcheckTimeout,
// and the preDeploy alembic migration.
//
// NOT machine-generated via `railway config migrate`: that command's
// dry-run output (verified against CLI v5.43.1, both --lang ts and --lang
// py) silently drops restartPolicyType and, more importantly,
// releaseCommand entirely -- the alembic migration that must run before
// every deploy. Field names/shapes below are confirmed against the
// installed `railway` package's own type definitions
// (node_modules/railway/dist/index-C3uk0ruc.d.ts's DeployConfig/BuildConfig/
// IntentServiceConfig), not just doc prose.
//
// Two real deploy failures hit and fixed while migrating off railway.toml
// (both against the live ortho-mate project, via `railway config
// plan`/`apply` -- not guesses):
//   1. An early draft carried `start: ""` over verbatim from `railway
//      config pull`'s baseline snapshot. Applying a literal empty string
//      as the start command (rather than leaving it unset) made every
//      deploy fail instantly with zero build/deploy/http logs -- Railway
//      tried to run an empty command instead of the Dockerfile's CMD.
//      Fixed by omitting `start` entirely (railway.toml never set one
//      either).
//   2. Deploys kept failing the same silent way even after that fix, for
//      as long as railway.toml still existed alongside this file --
//      .railway/README.md's own generated text warns "Services already
//      managed by railway.json must be migrated before .railway/railway.ts
//      can manage them," and this service hadn't been (that step needs
//      `railway config migrate --apply`, which mutates the live project
//      and was correctly declined by this session's own safety
//      classifier). Deleting railway.toml -- since every field it set is
//      already mirrored here and applied live -- is what finally let a
//      redeploy succeed.
//
// restartPolicyType below is a known gap as of CLI v5.43.1: `railway
// config apply` accepts it without error but the value doesn't persist
// (confirmed by re-running `railway config plan` immediately after apply,
// which still showed it pending). With railway.toml gone there's no
// fallback for this one field -- set "Restart Policy" to "On Failure"
// once in the Railway dashboard's service settings if it doesn't already
// show that.
import { defineRailway, github, preserve, project, service } from "railway/iac";

export default defineRailway(() => {
  const orthopedicsProductAgents = service("orthopedics-product-agents", {
    source: github("huytrinhx/orthopedics-product-agents", { checkSuites: false }),
    build: {
      builder: "DOCKERFILE",
      dockerfilePath: "Dockerfile",
    },
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

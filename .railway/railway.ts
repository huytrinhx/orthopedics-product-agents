// Infrastructure as Code replacement for railway.toml (Config as Code is
// deprecated -- existing railway.toml/railway.json keep working only until
// 2026-12-01, see https://docs.railway.com/infrastructure-as-code). This
// file is now the ONLY deploy config -- railway.toml has been deleted, its
// fields fully mirrored here.
//
// env below is a complete list of this service's variables, each wrapped
// in preserve() so real secret values never land in git -- add new ones
// here (as preserve() if already set via the dashboard) when the app gains
// one, or `railway config apply` will try to delete it from the service.
//
// Do NOT regenerate this file with `railway config migrate`: as of CLI
// v5.43.1 its output silently drops restartPolicyType and preDeployCommand
// entirely. Hand-edit instead, and always run `railway config plan` before
// `apply` to review the diff.
//
// OPEN ISSUES (as of 2026-09-06):
//   - The pre-deploy step (preDeploy below) fails silently on every
//     Railway deploy with no captured logs, cause not yet identified.
//     ACTION: after any deploy that adds a migration, run
//     `railway run .venv/bin/alembic upgrade head` from backend/ by hand
//     (idempotent, safe to rerun) -- see README's "Deploying to Railway".
//     If this recurs, escalate to Railway support with deployment IDs;
//     it isn't reproducible locally (the command runs fine standalone).
//   - restartPolicyType below doesn't reliably persist via `railway config
//     apply`. ACTION: confirm "Restart Policy" reads On Failure in the
//     Railway dashboard's service settings directly.
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
    // deploy.preDeployCommand only accepts a single-element array -- keep
    // this as one plain string, not an argv array (["sh","-c",...] fails
    // schema validation: "expected array to have <=1 items"). See the
    // OPEN ISSUES note above -- this currently fails silently on deploy.
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

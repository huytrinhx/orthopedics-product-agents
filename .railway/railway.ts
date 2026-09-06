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
// Three real bugs were found and fixed while migrating off railway.toml
// (against the live ortho-mate project, via `railway config plan`/`apply`,
// each verified rather than guessed):
//   1. An early draft carried `start: ""` over verbatim from `railway
//      config pull`'s baseline snapshot. Applying a literal empty string
//      as the start command (rather than leaving it unset) made every
//      deploy fail instantly with zero build/deploy/http logs -- Railway
//      tried to run an empty command instead of the Dockerfile's CMD.
//      Fixed by omitting `start` entirely (railway.toml never set one
//      either).
//   2. `.railway/README.md`'s own generated text warns "Services already
//      managed by railway.json must be migrated before .railway/railway.ts
//      can manage them," and this service never went through that formal
//      step (`railway config migrate --apply`, which mutates the live
//      project and was correctly declined by this session's own safety
//      classifier both times it was attempted). railway.toml was deleted
//      instead, since every field it set is already mirrored here.
// PRE-DEPLOY FAILURE INVESTIGATION (2026-09-06) -- unresolved, root cause
// narrowed but not fixed. Railway support's own diagnosis: "the pre-deploy
// command... fails within seconds and produces no captured log output."
// What was actually checked, in order:
//   - Theory: `cd` is a shell builtin, not a real executable, and
//     deploy.preDeployCommand's single array element gets exec'd directly
//     with no shell -- `docker run --entrypoint "cd backend && alembic
//     upgrade head" <image>` reproduces an instant, log-free failure
//     locally that superficially matches. Attempted fix: an explicit
//     3-element argv array, `["sh", "-c", "<command>"]`. This never
//     actually applied -- `railway config apply --json` revealed
//     deploy.preDeployCommand's schema caps the array at <=1 item
//     ("Too big: expected array to have <=1 items"), so every "successful"
//     apply of that shape was silently rejected server-side and the field
//     stayed at its old value the whole time. Reverted to the single-string
//     form below, which is schema-valid -- meaning the shell-exec theory,
//     while plausible, is NOT confirmed, since a single string is what
//     Railway's own schema expects (implying it likely does get some form
//     of shell handling internally; the local Docker repro used a
//     different exec path than Railway's own and may not be a faithful
//     simulation of it after all).
//   - Directly tested whether the command and credentials work at all:
//     `railway run .venv/bin/alembic upgrade head` (from backend/, which
//     injects this project's real production env vars into a LOCAL
//     process -- safe, since running this migration is the intended,
//     expected behavior, not a novel action) ran cleanly against the real
//     production Supabase host (aws-0-us-east-2.pooler.supabase.com) and
//     applied all 11 migrations from scratch. This was a significant,
//     unexpected finding: the production database had NO schema at all
//     before this -- meaning no prior deploy's pre-deploy step, under
//     either the old releaseCommand or the new preDeployCommand, had ever
//     successfully migrated it. `/health` (this file's own healthcheck
//     path) is a static `{"status": "ok"}` with no DB check, so its 200
//     response never actually proved otherwise.
//   - This rules out the command, credentials, and network path *from
//     outside Railway's infrastructure* as the cause. What remains
//     unconfirmed is whether Railway's own pre-deploy execution
//     environment can reach Supabase at all, or fails for some other
//     platform-side reason specific to their (still-new) IaC deploy path.
//     That's not diagnosable from here -- next step is genuinely what
//     Railway's own error message suggests: their support, or the
//     dashboard's deployment detail view, which may show more than the
//     CLI's `railway logs` does.
//
// restartPolicyType below is also a known, separate gap as of CLI
// v5.43.1: `railway config apply` accepts it without error but the value
// doesn't persist (confirmed by re-running `railway config plan`
// immediately after apply, twice). With railway.toml gone there's no
// fallback for this one field -- set "Restart Policy" to "On Failure"
// once in the Railway dashboard's service settings if it doesn't already
// show that, once deploys are working again.
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
    // A 3-element argv array (["sh","-c","<cmd>"]) was tried here as a fix
    // for a shell-exec theory that turned out to be wrong -- `railway
    // config apply --json` revealed the REAL reason it never took effect:
    // deploy.preDeployCommand's schema caps the array at <=1 item
    // ("Too big: expected array to have <=1 items"), so that attempt was
    // silently rejected by validation the whole time, never actually
    // applied. Reverted to the single-string form, which is schema-valid
    // and was already what's been failing. See the file-level comment for
    // where the investigation landed instead.
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

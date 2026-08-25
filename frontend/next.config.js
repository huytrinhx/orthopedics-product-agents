/** @type {import('next').NextConfig} */
// Static export so `next build` can be served as plain files by the
// backend's single Railway container (see root Dockerfile) — no Node
// server for the frontend in production, no CORS between two hosts.
const nextConfig = {
  output: "export",
};

module.exports = nextConfig;

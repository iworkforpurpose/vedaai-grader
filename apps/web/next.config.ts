import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // Page images and the SSE progress stream are served by the FastAPI worker,
  // not by Next. Uploads go direct to object storage via a presigned URL,
  // because Vercel caps a function request body at 4.5 MB and a scanned answer
  // sheet routinely exceeds that.
  env: {
    NEXT_PUBLIC_API_BASE: process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000",
  },
  typescript: { ignoreBuildErrors: false },
  eslint: { ignoreDuringBuilds: true },
};

export default config;

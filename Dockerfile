# One image, both processes, one origin.
#
# Next serves the browser and proxies /api to the FastAPI worker on loopback, so
# there is no CORS to configure and no second hostname to keep in sync. It also
# means one Fargate task rather than two services, which matters for a reason
# beyond cost: the submission store is in memory, so a second task would answer
# requests about submissions it has never heard of.

# ── Stage 1: Python dependencies, and the generated TypeScript contracts ─────
#
# Codegen lives here rather than in the web stage because the TypeScript types are
# emitted from the pydantic models by a Python script, and the generated file is
# not committed — it is build output, and a stale committed copy is exactly the
# drift the generator exists to prevent.
FROM python:3.12-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /repo
COPY apps/api/pyproject.toml apps/api/uv.lock apps/api/
COPY packages/contracts/ packages/contracts/

# `aws` brings Textract and S3, `grading` brings the marking client. The local OCR
# extra is deliberately absent: 600 MB of model weights for a recognizer the
# deployed service does not use.
RUN cd apps/api && uv sync --frozen --extra aws --extra grading --no-dev

# Emitted with the API's own environment, which already has pydantic, and verified
# by the generator itself against tsc.
RUN cd packages/contracts && /repo/apps/api/.venv/bin/python scripts/gen_types.py


# ── Stage 2: the frontend ────────────────────────────────────────────────────
FROM node:20-slim AS web

RUN corepack enable
WORKDIR /repo

# Manifests first, so editing application code does not re-resolve the graph.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml turbo.json ./
COPY apps/web/package.json apps/web/
COPY apps/api/package.json apps/api/
COPY packages/contracts/package.json packages/contracts/
COPY packages/evals/package.json packages/evals/
RUN pnpm install --frozen-lockfile

COPY packages/contracts/ packages/contracts/
COPY apps/web/ apps/web/
COPY --from=deps /repo/packages/contracts/dist/ packages/contracts/dist/

RUN pnpm --filter @vedaai/web build


# ── Stage 3: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# libgl and libglib are OpenCV's runtime dependencies. Their absence fails at
# import rather than at install, which is a confusing way to find out.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      nodejs \
      supervisor \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# The same absolute path as the build stage, deliberately.
#
# A virtualenv is not relocatable: every script in bin/ carries a shebang naming
# the interpreter by absolute path, and the editable install of the contracts
# package records its source directory the same way. Copying the venv to a
# different prefix produced exactly one symptom — "couldn't exec uvicorn: ENOENT"
# — which says nothing about shebangs at all.
WORKDIR /repo

COPY --from=deps /repo/apps/api/.venv /repo/apps/api/.venv
COPY --from=deps /repo/packages/contracts /repo/packages/contracts
COPY apps/api/src /repo/apps/api/src
COPY apps/api/pyproject.toml /repo/apps/api/

# The standalone output carries its own traced node_modules, so nothing else from
# the JS workspace has to come along.
COPY --from=web /repo/apps/web/.next/standalone /repo/web
COPY --from=web /repo/apps/web/.next/static /repo/web/apps/web/.next/static
# `public/` as well, which the standalone output deliberately leaves out — Next's
# file tracing only follows imports, and nothing imports a static asset.
#
# Missing it does not fail the build or the health check. The site serves, and
# every image on it 404s: the logo, the teacher illustration, the school crest and
# the avatar were all blank on the first deploy, with only "The requested resource
# isn't a valid image ... received null" in the log to say so.
COPY --from=web /repo/apps/web/public /repo/web/apps/web/public

COPY deploy/supervisord.conf /etc/supervisor/conf.d/grader.conf

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/repo/apps/api/.venv/bin:${PATH}" \
    PYTHONPATH=/repo/apps/api/src \
    NODE_ENV=production \
    PORT=8080 \
    HOSTNAME=0.0.0.0 \
    INTERNAL_API_BASE=http://127.0.0.1:8000 \
    PAGE_STORE_ROOT=/tmp/pagestore

# Only the browser-facing port is published. The worker stays on loopback, which
# makes "the API is not reachable from outside" true by construction rather than
# by a security group rule.
EXPOSE 8080

# Checks the worker, not just the web server. A container serving pages while the
# pipeline is dead is worse than one plainly down, because a load balancer keeps
# sending it work.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf", "-n"]

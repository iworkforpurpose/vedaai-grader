#!/usr/bin/env bash
# Build, push, and roll out the grader on ECS Fargate.
#
# Written to be read before it is run. Every step is idempotent, so re-running
# after a failure resumes rather than duplicating, and each resource is created
# only if it is missing.
#
#   deploy/deploy.sh bootstrap    # once: ECR, S3, IAM, cluster, ALB, service
#   deploy/deploy.sh release      # every time after: build, push, restart
#
# Requires the AWS CLI, Docker, and credentials with permission to create these
# resources. Reads AWS_REGION and S3_PAGE_BUCKET from .env if present.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a || true

REGION="${AWS_REGION:-ap-south-1}"
APP="${APP_NAME:-vedaai-grader}"
BUCKET="${S3_PAGE_BUCKET:-}"
CPU="${TASK_CPU:-512}"
MEMORY="${TASK_MEMORY:-1024}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REPO="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${APP}"

# The image tag, and with it the answer to "what is actually running?".
#
# Everything used to be `:latest`, which meant a running task named a tag that had
# been overwritten several times and could not be traced to anything. Worse, the
# build context is the working directory rather than a commit — `docker build .`
# reads the filesystem — so an uncommitted edit would deploy with nothing recording
# that it had.
#
# The tag is the short commit, suffixed `-dirty` when the tree does not match it.
# A dirty tag is deliberately ugly: seeing it in a task definition is the point.
if [ -z "${IMAGE_TAG:-}" ]; then
  IMAGE_TAG="$(git rev-parse --short HEAD 2>/dev/null || echo notgit)"
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    IMAGE_TAG="${IMAGE_TAG}-dirty"
    echo "WARNING: working tree has uncommitted changes; tagging ${IMAGE_TAG}" >&2
  fi
fi

# Progress goes to stderr, not stdout.
#
# It was stdout, and one function's stdout is captured as a value —
# `arn="$(register_task)"`. So the ARN arrived with the progress banner glued to
# the front of it, `update-service` was handed a malformed task definition, and it
# failed with "Invalid revision number" while the surrounding log read as a normal
# successful release. Progress is not data.
say() { printf '\n== %s\n' "$*" >&2; }

# ── build and push ───────────────────────────────────────────────────────────
release_image() {
  say "building ${APP} as ${IMAGE_TAG}"
  # linux/amd64 explicitly: Fargate runs x86 unless the task says otherwise, and
  # an image built on an Apple laptop is arm64 by default. The failure is an
  # "exec format error" in the task logs, long after the push looked fine.
  docker build --platform linux/amd64 -t "${APP}:${IMAGE_TAG}" .

  say "pushing to ${REPO}"
  aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

  # Both tags. The commit tag is what the task definition names, so a running task
  # can be traced to a commit; `latest` stays as a human-facing pointer to whatever
  # went out last.
  docker tag "${APP}:${IMAGE_TAG}" "${REPO}:${IMAGE_TAG}"
  docker tag "${APP}:${IMAGE_TAG}" "${REPO}:latest"
  docker push "${REPO}:${IMAGE_TAG}"
  docker push "${REPO}:latest"
}

# ── bootstrap ────────────────────────────────────────────────────────────────
bootstrap() {
  say "ECR repository"
  aws ecr describe-repositories --repository-names "${APP}" --region "${REGION}" >/dev/null 2>&1 \
    || aws ecr create-repository --repository-name "${APP}" --region "${REGION}" \
         --image-scanning-configuration scanOnPush=true >/dev/null

  if [ -z "${BUCKET}" ]; then
    BUCKET="${APP}-pages-${ACCOUNT}"
    echo "S3_PAGE_BUCKET was empty; using ${BUCKET}"
  fi

  say "S3 bucket ${BUCKET}"
  if ! aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
    aws s3api create-bucket --bucket "${BUCKET}" --region "${REGION}" \
      --create-bucket-configuration "LocationConstraint=${REGION}" >/dev/null
  fi
  # These are student answer scripts. Never public, and expired on a timer,
  # because they are fully regenerable from the upload.
  aws s3api put-public-access-block --bucket "${BUCKET}" \
    --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket "${BUCKET}" \
    --server-side-encryption-configuration \
      '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-bucket-lifecycle-configuration --bucket "${BUCKET}" \
    --lifecycle-configuration "file://deploy/bucket-lifecycle.json"

  # CORS, so the browser may PUT its own upload.
  #
  # Without this the presigned PUT fails a preflight and the browser reports it as a
  # network error with no detail — which reads as broken upload code rather than a
  # missing bucket rule. Scoped to the origins actually serving the app; a wildcard
  # would let any page on the internet upload into this bucket.
  local cors_origins="${WEB_ORIGINS:-*}"
  python3 - "${cors_origins}" > /tmp/${APP}-cors.json <<'PYCORS'
import json, sys
origins = [o.strip() for o in sys.argv[1].split(",") if o.strip()] or ["*"]
print(json.dumps({"CORSRules": [{
    "AllowedMethods": ["PUT"],
    "AllowedOrigins": origins,
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000,
}]}))
PYCORS
  aws s3api put-bucket-cors --bucket "${BUCKET}" \
    --cors-configuration "file:///tmp/${APP}-cors.json"
  rm -f /tmp/${APP}-cors.json
  echo "  CORS allows PUT from: ${cors_origins}"

  say "IAM roles"
  local trust='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

  # Execution role: pulls the image, writes logs, and resolves the marking
  # credential the task definition references by ARN.
  aws iam get-role --role-name "${APP}-execution" >/dev/null 2>&1 || \
    aws iam create-role --role-name "${APP}-execution" \
      --assume-role-policy-document "${trust}" >/dev/null

  # Outside the create guard, not inside it.
  #
  # Attaching only on first creation means a role that already exists never picks
  # up a policy change — so adding a credential and re-running would appear to
  # succeed and then fail at task start. Attach is idempotent, so there is no
  # reason for it to be conditional.
  aws iam attach-role-policy --role-name "${APP}-execution" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

  # The managed policy is not enough on its own.
  #
  # AmazonECSTaskExecutionRolePolicy covers the image pull and the log stream and
  # nothing else. Reading a secret referenced from `secrets` in the task definition
  # is a permission AWS requires you to add by hand, and the agent — not the
  # application — is what reads it, so it belongs on the execution role rather than
  # the task role. Without it the task does not start with a permissions error on
  # the API call; it fails during initialisation, before the container runs, which
  # is a much less obvious thing to debug.
  #
  # Scoped to the exact ARNs configured. A wildcard here would let anything that
  # can start this task read every secret in the account.
  local secret_arns=()
  [ -n "${OPENAI_SECRET_ARN:-}" ] && secret_arns+=("\"${OPENAI_SECRET_ARN}\"")
  [ -n "${ANTHROPIC_SECRET_ARN:-}" ] && secret_arns+=("\"${ANTHROPIC_SECRET_ARN}\"")
  # The tailnet key is read by the agent before the sidecar starts, exactly like
  # the marking credential, so it belongs on the same role for the same reason.
  [ -n "${TAILSCALE_SECRET_ARN:-}" ] && secret_arns+=("\"${TAILSCALE_SECRET_ARN}\"")

  if [ ${#secret_arns[@]} -gt 0 ]; then
    local arn_list
    arn_list=$(IFS=,; echo "${secret_arns[*]}")
    sed "s|REPLACE_SECRET_ARNS|${arn_list}|" \
      deploy/execution-role-secrets-policy.json > /tmp/${APP}-exec-policy.json
    aws iam put-role-policy --role-name "${APP}-execution" \
      --policy-name "${APP}-marking-credential" \
      --policy-document "file:///tmp/${APP}-exec-policy.json"
    rm -f /tmp/${APP}-exec-policy.json
  else
    # No credential configured: drop any grant a previous run left behind, so
    # turning marking off actually removes the permission.
    aws iam delete-role-policy --role-name "${APP}-execution" \
      --policy-name "${APP}-marking-credential" 2>/dev/null || true
  fi

  # Task role: what the application itself may do. Textract and one S3 prefix,
  # and nothing else — this is the credential the container actually runs with,
  # which is why no key pair is ever shipped in the image.
  aws iam get-role --role-name "${APP}-task" >/dev/null 2>&1 || \
    aws iam create-role --role-name "${APP}-task" \
      --assume-role-policy-document "${trust}" >/dev/null

  sed "s/REPLACE_BUCKET/${BUCKET}/g" deploy/task-role-policy.json > /tmp/${APP}-policy.json
  aws iam put-role-policy --role-name "${APP}-task" \
    --policy-name "${APP}-app" --policy-document "file:///tmp/${APP}-policy.json"

  say "log group"
  aws logs create-log-group --log-group-name "/ecs/${APP}" --region "${REGION}" 2>/dev/null || true
  aws logs put-retention-policy --log-group-name "/ecs/${APP}" \
    --retention-in-days 14 --region "${REGION}" >/dev/null

  say "cluster"
  aws ecs describe-clusters --clusters "${APP}" --region "${REGION}" \
    --query 'clusters[0].status' --output text 2>/dev/null | grep -q ACTIVE \
    || aws ecs create-cluster --cluster-name "${APP}" --region "${REGION}" >/dev/null

  echo
  echo "Bootstrapped. Networking is the one part left, and it is left on purpose:"
  echo "a VPC, subnets, a security group and a load balancer depend on what you"
  echo "already have, and guessing would either duplicate them or wire the service"
  echo "into the wrong one. See deploy/README.md for the four commands, or create"
  echo "the service in the console and point it at the task definition below."
  echo
  echo "  bucket        ${BUCKET}"
  echo "  image         ${REPO}:${IMAGE_TAG}"
  echo "  task role     arn:aws:iam::${ACCOUNT}:role/${APP}-task"
  echo "  exec role     arn:aws:iam::${ACCOUNT}:role/${APP}-execution"
}

# ── task definition ──────────────────────────────────────────────────────────
register_task() {
  [ -n "${BUCKET}" ] || BUCKET="${APP}-pages-${ACCOUNT}"
  say "registering task definition"

  # Referenced by ARN rather than by value, so the key is visible neither in the
  # task definition nor to anyone who can describe the task. Either provider, or
  # neither — marking degrades to a rubric without one.
  local secrets=""
  local entries=()
  [ -n "${OPENAI_SECRET_ARN:-}" ] && entries+=("$(printf '{"name":"OPENAI_API_KEY","valueFrom":"%s"}' "${OPENAI_SECRET_ARN}")")
  [ -n "${ANTHROPIC_SECRET_ARN:-}" ] && entries+=("$(printf '{"name":"ANTHROPIC_API_KEY","valueFrom":"%s"}' "${ANTHROPIC_SECRET_ARN}")")
  if [ ${#entries[@]} -gt 0 ]; then
    secrets=$(IFS=,; echo "${entries[*]}")
  fi

  # The tunnel sidecar, when a tailnet key is configured.
  #
  # Appended to the container list, so a deployment without a key is byte-for-byte
  # the task it was before. The tunnel is additive, not a different way of serving.
  local tunnel=""
  if [ -n "${TAILSCALE_SECRET_ARN:-}" ]; then
    # The serve config travels as base64, and that is not fussiness.
    #
    # It has to reach the container with `${TS_CERT_DOMAIN}` intact — the sidecar
    # substitutes it once the node learns its own name, which is the first moment
    # the hostname exists. Embedding the JSON in this script put that placeholder
    # inside a shell expansion, and `set -u` correctly killed the run over an
    # unbound variable. Base64 contains no `$` and no quotes, so nothing between
    # here and the container can interpret it.
    local serve_b64
    serve_b64="$(base64 < deploy/tailscale-serve.json | tr -d '\n')"

    tunnel=$(cat <<TUNNEL
    ,{
      "name": "tunnel",
      "image": "tailscale/tailscale:stable",
      "essential": true,
      "entryPoint": ["/bin/sh", "-c"],
      "command": ["echo \$SERVE_B64 | base64 -d > /tmp/serve.json && exec /usr/local/bin/containerboot"],
      "environment": [
        { "name": "TS_HOSTNAME", "value": "${APP}" },
        { "name": "TS_SERVE_CONFIG", "value": "/tmp/serve.json" },
        { "name": "TS_STATE_DIR", "value": "/tmp/tsstate" },
        { "name": "TS_USERSPACE", "value": "true" },
        { "name": "TS_ENABLE_HEALTH_CHECK", "value": "true" },
        { "name": "SERVE_B64", "value": "${serve_b64}" }
      ],
      "secrets": [
        { "name": "TS_AUTHKEY", "valueFrom": "${TAILSCALE_SECRET_ARN}" }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${APP}",
          "awslogs-region": "${REGION}",
          "awslogs-stream-prefix": "tunnel"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "wget -q -O- http://127.0.0.1:9002/healthz || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 90
      }
    }
TUNNEL
)
  fi

  cat > /tmp/${APP}-task.json <<JSON
{
  "family": "${APP}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "${CPU}",
  "memory": "${MEMORY}",
  "runtimePlatform": { "cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX" },
  "executionRoleArn": "arn:aws:iam::${ACCOUNT}:role/${APP}-execution",
  "taskRoleArn": "arn:aws:iam::${ACCOUNT}:role/${APP}-task",
  "containerDefinitions": [
    {
      "name": "grader",
      "image": "${REPO}:${IMAGE_TAG}",
      "essential": true,
      "portMappings": [{ "containerPort": 8080, "protocol": "tcp" }],
      "environment": [
        { "name": "AWS_REGION", "value": "${REGION}" },
        { "name": "OCR_ENGINE", "value": "textract" },
        { "name": "S3_PAGE_BUCKET", "value": "${BUCKET}" },
        { "name": "S3_PAGE_PREFIX", "value": "pages/" },
        { "name": "WEB_ORIGINS", "value": "${WEB_ORIGINS:-}" },
        { "name": "GRADER_PROVIDER", "value": "${GRADER_PROVIDER:-}" },
        { "name": "GRADER_MODEL", "value": "${GRADER_MODEL:-}" }
      ],
      "secrets": [${secrets}],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${APP}",
          "awslogs-region": "${REGION}",
          "awslogs-stream-prefix": "task"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c \\"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)\\""],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 30
      }
    }${tunnel}
  ]
}
JSON

  aws ecs register-task-definition --cli-input-json "file:///tmp/${APP}-task.json" \
    --region "${REGION}" --query 'taskDefinition.taskDefinitionArn' --output text
}

release() {
  release_image
  local arn
  arn="$(register_task)"
  echo "registered ${arn}"

  if aws ecs describe-services --cluster "${APP}" --services "${APP}" --region "${REGION}" \
       --query 'services[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
    say "rolling out"
    aws ecs update-service --cluster "${APP}" --service "${APP}" \
      --task-definition "${arn}" --force-new-deployment --region "${REGION}" \
      --query 'service.deployments[0].id' --output text

    # Read back what the service is actually pinned to.
    #
    # Because the failure this guards against was silent from the outside. The
    # image pushed, the revision registered, the log said "rolling out" — and the
    # service kept serving the previous revision, so two fixes looked like they
    # had failed on the live task when they had never been deployed at all.
    local live
    live="$(aws ecs describe-services --cluster "${APP}" --services "${APP}" \
              --region "${REGION}" --query 'services[0].taskDefinition' --output text)"
    if [ "${live}" != "${arn}" ]; then
      echo "ROLLOUT DID NOT TAKE: service is on ${live}, expected ${arn}" >&2
      return 1
    fi
    echo "service now on ${live}"
    echo "watch it with: aws ecs wait services-stable --cluster ${APP} --services ${APP} --region ${REGION}"
  else
    echo
    echo "No service named ${APP} yet. Create it once against ${arn} —"
    echo "see deploy/README.md — and later releases will roll out automatically."
  fi
}

case "${1:-}" in
  bootstrap) bootstrap ;;
  release)   release ;;
  image)     release_image ;;
  task)      register_task ;;
  *) echo "usage: $0 {bootstrap|release|image|task}" >&2; exit 2 ;;
esac

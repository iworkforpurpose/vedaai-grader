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

# Resolved here rather than inside each command that needs it. The name was derived
# in three separate places, so a command run on its own could act on the wrong
# bucket depending on which one had run first.
[ -n "${BUCKET}" ] || BUCKET="${APP}-pages-${ACCOUNT}"
TABLE="${SUBMISSIONS_TABLE:-${APP}-submissions}"

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
  # The full commit, matching what CI passes.
  #
  # This was the short form, and CI passes `github.sha` in full — so a task
  # definition registered from a laptop named a tag only a local build would have
  # pushed, and the task failed with CannotPullContainerError. Two conventions for
  # the same identifier is one too many.
  IMAGE_TAG="$(git rev-parse HEAD 2>/dev/null || echo notgit)"
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    IMAGE_TAG="${IMAGE_TAG}-dirty"
    echo "WARNING: working tree has uncommitted changes; tagging ${IMAGE_TAG}" >&2
  fi
fi

# Refuse to register a task definition for an image that was never pushed.
#
# A `-dirty` tag is by construction not in the registry: it names a tree that only
# exists on one laptop. Registering it produces a revision the service cannot pull,
# and the failure arrives minutes later as a placement error rather than here. The
# warning above was not enough — I ignored it and did exactly this.
require_pushed_image() {
  case "${IMAGE_TAG}" in
    *-dirty)
      echo "REFUSING: ${IMAGE_TAG} names an uncommitted tree, so no such image exists." >&2
      echo "  Commit and let CI build it, or run 'deploy/deploy.sh release' to build now." >&2
      return 1
      ;;
  esac
  aws ecr describe-images --repository-name "${APP}" --image-ids "imageTag=${IMAGE_TAG}" \
    --region "${REGION}" >/dev/null 2>&1 || {
      echo "REFUSING: ${APP}:${IMAGE_TAG} is not in the registry." >&2
      echo "  Run 'deploy/deploy.sh release' to build and push it first." >&2
      return 1
    }
}

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
# CORS, so the browser may POST its own upload.
#
# Without this the presigned POST fails a preflight and the browser reports it as a
# network error with no detail — which reads as broken upload code rather than a
# missing bucket rule.
#
# Its own command as well as part of bootstrap, because the allowed origin is not
# known until the gateway exists: the bucket is created first, and re-running the
# whole of bootstrap to change one rule would touch roles and lifecycle policies
# that are already correct. A wildcard here would let any page on the internet
# upload into a bucket holding student work, so it is worth being able to narrow it
# on its own.
# The origin the deployed app is served from.
#
# Read from the gateway rather than written down. A copy in a config file is a
# second source of truth for something AWS already knows, and the obvious place to
# put it — .env — is the file local development reads, where a non-empty value
# switches off the loopback allowance the local browser needs. Configuring
# production would have broken reassignment on a developer's machine.
app_origin() {
  if [ -n "${WEB_ORIGINS:-}" ]; then
    echo "${WEB_ORIGINS}"
    return 0
  fi
  bash deploy/gateway.sh url 2>/dev/null || true
}

# The policy CI deploys under.
#
# A command rather than a documented sequence of console steps, because this policy
# has drifted from the file twice: once holding a wildcard that let it roll out an
# unrelated production service, and once still naming a secret the execution role
# had stopped being allowed to read, which took the site down. If the file is the
# source of truth then applying it has to be one command.
ci_role() {
  local role="${APP}-github-deploy"
  local rendered="/tmp/${APP}-ci-policy.json"
  sed -e "s/REPLACE_REGION/${REGION}/g" -e "s/REPLACE_ACCOUNT/${ACCOUNT}/g" \
    deploy/github-deploy-policy.json > "${rendered}"
  python3 -c "import json,sys; json.load(open(sys.argv[1]))" "${rendered}"
  case "$(cat "${rendered}")" in
    *REPLACE_*) echo "REFUSING: a placeholder survived substitution" >&2; return 1 ;;
  esac
  aws iam put-role-policy --role-name "${role}" \
    --policy-name deploy --policy-document "file://${rendered}"
  rm -f "${rendered}"
  echo "  ${role}: policy applied from deploy/github-deploy-policy.json"
}

# Which credential ARNs the running task definition asks the agent to fetch.
#
# ARNs only — the names of the boxes, never what is in them. Read rather than
# assumed because these are exactly what the execution role must be allowed to
# read for a task to start, so the task definition is the authority on the grant.
# An empty result on a first-ever bootstrap is correct: no task definition exists.
referenced_credentials() {
  aws ecs describe-task-definition --task-definition "${APP}" \
    --region "${REGION}" --output json 2>/dev/null | python3 -c '
import json, sys
try:
    definition = json.load(sys.stdin)["taskDefinition"]
except Exception:
    raise SystemExit(0)
for container in definition.get("containerDefinitions") or []:
    for reference in container.get("secrets") or []:
        print(reference["valueFrom"])
' || true
}

# Let the agent read the marking credential the task definition names.
#
# AmazonECSTaskExecutionRolePolicy covers the image pull and the log stream and
# nothing else. This permission AWS requires you to add by hand, and the agent —
# not the application — is what reads it, so it belongs on the execution role. Get
# it wrong and the task does not fail with a permissions error on an API call; it
# fails during initialisation, before any container starts, so nothing appears in
# the application log at all.
#
# Its own command as well as part of bootstrap, because restoring this one grant
# should not mean re-running everything else.
exec_role_secrets() {
  # Sourced from the registered task definition as well as the environment, and
  # that is a fix rather than a refinement: this block caused an outage.
  #
  # The ARNs used to come only from the local environment, so running bootstrap
  # from a machine whose .env does not carry them took the "nothing configured"
  # branch and deleted the grant — while the running task definition still
  # referenced the secret. Every task then failed to initialise. The environment
  # may add to what must be granted; it may not be read as evidence that nothing
  # needs granting.
  local arns=() arn
  for arn in "${GROQ_SECRET_ARN:-}" "${CEREBRAS_SECRET_ARN:-}" "${GEMINI_SECRET_ARN:-}" "${OPENAI_SECRET_ARN:-}" "${ANTHROPIC_SECRET_ARN:-}" $(referenced_credentials); do
    [ -n "${arn}" ] || continue
    case " ${arns[*]-} " in *"\"${arn}\""*) continue ;; esac
    arns+=("\"${arn}\"")
  done

  if [ ${#arns[@]} -gt 0 ]; then
    # Scoped to those exact ARNs. A wildcard would let anything able to start this
    # task read every secret in the account.
    local arn_list
    arn_list=$(IFS=,; echo "${arns[*]}")
    sed "s|REPLACE_SECRET_ARNS|${arn_list}|" \
      deploy/execution-role-secrets-policy.json > "/tmp/${APP}-exec-policy.json"
    aws iam put-role-policy --role-name "${APP}-execution" \
      --policy-name "${APP}-marking-credential" \
      --policy-document "file:///tmp/${APP}-exec-policy.json"
    rm -f "/tmp/${APP}-exec-policy.json"
    echo "  execution role may read ${#arns[@]} credential(s)"
  else
    # Nothing references one — not the environment, not the task definition — so a
    # grant from an earlier run is genuinely stale, and dropping it is how turning
    # marking off removes the permission.
    aws iam delete-role-policy --role-name "${APP}-execution" \
      --policy-name "${APP}-marking-credential" 2>/dev/null || true
    echo "  no credential referenced; grant removed if present"
  fi
}

bucket_cors() {
  local origins
  origins="$(app_origin)"
  case "${origins}" in https://*) : ;; *) origins="" ;; esac
  if [ -z "${origins}" ]; then
    echo "REFUSING: no app origin, and a wildcard would let any page on the" >&2
    echo "  internet POST into ${BUCKET}. Run 'deploy/gateway.sh create', or set" >&2
    echo "  WEB_ORIGINS to the origin serving the app." >&2
    return 1
  fi
  python3 - "${origins}" > "/tmp/${APP}-cors.json" <<'PYCORS'
import json, sys
origins = [o.strip() for o in sys.argv[1].split(",") if o.strip()]
assert origins, "no origins"
print(json.dumps({"CORSRules": [{
    "AllowedMethods": ["POST"],
    "AllowedOrigins": origins,
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000,
}]}))
PYCORS
  aws s3api put-bucket-cors --bucket "${BUCKET}" \
    --cors-configuration "file:///tmp/${APP}-cors.json"
  rm -f "/tmp/${APP}-cors.json"
  echo "  CORS allows POST from: ${origins}"
}

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

  bucket_cors

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

  exec_role_secrets

  # Task role: what the application itself may do. Textract and one S3 prefix,
  # and nothing else — this is the credential the container actually runs with,
  # which is why no key pair is ever shipped in the image.
  aws iam get-role --role-name "${APP}-task" >/dev/null 2>&1 || \
    aws iam create-role --role-name "${APP}-task" \
      --assume-role-policy-document "${trust}" >/dev/null

  sed -e "s/REPLACE_BUCKET/${BUCKET}/g" -e "s/REPLACE_TABLE/${TABLE}/g" \
      -e "s/REPLACE_REGION/${REGION}/g" -e "s/REPLACE_ACCOUNT/${ACCOUNT}/g" \
      deploy/task-role-policy.json > /tmp/${APP}-policy.json
  case "$(cat /tmp/${APP}-policy.json)" in
    *REPLACE_*) echo "REFUSING: a placeholder survived substitution" >&2; return 1 ;;
  esac
  aws iam put-role-policy --role-name "${APP}-task" \
    --policy-name "${APP}-app" --policy-document "file:///tmp/${APP}-policy.json"

  say "submissions table ${TABLE}"
  # On-demand billing, not provisioned. This is a test deployment whose traffic is
  # a few teachers a day; provisioned capacity would mean paying for a baseline
  # nobody uses and still throttling on a burst.
  if aws dynamodb describe-table --table-name "${TABLE}" --region "${REGION}" \
       >/dev/null 2>&1; then
    echo "  exists"
  else
    aws dynamodb create-table --table-name "${TABLE}" --region "${REGION}" \
      --attribute-definitions AttributeName=pk,AttributeType=S \
      --key-schema AttributeName=pk,KeyType=HASH \
      --billing-mode PAY_PER_REQUEST >/dev/null
    aws dynamodb wait table-exists --table-name "${TABLE}" --region "${REGION}"
    echo "  created"
  fi

  # Expiry is the table's job, not the application's. Nothing in the service
  # deletes a submission, so without this the records accumulate for ever — and a
  # sweep written in the app would be one more thing to run and to get wrong.
  aws dynamodb update-time-to-live --table-name "${TABLE}" --region "${REGION}" \
    --time-to-live-specification "Enabled=true,AttributeName=expires_at" \
    >/dev/null 2>&1 || echo "  TTL already set"

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
  # Same source as the bucket rule, so the two cannot disagree.
  local APP_ORIGIN
  APP_ORIGIN="$(app_origin)"
  case "${APP_ORIGIN}" in https://*) : ;; *) APP_ORIGIN="" ;; esac
  [ -n "${BUCKET}" ] || BUCKET="${APP}-pages-${ACCOUNT}"
  require_pushed_image || return 1
  say "registering task definition"

  # Referenced by ARN rather than by value, so the key is visible neither in the
  # task definition nor to anyone who can describe the task. Either provider, or
  # neither — marking degrades to a rubric without one.
  local secrets=""
  local entries=()
  # Groq first, and OpenAI kept only so an existing deployment is not broken by
  # this change. The product now derives its check banks on an open-weight model
  # and answers them with a cross-encoder on the task itself, so a deployment
  # needs no OpenAI credential at all.
  [ -n "${GROQ_SECRET_ARN:-}" ] && entries+=("$(printf '{"name":"GROQ_API_KEY","valueFrom":"%s"}' "${GROQ_SECRET_ARN}")")
  # A second host is a second daily allowance for the same model, so it is worth
  # wiring even though nothing breaks without it: the chain simply skips a host
  # it has no key for.
  [ -n "${CEREBRAS_SECRET_ARN:-}" ] && entries+=("$(printf '{"name":"CEREBRAS_API_KEY","valueFrom":"%s"}' "${CEREBRAS_SECRET_ARN}")")
  [ -n "${GEMINI_SECRET_ARN:-}" ] && entries+=("$(printf '{"name":"GEMINI_API_KEY","valueFrom":"%s"}' "${GEMINI_SECRET_ARN}")")
  [ -n "${OPENAI_SECRET_ARN:-}" ] && entries+=("$(printf '{"name":"OPENAI_API_KEY","valueFrom":"%s"}' "${OPENAI_SECRET_ARN}")")
  [ -n "${ANTHROPIC_SECRET_ARN:-}" ] && entries+=("$(printf '{"name":"ANTHROPIC_API_KEY","valueFrom":"%s"}' "${ANTHROPIC_SECRET_ARN}")")
  if [ ${#entries[@]} -gt 0 ]; then
    secrets=$(IFS=,; echo "${entries[*]}")
  fi

  # ACCESS_CODE rides in the plain task environment rather than Secrets Manager.
  # It is a door code for a test deployment — it spends nothing and reaches
  # nothing on its own — and anybody who can read a task definition already has
  # the account. Injecting it as a secret would add a grant and a rotation path
  # for a string whose whole purpose is being handed to people.
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
        { "name": "SUBMISSIONS_TABLE", "value": "${TABLE}" },
        { "name": "WEB_ORIGINS", "value": "${APP_ORIGIN}" },
        { "name": "GRADER_PROVIDER", "value": "${GRADER_PROVIDER:-groq}" },
        { "name": "MARK_SAMPLES", "value": "${MARK_SAMPLES:-5}" },
        { "name": "GRADER_MODEL", "value": "${GRADER_MODEL:-}" },
        { "name": "ACCESS_CODE", "value": "${ACCESS_CODE:-}" },
        { "name": "RATE_LIMIT_INGEST_PER_HOUR", "value": "${RATE_LIMIT_INGEST_PER_HOUR:-30}" },
        { "name": "RATE_LIMIT_REMARK_PER_HOUR", "value": "${RATE_LIMIT_REMARK_PER_HOUR:-120}" }
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
    }
  ]
}
JSON

  aws ecs register-task-definition --cli-input-json "file:///tmp/${APP}-task.json" \
    --region "${REGION}" --query 'taskDefinition.taskDefinitionArn' --output text
}

# A deployment with no access code serves every stored script to anyone who finds
# the address, and does it silently — which is the one way a gate is worse than no
# gate, because it is believed. So this refuses rather than shipping an open
# origin, and says what to do about it.
#
# ALLOW_OPEN_ORIGIN=1 is the deliberate way past, for a deployment that is meant
# to be open. Typing it is the point.
require_access_code() {
  if [ -n "${ACCESS_CODE:-}" ] || [ "${ALLOW_OPEN_ORIGIN:-}" = "1" ]; then
    return 0
  fi
  cat >&2 <<'MSG'

  ACCESS_CODE is not set, so this would deploy an origin anyone can read.

  Every submission holds a real student's handwriting. Set a code and try again:

      # locally
      echo "ACCESS_CODE=$(openssl rand -hex 8)" >> .env

      # for CI, which is what actually deploys
      gh secret set ACCESS_CODE

  Or, if this deployment is meant to be public, say so explicitly:

      ALLOW_OPEN_ORIGIN=1 bash deploy/deploy.sh release

MSG
  exit 1
}

release() {
  require_access_code
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
  cors)      bucket_cors ;;
  ci-role)   ci_role ;;
  secrets)   exec_role_secrets ;;
  release)   release ;;
  image)     release_image ;;
  task)      register_task ;;
  *) echo "usage: $0 {bootstrap|cors|ci-role|secrets|release|image|task}" >&2; exit 2 ;;
esac

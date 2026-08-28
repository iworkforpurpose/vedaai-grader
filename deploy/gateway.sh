#!/usr/bin/env bash
# A stable HTTPS address in front of the task.
#
# The task's public IP changes every deploy — it moved five times in one afternoon —
# and it serves plain HTTP, so a reviewer got a link that expired and a browser
# warning. An HTTP API gives a fixed hostname with an AWS-managed certificate for
# about a dollar per million requests, with no load balancer and no domain.
#
# The conventional shape is CloudFront in front of an ALB. An ALB with a certificate
# is roughly 1,500 rupees a month, and this is a test deployment, so the gateway
# stands in. `deploy/README.md` records the trade-off; the code path is identical.
#
# Two of its quotas decided the design:
#
#   30-second integration timeout, not increasable. Fine only because ingest moved
#   to a background task — the upload now answers in about a second. It would have
#   made this impossible a day ago.
#
#   10 MB request body, not increasable, against documents this service accepts up
#   to 40 MB. That is why uploads go straight to object storage and never traverse
#   the gateway at all.
#
#   deploy/gateway.sh create   # once
#   deploy/gateway.sh point    # after each deploy, aims it at the current task
#   deploy/gateway.sh url      # print the stable address
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a || true

REGION="${AWS_REGION:-ap-south-1}"
APP="${APP_NAME:-vedaai-grader}"
PORT="${TASK_PORT:-8080}"

say() { printf '\n== %s\n' "$*" >&2; }

api_id() {
  aws apigatewayv2 get-apis --region "$REGION" \
    --query "Items[?Name=='${APP}'].ApiId | [0]" --output text 2>/dev/null
}

task_ip() {
  local task eni
  task="$(aws ecs list-tasks --cluster "$APP" --service-name "$APP" \
            --desired-status RUNNING --region "$REGION" \
            --query 'taskArns[0]' --output text)"
  [ "$task" = "None" ] || [ -z "$task" ] && { echo "no running task" >&2; return 1; }
  eni="$(aws ecs describe-tasks --cluster "$APP" --tasks "$task" --region "$REGION" \
          --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value | [0]" \
          --output text)"
  aws ec2 describe-network-interfaces --network-interface-ids "$eni" --region "$REGION" \
    --query 'NetworkInterfaces[0].Association.PublicIp' --output text
}

create() {
  local id
  id="$(api_id)"
  if [ "$id" != "None" ] && [ -n "$id" ]; then
    say "api already exists: ${id}"
  else
    say "creating the http api"
    # No CORS configuration on purpose: the browser only ever talks to this one
    # origin. Next serves the page and proxies /api to the worker beside it, so
    # there is no cross-origin request to permit — adding a policy would be
    # declaring a problem that does not exist.
    id="$(aws apigatewayv2 create-api --name "$APP" --protocol-type HTTP \
            --region "$REGION" --query ApiId --output text)"
    echo "  api: ${id}"
  fi

  local ip
  ip="$(task_ip)"
  say "integration -> http://${ip}:${PORT}"

  # ANY /{proxy+} and nothing else. The app owns its own routing; enumerating routes
  # here would mean a second place to update whenever a page is added.
  local integration
  integration="$(aws apigatewayv2 create-integration --api-id "$id" \
      --integration-type HTTP_PROXY --integration-method ANY \
      --integration-uri "http://${ip}:${PORT}/{proxy}" \
      --payload-format-version 1.0 --region "$REGION" \
      --query IntegrationId --output text)"

  aws apigatewayv2 create-route --api-id "$id" --route-key 'ANY /{proxy+}' \
    --target "integrations/${integration}" --region "$REGION" >/dev/null 2>&1 || true

  # The bare path too. `/{proxy+}` does not match an empty path, so without this the
  # home page 404s while every other page works — a confusing way to be broken.
  local root_integration
  root_integration="$(aws apigatewayv2 create-integration --api-id "$id" \
      --integration-type HTTP_PROXY --integration-method ANY \
      --integration-uri "http://${ip}:${PORT}/" \
      --payload-format-version 1.0 --region "$REGION" \
      --query IntegrationId --output text)"
  aws apigatewayv2 create-route --api-id "$id" --route-key 'ANY /' \
    --target "integrations/${root_integration}" --region "$REGION" >/dev/null 2>&1 || true

  aws apigatewayv2 create-stage --api-id "$id" --stage-name '$default' \
    --auto-deploy --region "$REGION" >/dev/null 2>&1 || true

  url
}

point() {
  local id ip
  id="$(api_id)"
  [ "$id" = "None" ] || [ -z "$id" ] && { echo "no api yet; run create" >&2; exit 1; }
  ip="$(task_ip)"
  say "aiming ${id} at http://${ip}:${PORT}"

  # Every integration is re-pointed, which is what makes the address stable rather
  # than stable-until-the-next-deploy. The task IP changes on each rollout, so this
  # runs from CI immediately after one.
  local ids
  ids="$(aws apigatewayv2 get-integrations --api-id "$id" --region "$REGION" \
          --query 'Items[].IntegrationId' --output text)"
  for i in $ids; do
    local uri
    uri="$(aws apigatewayv2 get-integration --api-id "$id" --integration-id "$i" \
            --region "$REGION" --query IntegrationUri --output text)"
    case "$uri" in
      */\{proxy\}) uri="http://${ip}:${PORT}/{proxy}" ;;
      *)           uri="http://${ip}:${PORT}/" ;;
    esac
    aws apigatewayv2 update-integration --api-id "$id" --integration-id "$i" \
      --integration-uri "$uri" --region "$REGION" --query IntegrationUri --output text
  done
  url
}

url() {
  local id
  id="$(api_id)"
  aws apigatewayv2 get-api --api-id "$id" --region "$REGION" \
    --query ApiEndpoint --output text
}

case "${1:-}" in
  create) create ;;
  point)  point ;;
  url)    url ;;
  *) echo "usage: $0 {create|point|url}" >&2; exit 2 ;;
esac

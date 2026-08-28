# Deploying the grader

One ECS Fargate task in `ap-south-1`, serving both the frontend and the API behind
one origin.

## Why it is shaped this way

**One task, not two services.** The submission store is in memory, which the brief
allowed. That decision has a deployment consequence: a second task would answer
requests about submissions it has never heard of. The same reason fixes uvicorn at
one worker. Going multi-task later means moving submissions to DynamoDB first —
nothing else in the design blocks it.

**One origin.** Next serves the browser and proxies `/api/*` to the worker on
loopback. No CORS, no second hostname, and no request-body cap on uploads. Only
port 8080 is published; the worker is unreachable from outside by construction
rather than by a security group rule.

**No long-lived keys in the image.** Textract and S3 authenticate through the task
role. The marking credential is referenced by ARN in the task definition and
injected by ECS at container start, so it appears neither in the image nor in this
repository.

## What it costs

| | |
|---|---|
| Fargate, 0.5 vCPU / 1 GB, always on | ~$18/month |
| Application Load Balancer | ~$17/month |
| Textract | $1.50 per 1,000 pages |
| S3 | pennies, and expired after 7 days |

Always-on because the job is interactive and 60–180s long; a cold start would land
in the middle of a teacher waiting. 1 GB rather than 2 because dropping the local
model dropped the memory with it.

## First time

```bash
cp .env.example .env
deploy/deploy.sh bootstrap
```

Creates the ECR repository, the S3 bucket (private, encrypted, 7-day expiry), both
IAM roles, the log group and the cluster. Idempotent — re-run it freely.

### What the two roles may do

| Role | Assumed by | Grants |
|---|---|---|
| `vedaai-grader-task` | the application | `textract:DetectDocumentText`, and read/write on one S3 prefix. Nothing else. This is the credential the container runs with, which is why no key pair is ever built into the image. |
| `vedaai-grader-execution` | the ECS agent, before the container starts | Image pull, log stream, and reading the marking credential by ARN. |

`textract:DetectDocumentText` is granted on `"*"` because it has to be: the action
takes no resource types and no condition keys, so there is nothing to scope it to.
`s3:ListBucket` is scoped by an `s3:prefix` condition rather than by resource,
because the bucket is the resource for a list and the prefix is the only thing that
can narrow it.

### What the operator running this needs

Not documented for completeness — documented because the obvious way to fix the
error you get without it is dangerous.

`deploy.sh register` calls `ecs register-task-definition` with both role ARNs, and
naming a role in a task definition requires `iam:PassRole` on it. When that fails,
the tempting fix is `iam:PassRole` on `"*"`. **Don't.** `iam:PassRole` on `"*"`
combined with the ability to register a task definition is privilege escalation to
every role in the account, including any administrator role — you write a task
definition naming that role and run it. Scope it:

```json
{
  "Effect": "Allow",
  "Action": "iam:PassRole",
  "Resource": [
    "arn:aws:iam::ACCOUNT:role/vedaai-grader-task",
    "arn:aws:iam::ACCOUNT:role/vedaai-grader-execution"
  ],
  "Condition": { "StringEquals": { "iam:PassedToService": "ecs-tasks.amazonaws.com" } }
}
```

`bootstrap` additionally needs `iam:CreateRole`, `iam:AttachRolePolicy` and
`iam:PutRolePolicy`. The last two can attach any policy to any role, so an identity
holding them can grant itself administrator. Treat `bootstrap` as a one-off run by
a human with elevated access, and give CI only the `register`/`deploy` subset with
the scoped `PassRole` above.

### One hardening deliberately not applied

A trust policy for a service principal is normally narrowed with `aws:SourceAccount`
or `aws:SourceArn` to prevent a confused deputy. Both roles here trust
`ecs-tasks.amazonaws.com` with no such condition, matching AWS's own published
example for ECS task roles. I did not add one: I could not confirm from the
documentation that ECS populates those keys when it assumes a task role, and a
trust-policy condition on a key the service does not set denies every assumption —
which surfaces as tasks that will not start, with no indication why. Worth
revisiting against the service authorization reference rather than by guesswork.

Networking is deliberately not automated. A VPC, subnets, a security group and a
load balancer depend on what the account already has, and a script that guesses
either duplicates them or wires the service into the wrong one. Four commands, or
the console:

```bash
REGION=ap-south-1
APP=vedaai-grader
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
        --query 'Vpcs[0].VpcId' --output text --region $REGION)
SUBNETS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC \
        --query 'Subnets[].SubnetId' --output text --region $REGION | tr '\t' ',')

# 1. Security groups: the load balancer is public, the task accepts traffic only
#    from the load balancer.
ALB_SG=$(aws ec2 create-security-group --group-name $APP-alb --vpc-id $VPC \
        --description "$APP load balancer" --query GroupId --output text --region $REGION)
aws ec2 authorize-security-group-ingress --group-id $ALB_SG --protocol tcp \
        --port 80 --cidr 0.0.0.0/0 --region $REGION

TASK_SG=$(aws ec2 create-security-group --group-name $APP-task --vpc-id $VPC \
        --description "$APP task" --query GroupId --output text --region $REGION)
aws ec2 authorize-security-group-ingress --group-id $TASK_SG --protocol tcp \
        --port 8080 --source-group $ALB_SG --region $REGION

# 2. Load balancer and target group.
ALB=$(aws elbv2 create-load-balancer --name $APP --type application \
        --subnets ${SUBNETS//,/ } --security-groups $ALB_SG \
        --query 'LoadBalancers[0].LoadBalancerArn' --output text --region $REGION)
TG=$(aws elbv2 create-target-group --name $APP --protocol HTTP --port 8080 \
        --vpc-id $VPC --target-type ip --health-check-path /api/health \
        --query 'TargetGroups[0].TargetGroupArn' --output text --region $REGION)
aws elbv2 create-listener --load-balancer-arn $ALB --protocol HTTP --port 80 \
        --default-actions Type=forward,TargetGroupArn=$TG --region $REGION

# 3. Raise the idle timeout. The progress stream is a long-lived connection, and
#    the 60s default closes it mid-job.
aws elbv2 modify-load-balancer-attributes --load-balancer-arn $ALB \
        --attributes Key=idle_timeout.timeout_seconds,Value=300 --region $REGION

# 4. The service. One task, and no autoscaling — see above.
aws ecs create-service --cluster $APP --service-name $APP \
        --task-definition $APP --desired-count 1 --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$TASK_SG],assignPublicIp=ENABLED}" \
        --load-balancers "targetGroupArn=$TG,containerName=grader,containerPort=8080" \
        --health-check-grace-period-seconds 60 --region $REGION
```

Then the URL:

```bash
aws elbv2 describe-load-balancers --names vedaai-grader \
  --query 'LoadBalancers[0].DNSName' --output text --region ap-south-1
```

Put that hostname in `WEB_ORIGINS` and release again, which switches off the
development allowance that accepts any loopback origin.

`assignPublicIp=ENABLED` is there because the task has to reach Textract and the
marking API. Private subnets with a NAT gateway are the tidier answer and cost more
than everything else here combined; for a hundred test users this is the right
trade.

## Every release after that

```bash
deploy/deploy.sh release
```

Builds for `linux/amd64` — explicitly, because an image built on an Apple laptop is
arm64 and the failure is an "exec format error" in the task logs long after the
push looked fine — pushes to ECR, registers a task definition, and forces a new
deployment.

## The marking credential

Either provider works. Create the secret once in the Secrets Manager console under
a name such as `vedaai-grader/openai`, holding the key. Copy the resulting ARN,
then:

```bash
export OPENAI_SECRET_ARN=<the ARN from the console>
# or, for the other provider:
# export ANTHROPIC_SECRET_ARN=<the ARN from the console>
deploy/deploy.sh release
```

The task definition references it by ARN and ECS injects it at container start, so
the value lives in neither the image nor this repository. The execution role needs
read access to that one ARN — attach it in the console alongside the managed
execution policy.

With no `GRADER_PROVIDER` set, whichever key is present is used. That is only safe
because the engine and model that produced a grade are recorded on the grade
itself, so which one judged an answer is never a guess.

The OpenAI default is the small model, `gpt-4o-mini`, and the reasoning is
structural rather than optimistic. Marking is a short, tightly constrained call —
read a rubric, read numbered lines, fill in a schema — and the two ways a weak
model fails that are both already contained: malformed output is prevented by
demanding a schema in strict mode rather than prose, and invented line citations
are caught by validation, which refuses the grade rather than displaying it, so
the failure mode is *no mark* rather than a wrong one.

What is not contained is judgement — whether a student's own wording satisfies a
criterion, in text a recognizer has already damaged. Nothing in the architecture
rescues that. Set `GRADER_MODEL` to something larger if the marks disappoint, and
measure rather than assume.

Without any of this the app still works. Marking returns the rubric derived from
the paper and the located answer, with every point left for the teacher, rather
than inventing a score.

## Checking it

```bash
curl http://YOUR-ALB/api/health

API=http://YOUR-ALB/api tooling/scripts/upload.py \
  samples/reading_comprehension_unit_test.pdf \
  data/samples/theory_a_in_order.pdf

aws logs tail /ecs/vedaai-grader --follow --region ap-south-1
```

The health endpoint reports the render DPI, which the frontend cross-checks. A
mismatch would not throw — normalized coordinates mean the browser never divides by
DPI — it would silently offset every highlight, so it is worth seeing.

## Known limits

- **One task.** In-memory submissions. Restarting the service loses in-flight work.
- ~~**HTTP, not HTTPS.**~~ Resolved by `gateway.sh` — an API Gateway HTTP API in
  front of the task gives a fixed hostname and an AWS-managed certificate. Run
  `deploy/gateway.sh create` once; CI runs `point` after each rollout, which is
  what makes the address stable rather than stable-until-the-next-deploy. Its 10 MB
  body cap is why uploads go straight to object storage, and its 30-second timeout
  is why ingest runs in the background. Kept below for the record:

  **HTTP, not HTTPS.** Add a certificate in ACM and a 443 listener; the load
  balancer is already there for it.
- **Textract is unmeasured on this data.** The comparison script is written and
  ready — `tooling/scripts/compare_ocr.py` — and recall is the ceiling on
  everything downstream, so run it before trusting the accuracy figures quoted
  elsewhere in this repo, which were all measured with the local recognizer.

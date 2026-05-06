# AWS CodeBuild Setup for deepep-v2-efa-base

This document explains how to set up AWS CodeBuild to build and publish the base image to Amazon ECR.

## Prerequisites

- AWS CLI configured with appropriate credentials
- An AWS account with CodeBuild and ECR permissions
- Docker privileged mode support in your AWS account

## 1. Create ECR Repository

```bash
export AWS_REGION=us-east-1  # adjust to your region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws ecr create-repository \
  --repository-name deepep-v2-efa-base \
  --region $AWS_REGION \
  --image-scanning-configuration scanOnPush=true
```

## 2. Create IAM Role for CodeBuild

Create a trust policy file `codebuild-trust-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "codebuild.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Create the role:

```bash
aws iam create-role \
  --role-name deepep-v2-efa-base-codebuild-role \
  --assume-role-policy-document file://codebuild-trust-policy.json
```

Attach managed policies:

```bash
aws iam attach-role-policy \
  --role-name deepep-v2-efa-base-codebuild-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

aws iam attach-role-policy \
  --role-name deepep-v2-efa-base-codebuild-role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
```

## 3. Create CodeBuild Project

```bash
aws codebuild create-project \
  --name deepep-v2-efa-base-build \
  --source type=GITHUB,location=https://github.com/antonai-work/deepep-v2-efa-base.git \
  --artifacts type=NO_ARTIFACTS \
  --environment type=LINUX_CONTAINER,computeType=BUILD_GENERAL1_2XLARGE,image=aws/codebuild/amazonlinux2-x86_64-standard:5.0,privilegedMode=true \
  --service-role arn:aws:iam::${AWS_ACCOUNT_ID}:role/deepep-v2-efa-base-codebuild-role \
  --environment-variables-override \
    name=AWS_ACCOUNT_ID,value=${AWS_ACCOUNT_ID},type=PLAINTEXT \
    name=AWS_REGION,value=${AWS_REGION},type=PLAINTEXT \
    name=ECR_REPO,value=deepep-v2-efa-base,type=PLAINTEXT \
    name=BASE_VERSION_TAG,value=v0.1.2-sm90a,type=PLAINTEXT
```

## 4. Trigger Build

```bash
aws codebuild start-build --project-name deepep-v2-efa-base-build
```

Monitor build progress:

```bash
# Get the build ID from the previous command output
BUILD_ID="deepep-v2-efa-base-build:YOUR_BUILD_ID"
aws codebuild batch-get-builds --ids $BUILD_ID
```

## 5. Verify Image in ECR

```bash
aws ecr describe-images \
  --repository-name deepep-v2-efa-base \
  --region $AWS_REGION
```

## Expected Build Time

- Image build: 30-45 minutes
- Preflight validation: 1-2 minutes
- ECR push: 2-5 minutes

**Total: ~40-55 minutes**

## Notes

- This is an ALTERNATIVE to the GitHub Actions workflow at `.github/workflows/build-and-push.yml`
- GitHub Actions publishes to GHCR (public, zero-config)
- CodeBuild publishes to ECR (opt-in, for AWS-native deployments)
- The two build paths do not conflict
- Preflight must show `5/5 checks PASS` or the build will fail
- No account IDs are hardcoded in buildspec.yml - all values come from environment variables

# CI Directory

This directory hosts AWS CodeBuild specifications as an alternative to the GitHub Actions workflow at `.github/workflows/build-and-push.yml`.

## Two Build Options

### GitHub Actions → GHCR (default)
- **Registry**: GitHub Container Registry (ghcr.io)
- **Visibility**: Public
- **Cost**: Free for public repositories
- **Setup**: Zero configuration - runs automatically on push/tag
- **Use case**: Default choice for open-source consumers

### AWS CodeBuild → ECR (opt-in)
- **Registry**: Amazon Elastic Container Registry (ECR)
- **Visibility**: Private by default
- **Cost**: AWS charges apply
- **Setup**: Requires IAM role, ECR repo, and CodeBuild project (see `CODEBUILD-SETUP.md`)
- **Use case**: AWS-native deployments, private caching, compliance requirements

## Files

- `buildspec.yml`: AWS CodeBuild build specification
- `CODEBUILD-SETUP.md`: Step-by-step setup instructions for CodeBuild
- `README.md`: This file

## Which Should I Use?

Use **GitHub Actions** (the default) unless you need:
- Private registry hosting in your AWS account
- Build artifacts in the same region as your compute
- Compliance requirements that prohibit public registries

The two paths are independent and do not conflict.

#!/bin/bash
# Script to set up GitHub organization-level secrets for Lambda pipeline execution
# This script should be run once by an organization admin to configure secrets
# Usage: ./scripts/setup-github-secrets.sh

set -e

# Configuration
ORG_NAME="plantcad"

echo "GitHub Secrets Setup for ${ORG_NAME} organization"
echo "=================================================="
echo ""
echo "This script will create organization-level secrets for the Lambda pipeline workflow."
echo "You will be prompted to enter values for each secret."
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub CLI."
    echo "Please run: gh auth login"
    exit 1
fi

# Function to set a secret
set_org_secret() {
    local secret_name=$1
    local secret_description=$2
    local secret_value

    echo ""
    echo "Setting ${secret_name}"
    echo "Description: ${secret_description}"
    read -sp "Enter value for ${secret_name}: " secret_value
    echo ""

    if [ -z "$secret_value" ]; then
        echo "Warning: Empty value provided for ${secret_name}, skipping..."
        return
    fi

    echo "$secret_value" | gh secret set "$secret_name" \
        --org "$ORG_NAME" \
        --visibility all

    echo "✓ ${secret_name} set successfully"
}

echo "Setting up secrets for organization: ${ORG_NAME}"
echo ""

# Set each secret
set_org_secret "LAMBDA_API_KEY" \
    "Lambda Cloud API key (create at https://cloud.lambda.ai/api-keys/cloud-api)"

set_org_secret "AWS_ACCESS_KEY_ID" \
    "AWS Access Key ID for S3 storage access"

set_org_secret "AWS_SECRET_ACCESS_KEY" \
    "AWS Secret Access Key for S3 storage access"

set_org_secret "HUGGING_FACE_HUB_TOKEN" \
    "Hugging Face Hub token (create at https://huggingface.co/settings/tokens)"

echo ""
echo "=================================================="
echo "All secrets have been configured successfully!"
echo ""
echo "You can verify the secrets at:"
echo "https://github.com/orgs/${ORG_NAME}/settings/secrets/actions"
echo ""
echo "The Lambda Pipeline workflow is now ready to use."

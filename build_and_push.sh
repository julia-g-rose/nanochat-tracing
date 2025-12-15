#!/bin/bash
# Build and push Docker image for CoreWeave training
#
# Usage:
#   ./build_and_push.sh <dockerhub_username> <version_tag>
#
# Example:
#   ./build_and_push.sh myusername v1

set -e  # Exit on error

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Error: Missing arguments"
    echo "Usage: ./build_and_push.sh <dockerhub_username> <version_tag>"
    echo "Example: ./build_and_push.sh myusername v1"
    exit 1
fi

DOCKERHUB_USERNAME=$1
VERSION_TAG=$2
IMAGE_NAME="${DOCKERHUB_USERNAME}/nanochat:${VERSION_TAG}"

echo "================================================"
echo "Building Docker image: ${IMAGE_NAME}"
echo "================================================"

# Build with BuildKit for faster builds and better caching
DOCKER_BUILDKIT=1 docker buildx build \
    --platform linux/amd64 \
    -f DOCKERFILE \
    -t "${IMAGE_NAME}" \
    --push \
    .

echo ""
echo "================================================"
echo "✅ Successfully built and pushed: ${IMAGE_NAME}"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Update k8s_job.yaml with your image name:"
echo "   sed -i '' 's|<YOUR_DOCKERHUB_USERNAME>/nanochat:v1|${IMAGE_NAME}|g' k8s_job.yaml"
echo ""
echo "2. (Optional) Add your W&B API key to k8s_job.yaml for logging"
echo ""
echo "3. Launch the job on CoreWeave:"
echo "   kubectl create -f k8s_job.yaml"
echo ""
echo "4. Monitor the job:"
echo "   kubectl get jobs"
echo "   kubectl get pods"
echo "   kubectl logs -f <pod-name>"
echo ""

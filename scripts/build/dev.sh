VERSION=$(cat ./VERSION)

DOCKER_IMAGE_NAME="faceoff:dev-${VERSION}"
DOCKERFILE="dev.dockerfile"
echo "Building docker image: ${DOCKER_IMAGE_NAME} from ${DOCKERFILE}"
# for using 127.0.0.1 proxy, we need build with host network
# please read: https://stackoverflow.com/questions/47067944/how-to-access-the-hosts-machines-localhost-127-0-0-1-from-docker-container
docker build --network="host" --progress=plain -t ${DOCKER_IMAGE_NAME} -f ${DOCKERFILE} .

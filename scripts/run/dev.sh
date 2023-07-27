VERSION=$(cat ./VERSION)
DOCKER_IMAGE_NAME="faceoff:dev-${VERSION}"
docker run -it -v ./app.py:/workdir/app.py -p 8501:8501 --rm  ${DOCKER_IMAGE_NAME} streamlit run app.py
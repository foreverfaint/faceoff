# syntax=docker/dockerfile:1

# Image from dockerhub https://hub.docker.com/_/ubuntu?tab=tags&page=1&name=20.04
FROM ubuntu:22.04

# Setting PYTHONUNBUFFERED to a non empty value ensures that the python output is 
# sent straight to terminal (e.g. your container log) without being first buffered 
ENV PYTHONUNBUFFERED=1 

# Improve script reliability by setting XE environment variables
# REF: https://stackoverflow.com/questions/29141436/set-e-and-set-x-in-shell-script
RUN set -xe

# `apt install tzdata` will set timezone without interactive
# using ARG, which is not persisted in the final image.
# using ENV changes the behavior of apt-get, and may confuse users of your image.
ARG DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get -y update && \    
    apt-get -y clean && \
    apt-get -y install curl ffmpeg libbz2-dev python3-all python3-dev python-is-python3 python-pip
    # apt-get -y install curl unzip git build-essential gcc make yasm autoconf automake cmake libtool checkinstall libmp3lame-dev pkg-config libunwind-dev zlib1g-dev libssl-dev

# RUN apt-get update \
#     && apt-get clean \
#     && apt-get install -y --no-install-recommends libc6-dev libgdiplus wget software-properties-common

# Make a working directory in the container
WORKDIR /workdir

# We use poetry for package managements
RUN curl -sSL https://install.python-poetry.org | python3 -
# Need update PATH env, please read: https://github.com/python-poetry/poetry/issues/525
ENV PATH="${PATH}:/root/.local/bin"

COPY ./pyproject.toml ./
COPY ./poetry.lock ./
RUN poetry --version && poetry install -vvv

COPY ./.streamlit ./.streamlit
COPY ./static ./static
COPY ./app.py ./
COPY ./models /root/.insightface/models
RUN mkdir /workdir/.output

ENTRYPOINT ["poetry", "run", "streamlit", "run", "app.py"]

#!/bin/bash

set -e

docker build -f PyTorchDockerfile -t alaska2 .

docker run --rm -it \
    --gpus all \
    --privileged=true \
    --shm-size=300g \
    -v "$(pwd):/alaska2/" \
    -w /alaska2/ \
    alaska2

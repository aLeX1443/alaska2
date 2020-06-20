#!/bin/bash

set -e

docker build -f TensorFlowDockerfile -t alaska2 .

docker run --rm -it \
    --gpus all \
    --privileged=true \
    -v "$(pwd):/alaska2/" \
    -w /alaska2/ \
    alaska2

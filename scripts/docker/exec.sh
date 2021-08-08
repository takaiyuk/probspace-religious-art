#!/bin/bash
CONTAINER_NAME="probspace-religious-art"
docker start ${CONTAINER_NAME} && docker exec -it ${CONTAINER_NAME} /bin/bash

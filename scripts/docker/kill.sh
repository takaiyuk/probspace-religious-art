#!/bin/bash
CONTAINER_NAME="probspace-religious-art"
docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}

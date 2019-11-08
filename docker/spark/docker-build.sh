#!/usr/bin/env bash

set -e

docker build -t spark-base:2.4.4 ./docker/base
docker build -t spark-master:2.4.4 ./docker/spark-master
docker build -t spark-worker:2.4.4 ./docker/spark-worker
docker build -t spark-submit:2.4.4 ./docker/spark-submit
docker build -t spark-dl4j:latest ./docker/spark-dl4j

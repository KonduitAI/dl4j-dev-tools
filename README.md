# spark-k8s
This repository contains docker/k8s setup for Apache Spark in-house DL4J testing

# How to use
- build images with `docker-build.sh`
- run cluster those using `docker-compose -f spark-cluster-docker-compose.yml up`
- run `docker run -it spark-dl4j bash` and start experimenting
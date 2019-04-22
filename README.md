# spark-k8s
This repository contains docker/k8s setup for Apache Spark in-house DL4J testing

# How to use
- build images with `docker-build.sh`
- run cluster using `docker-compose -f spark-cluster-docker-compose.yml up`
- run `docker run -it spark-dl4j bash` and start experimenting

# Example
- `cd dl4j-examples` and run `mvn install`
- `cp dl4j-spark-examples/dl4j-spark/target/dl4j-spark-1.0.0-beta3-bin.jar /opt/spark-apps`
- `/spark/bin/spark-submit --class org.deeplearning4j.legacyExamples.mlp.MnistMLPDistributedExample --master spark://10.5.0.2:6066 --deploy-mode cluster --driver-memory 2G --executor-memory 2G  /opt/spark-apps/dl4j-spark-1.0.0-beta3-bin.jar`
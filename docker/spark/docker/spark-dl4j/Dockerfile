#
#   deeplearning4j docker image
#

FROM ubuntu:bionic as builder

LABEL Description="Deeplearning4j Spark snapshots container"  Vendor="SkyMind.ai"  Version="1.0.0-SNAPSHOT"

ENV DAEMON_RUN=true
ENV SPARK_VERSION=2.3.3
ENV HADOOP_VERSION=2.7
ENV SCALA_VERSION=2.11.11
ENV SCALA_HOME=/usr/share/scala
ENV DL4J_BRANCH=master
ENV EXAMPLES_BRANCH=1.0.0-changes

ENV SPARK_MASTER_URL="spark://spark-master:7077"
ENV SPARK_SUBMIT_ARGS=""
ENV SPARK_APPLICATION_ARGS ""

RUN apt-get update && \
    apt-get -y --no-install-recommends install \
            wget git openjdk-8-jdk-headless maven gcc g++ automake cmake build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev libgomp1 libprotobuf-dev && \
    apt-get clean

RUN wget --no-verbose http://apache.mirror.iphh.net/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
      && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark \
      && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

RUN git clone https://github.com/deeplearning4j/deeplearning4j.git && cd deeplearning4j && ./change-scala-versions.sh 2.11 && ./change-spark-versions.sh 2
RUN cd deeplearning4j/ && git checkout ${DL4J_BRANCH}
RUN cd deeplearning4j/libnd4j && ./buildnativeoperations.sh -a native -j 6 && cd ../
RUN cd deeplearning4j/nd4j && mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd deeplearning4j/datavec && mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd deeplearning4j/deeplearning4j && mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd deeplearning4j/rl4j && mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd deeplearning4j/arbiter && mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true

RUN git clone https://github.com/deeplearning4j/dl4j-examples.git
RUN cd dl4j-examples/ && git checkout ${EXAMPLES_BRANCH} && mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true

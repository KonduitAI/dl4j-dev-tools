FROM ubuntu:bionic

ENV DAEMON_RUN=true
ENV SPARK_VERSION=2.3.3
ENV HADOOP_VERSION=2.7
ENV SCALA_VERSION=2.11.11
ENV SCALA_HOME=/usr/share/scala

RUN apt-get update && \
    apt-get -y --no-install-recommends install \
            wget vim inetutils-ping git scala openjdk-8-jre openjdk-8-jdk-headless maven gcc g++ automake cmake build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev libgomp1 libprotobuf-dev && \
    apt-get clean && \
    update-java-alternatives -s java-1.8.0-openjdk-amd64

RUN wget --no-verbose http://apache.mirror.iphh.net/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
      && mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark \
      && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
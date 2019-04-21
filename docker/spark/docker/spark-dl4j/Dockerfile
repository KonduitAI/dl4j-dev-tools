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
ENV EXAMPLES_BRANCH=1.0.0_changes

ENV SPARK_MASTER_URL="spark://spark-master:7077"
ENV SPARK_SUBMIT_ARGS=""
ENV SPARK_APPLICATION_ARGS ""

RUN git clone https://github.com/deeplearning4j/dl4j-examples.git
RUN git clone https://github.com/deeplearning4j/deeplearning4j.git && cd deeplearning4j && ./change-scala-versions.sh 2.11 && ./change-spark-versions.sh 2
RUN cd deeplearning4j/ && git checkout ${DL4J_BRANCH} && mvn dependency:resolve
RUN cd deeplearning4j/libnd4j && ./buildnativeoperations.sh -a native -j 6 && cd ../
RUN cd deeplearning4j/nd4j && mvn -T 2 clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd deeplearning4j/datavec && mvn -T 4 clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd deeplearning4j/deeplearning4j && mvn -T 4 clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd deeplearning4j/rl4j && mvn clean -T 4 install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true
RUN cd deeplearning4j/arbiter && mvn -T 4 clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true


# RUN cd dl4j-examples/ && git pull && git checkout ${EXAMPLES_BRANCH} && mvn clean install -DskipTests=true -Dmaven.javadoc.skip=true -Dmaven.test.skip=true

# just sleeping
CMD ["/bin/sleep", "365d"]

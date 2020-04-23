FROM continuumio/miniconda3:latest
RUN conda install tensorflow=2.1.0
RUN conda install psutil
ENV DL4J_TEST_RESOURCES=/dl4j-test-resources

#Usage:
#docker build --tag tfops:tf21 .
#docker run -v C:\DL4J\Git\dl4j-dev-tools\import-tests:/ImportTests -v C:/DL4J/Git/dl4j-test-resources:/dl4j-test-resources -v C:/Temp/TF_Graphs/:/TF_Graphs/ -it tfops:tf21
#cd ImportTests
#python setup.py develop
#python tests/test_ops.py
#python model_zoo/util/zoo_evaluation.py

#pip install tf2onnx
#cd ImportTests
#apt-get install dos2unix
#dos2unix convert.sh
#./convert.sh

#cd / && wget https://github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u242-b08/OpenJDK8U-jdk_x64_linux_hotspot_8u242b08.tar.gz && tar -xf OpenJDK8U-jdk_x64_linux_hotspot_8u242b08.tar.gz && export PATH=$PWD/jdk8u242-b08/bin:$PATH
#cd / && wget http://apache.mirror.amaze.com.au/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz && tar -xf apache-maven-3.6.3-bin.tar.gz && export PATH=/apache-maven-3.6.3/bin:$PATH
#cd /ImportTests/java/conversion && mvn package -DskipTests
#java -cp target/conversion-1.0-SNAPSHOT-bin.jar ai.konduit.ConvertTF2ONNX
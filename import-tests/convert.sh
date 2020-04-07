
pip install tf2onnx
pip install tensorflow=1.15.0

cd / && wget https://github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u242-b08/OpenJDK8U-jdk_x64_linux_hotspot_8u242b08.tar.gz && tar -xf OpenJDK8U-jdk_x64_linux_hotspot_8u242b08.tar.gz && export PATH=$PWD/jdk8u242-b08/bin:$PATH
cd / && wget http://apache.mirror.amaze.com.au/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz && tar -xf apache-maven-3.6.3-bin.tar.gz && export PATH=/apache-maven-3.6.3/bin:$PATH
cd /ImportTests/java/conversion && mvn package -DskipTests
java -cp target/conversion-1.0-SNAPSHOT-bin.jar ai.konduit.ConvertTF2ONNX

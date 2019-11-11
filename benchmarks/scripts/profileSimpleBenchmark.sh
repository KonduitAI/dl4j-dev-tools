#!/usr/bin/env bash

# This script: launches the simple benchmark and captures the snapshots automatically

cd ..
rootDir=$(pwd)
#mvn clean package -DskipTests -Pv091,cudnn8
#mvn clean package -DskipTests -Pv100beta,cudnn8
#mvn clean package -DskipTests -Pv100beta,cudnn91
#mvn clean package -DskipTests -Pv100snapshot,cudnn91
cd dl4j-core-benchmark
mkdir -p ../scripts/SimpleBenchmark

#declare -a versionBackend=("v091_cuda8-cudnn" "v100beta_cuda8-cudnn")
declare -a versionBackend=("v100snapshot_cuda91-cudnn")
declare -a batchSize=("64")
declare -a dataTypes=("FLOAT" "HALF")

#model=ALEXNET
model=RESNET50PRE
xmx=16G
javacpp=16G

# Launching with profiling:
# https://www.yourkit.com/docs/java/help/agent.jsp
# https://www.yourkit.com/docs/java/help/startup_options.jsp
#yourkitPath=/home/alex/Downloads/YourKit-JavaProfiler-2017.02/bin/linux-x86-64/libyjpagent.so
#yourkitJar=/home/alex/Downloads/YourKit-JavaProfiler-2017.02/lib/yjp-controller-api-redist.jar
#yourkitPath=C:/PROGRA~1/YOURKI~2.02-/bin/win64/yjpagent.dll
#yourkitJar=C:/PROGRA~1/YOURKI~2.02-/lib/yjp-controller-api-redist.jar
yourkitPath=/mnt/resource/src/YourKit-JavaProfiler-2018.04/bin/linux-x86-64/libyjpagent.so
yourkitJar=/mnt/resource/src/YourKit-JavaProfiler-2018.04/lib/yjp-controller-api-redist.jar

profilingSettingsPath=$rootDir/scripts/profiling/profiling_settings.txt

for i in "${versionBackend[@]}"
do
   for j in "${batchSize[@]}"
   do
      for k in "${dataTypes[@]}"
      do
          export SNAPSHOT_DIR=../scripts/SimpleBenchmark/$model/snapshot-$i-$j-$k/
          mkdir -p $SNAPSHOT_DIR
          echo "Running test: $i, batch size $j, data type $k"
          echo java -agentpath:"$yourkitPath"=tracing,port=10001,dir=$SNAPSHOT_DIR,tracing_settings_path=$profilingSettingsPath -cp dl4j-core-benchmark-$i.jar  -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.simple.SimpleBenchmark --nIter 100 --fit --minibatch $j --model $model --datatype $k > ../scripts/SimpleBenchmark/$model/output_"$i"_"$j"_"$k".txt
          java -agentpath:"$yourkitPath"=tracing,port=10001,dir=$SNAPSHOT_DIR,tracing_settings_path=$profilingSettingsPath -cp dl4j-core-benchmark-$i.jar  -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.simple.SimpleBenchmark --nIter 100 --fit --minibatch $j --model $model --datatype $k >> ../scripts/SimpleBenchmark/$model/output_"$i"_"$j"_"$k".txt

          #Note: snapshot recording will automatically occur when JVM shuts down
      done
   done
done



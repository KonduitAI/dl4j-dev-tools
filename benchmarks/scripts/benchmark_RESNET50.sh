#!/usr/bin/env bash
cd ..
#mvn clean package -DskipTests -Pv091,cudnn8
#mvn clean package -DskipTests -Pv100alpha,cudnn8
#mvn clean package -DskipTests -Pv100beta,cudnn8
#mvn clean package -DskipTests -Pv100beta3,native
mvn clean package -DskipTests -Pv100beta5,native
mvn clean package -DskipTests -Pv100beta6,native
mvn clean package -DskipTests -Pv100snapshot,native
#mvn clean package -DskipTests -Pv100beta3,cuda10
#mvn clean package -DskipTests -Pv100snapshot,cuda10
#mvn clean package -DskipTests -Pv100beta3,cudnn10
#mvn clean package -DskipTests -Pv100snapshot,cudnn10
cd dl4j-core-benchmark
#declare -a versionBackend=("v100beta3_native" "v100snapshot_native" "v100beta3_cuda10" "v100snapshot_cuda10" "v100beta3_cuda10-cudnn" "v100snapshot_cuda10-cudnn")
#declare -a versionBackend=("v100snapshot_native")
declare -a versionBackend=("v100beta5_native" "v100beta6_native" "v100snapshot_native")
declare -a batchSize=("32")
declare -a dataTypes=("FLOAT")
modelType=RESNET50
xmx=16G
javacpp=16G
totalIterations=30
mkdir -p ../scripts/${modelType}_${totalIterations}_iter
## now loop through the above array
for i in "${versionBackend[@]}"
do
   for j in "${batchSize[@]}"
   do
      for k in "${dataTypes[@]}"
      do
         echo "Running test: $i, batch size $j, data type $k"
         echo java -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --datatype $k --cacheMode NONE --totalIterations $totalIterations > ../scripts/${modelType}_${totalIterations}_iter/output_"$i"_"$j"_"$k".txt
         java -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --datatype $k --cacheMode NONE --totalIterations $totalIterations >> ../scripts/${modelType}_${totalIterations}_iter/output_"$i"_"$j"_"$k".txt
      done
   done
done

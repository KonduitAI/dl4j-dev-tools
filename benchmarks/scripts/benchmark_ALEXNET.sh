#!/usr/bin/env bash
#cd ..
#mvn clean package -DskipTests -Pv091,cudnn8
#mvn clean package -DskipTests -Pv100alpha,cudnn8
#mvn clean package -DskipTests -Pv100beta,cudnn8
#mvn clean package -DskipTests -Pv100beta,cudnn91
#mvn clean package -DskipTests -Pv100beta5,native
#mvn clean package -DskipTests -Pv100beta6,native
#mvn clean package -DskipTests -Pv100snapshot,native
#mvn clean package -DskipTests -Pv100beta5,cuda101
#mvn clean package -DskipTests -Pv100beta6,cuda101
#mvn clean package -DskipTests -Pv100beta6,cudnn101
#mvn clean package -DskipTests -Pv100snapshot,cudnn101
#mvn clean package -DskipTests -Pv100snapshot,cuda101

#mvn clean package -DskipTests -Pv100beta6,cuda102
#mvn clean package -DskipTests -Pv100snapshot,cuda102
mvn clean package -DskipTests -Pv100beta6,cudnn102
mvn clean package -DskipTests -Pv100snapshot,cudnn102

cd dl4j-core-benchmark
#declare -a versionBackend=("v091_cuda8-cudnn" "v100alpha_cuda8-cudnn" "v100beta_cuda8-cudnn")
#declare -a versionBackend=("v100beta5_native" "v100beta6_native" "v100snapshot_native")
#declare -a versionBackend=( "v100beta6_native" "v100snapshot_native")
#declare -a versionBackend=( "v100beta6_cuda101-cudnn" "v100snapshot_cuda101-cudnn")
#declare -a versionBackend=( "v100beta6_cuda102" "v100snapshot_cuda102")
declare -a versionBackend=( "v100beta6_cuda102-cudnn" "v100snapshot_cuda102-cudnn")

declare -a batchSize=("32")
modelType=ALEXNET
xmx=12G
javacpp=12G
mkdir -p ../scripts/$modelType
## now loop through the above array
for i in "${versionBackend[@]}"
do
   for j in "${batchSize[@]}"
   do
      echo "Running test: $i, batch size $j"
      echo java -cp dl4j-core-benchmark-$i.jar -Xmx$xmx -Dorg.bytedeco.javacpp.maxbytes=$javacpp -Dorg.bytedeco.javacpp.maxphysicalbytes=$javacpp org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --cacheMode NONE > ../scripts/$modelType/output_"$i"_"$j".txt
      java -cp dl4j-core-benchmark-$i.jar org.deeplearning4j.benchmarks.BenchmarkCnn --modelType $modelType --batchSize $j --cacheMode NONE >> ../scripts/$modelType/output_"$i"_"$j".txt
   done
done

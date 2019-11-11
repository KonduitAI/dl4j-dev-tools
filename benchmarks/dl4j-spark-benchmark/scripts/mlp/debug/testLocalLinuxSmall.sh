JARARGS="-tempPath /tmp/MLPSparkTestSmall/Temp/ -resultPath /tmp/MLPSparkTestSmall/Results/ -useSparkLocal true"
JARARGS="$JARARGS -dataLoadingMethods SparkBinaryFiles Parallelize -numTestFiles 1920 -numParams 100000 -dataSize 128"
JARARGS="$JARARGS -miniBatchSizePerWorker 32 -saveUpdater true -repartition Always -repartitionStrategy Balanced"
JARARGS="$JARARGS -workerPrefetchNumBatches 0"
java -Xms16G -Xmx16G -cp dl4j-spark-benchmark.jar org.deeplearning4j.train.RunTrainingTests $JARARGS

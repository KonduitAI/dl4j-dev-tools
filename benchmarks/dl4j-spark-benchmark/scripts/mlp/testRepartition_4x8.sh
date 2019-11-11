JARARGS="-tempPath hdfs:/exp2_repart_4x8/temp/ -resultPath hdfs:/exp2_repart_4x8/results/ -useSparkLocal false"
JARARGS="$JARARGS -dataLoadingMethods SparkBinaryFiles StringPath Parallelize -numDataSetObjects 3200 -numParams 100000 -dataSize 1000"
JARARGS="$JARARGS -miniBatchSizePerWorker 8 -saveUpdater true -repartition Always -repartitionStrategy Balanced SparkDefault"
JARARGS="$JARARGS -workerPrefetchNumBatches 2"
SPARKARGS="--class org.deeplearning4j.train.RunTrainingTests --num-executors 4 --executor-cores 8 --executor-memory 10G --driver-memory 10G"
spark-submit $SPARKARGS dl4j-spark-benchmark.jar $JARARGS
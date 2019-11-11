SET JARARGS=-tempPath file:/C:/Temp/MLPSparkTestSmall/Temp/ -resultPath file:/C:/Temp/MLPSparkTestSmall/Results/ -useSparkLocal true
SET JARARGS=%JARARGS% -dataLoadingMethods SequenceFile SparkBinaryFiles Parallelize StringPath CSV -numDataSetObjects 1920 -numParams 100000 -dataSize 128
SET JARARGS=%JARARGS% -miniBatchSizePerWorker 32 -saveUpdater true -repartition Always -repartitionStrategy Balanced
SET JARARGS=%JARARGS% -workerPrefetchNumBatches 0
java -Xms16G -Xmx16G -cp dl4j-spark-benchmark.jar org.deeplearning4j.train.RunTrainingTests %JARARGS%
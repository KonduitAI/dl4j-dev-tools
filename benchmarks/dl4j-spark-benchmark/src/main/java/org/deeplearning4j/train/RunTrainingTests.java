package org.deeplearning4j.train;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.data.DataSetExportFunction;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.deeplearning4j.spark.time.TimeSource;
import org.deeplearning4j.spark.time.TimeSourceProvider;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.train.config.CsvCompressionCodec;
import org.deeplearning4j.train.config.MLPTest;
import org.deeplearning4j.train.config.RNNTest;
import org.deeplearning4j.train.config.SparkTest;
import org.deeplearning4j.train.functions.GenerateCsvDataFunction;
import org.deeplearning4j.train.functions.GenerateDataFunction;
import org.deeplearning4j.train.functions.sequence.FromSequenceFilePairFunction;
import org.deeplearning4j.train.functions.sequence.ToSequenceFilePairFunction;
import org.deeplearning4j.train.misc.EnumConverters;
import org.nd4j.linalg.dataset.DataSet;

import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Alex on 23/07/2016.
 */
@Slf4j
public class RunTrainingTests {

    @Parameter(names = "-useSparkLocal", description = "Whether to use spark local (if false: use spark submit)", arity = 1)
    protected boolean useSparkLocal = false;

    @Parameter(names = "-testType", description = "Type of test to run. One of: MLP, RNN, CNN")
    protected String testType = TestType.MLP.name();

    @Parameter(names = "-averagingFrequency", description = "Averaging frequency (or frequencies) to test: as a list", variableArity = true)
    protected List<Integer> averagingFrequency = Arrays.asList(5);

    @Parameter(names = "-dataLoadingMethods", description = "List of data loading methods to test", variableArity = true, converter = EnumConverters.DataLoadingMethodEnumConverter.class)
    protected List<DataLoadingMethod> dataLoadingMethods = new ArrayList<>(Arrays.asList(DataLoadingMethod.SparkBinaryFiles, DataLoadingMethod.Parallelize, DataLoadingMethod.StringPath));

    @Parameter(names = "-csvCompressionCodec", description = "List of compression codecs used for CSV tests. Options: None, Deflate, GZip, BZip, LZ4, Snappy", variableArity = true)
    protected List<CsvCompressionCodec> csvCompressionCodecs = new ArrayList<>(Arrays.asList(CsvCompressionCodec.None));

    @Parameter(names = "-csvCoalesceSize", description = "Before writing the CSV file: should we coalesce the data, or not? -1 = don't coalesce, otherwise specify the number of partitions to coalesce to")
    protected int csvCoalesceSize = -1;

    @Parameter(names = "-numDataSetObjects", description = "Number of test files (DataSet objects of size miniBatchSizePerWorker) - as list", variableArity = true)
    protected List<Integer> numDataSetObjects = new ArrayList<>(Arrays.asList(2000));

    @Parameter(names = "-tempPath", description = "Path to the test directory (typically HDFS), in which to generate data", required = true)
    protected String tempPath;

    @Parameter(names = "-resultPath", description = "Path to the base output directory. Results will be placed in a subdirectory. For example, HDFS or S3", required = true)
    protected String resultPath;

    @Parameter(names = "-skipExisting", description = "Flag to skip (don't re-run) any tests that have already been completed")
    protected boolean skipExisting;

    @Parameter(names = "-numParams", variableArity = true, description = "Number of parameters in the network, as a list: \"-numParams 100000 1000000 10000000\"")
    protected List<Integer> numParams = new ArrayList<>(Arrays.asList(100_000, 1_000_000, 10_000_000));

    @Parameter(names = "-dataSize", variableArity = true, description = "Size of the data set (i.e., num inputs/outputs)")
    protected List<Integer> dataSize = new ArrayList<>(Arrays.asList(16, 128, 512, 2048));

    @Parameter(names = "-cnnImageWidth", variableArity = true, description = "Width (and, height) the image data set")
    protected List<Integer> cnnImageWidth = new ArrayList<>(Arrays.asList(32, 128, 256));

    @Parameter(names = "-miniBatchSizePerWorker", variableArity = true, description = "Number of examples per worker/minibatch, as a list: \"-miniBatchSizePerWorker 8 32 128\"")
    protected List<Integer> miniBatchSizePerWorker = new ArrayList<>(Arrays.asList(8, 32, 128));

    @Parameter(names = "-saveUpdater", description = "Whether the updater should be saved or not", arity = 1)
    protected boolean saveUpdater = true;

    @Parameter(names = "-repartition", description = "When repartitioning should occur", converter = EnumConverters.RepartitionEnumConverter.class)
    protected Repartition repartition = Repartition.Always;

    @Parameter(names = "-repartitionStrategy", description = "Repartition strategies to use when repartitioning, as list (options: Balanced, SparkDefault)", converter = EnumConverters.RepartitionStrategyEnumConverter.class, variableArity = true)
    protected List<RepartitionStrategy> repartitionStrategy = Arrays.asList(RepartitionStrategy.Balanced);

    @Parameter(names = "-workerPrefetchNumBatches", description = "Number of batches to prefetch")
    protected int workerPrefetchNumBatches = 0;

    @Parameter(names = "-rnnTimeSeriesLength", description = "Data length (number of time steps) for RNN data")
    protected int rnnTimeSeriesLength = 100;

    @Parameter(names = "-rnnUseTBPTT", description = "Whether to use truncated BPTT for RNNs")
    protected boolean rnnUseTBPTT = false;

    @Parameter(names = "-rnnTBPTTLength", description = "Length to use for truncated BPTT")
    protected int rnnTBPTTLength = 100;

    public static void main(String[] args) throws Exception {
        new RunTrainingTests().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        TimeSource ts = TimeSourceProvider.getInstance();

        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }

        String launchArgsPath = resultPath + (resultPath.endsWith("/") ? "" : "/") + System.currentTimeMillis() + "_" + testType + "_launchConf.txt";
        //Log the launch configuration
        String f = "%-40s\t%s\n";
        StringBuilder lp = new StringBuilder();
        lp.append("Launching job with args:\n");
        lp.append(String.format(f,"launchArgsPath",launchArgsPath));
        lp.append(String.format(f,"useSparkLocal",useSparkLocal));
        lp.append(String.format(f,"testType",testType));
        lp.append(String.format(f,"dataLoadingMethods",dataLoadingMethods));
        lp.append(String.format(f,"numDataSetObjects", numDataSetObjects));
        lp.append(String.format(f,"tempPath",tempPath));
        lp.append(String.format(f,"resultPath",resultPath));
        lp.append(String.format(f,"skipExisting",skipExisting));
        lp.append(String.format(f,"numParams",numParams));
        if(TestType.valueOf(testType) != TestType.CNN) lp.append(String.format(f,"dataSize",dataSize));
        else lp.append(String.format(f,"cnnImageSize",cnnImageWidth));
        lp.append(String.format(f,"miniBatchSizePerWorker",miniBatchSizePerWorker));
        lp.append(String.format(f,"saveUpdater",saveUpdater));
        lp.append(String.format(f,"repartition",repartition));
        lp.append(String.format(f,"repartitionStrategy",repartitionStrategy));
        lp.append(String.format(f,"workerPrefetchNumBatches",workerPrefetchNumBatches));
        lp.append(String.format(f,"rnnTimeSeriesLength",rnnTimeSeriesLength));
        lp.append(String.format(f,"rnnUseTBPTT",rnnUseTBPTT));
        lp.append(String.format(f,"rnnTBPTTLength",rnnTBPTTLength));
        log.info(lp.toString());

        SparkConf conf = new SparkConf();
        conf.setAppName("RunTrainingTests");
        if(useSparkLocal) conf.setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //Write launch args to file:
        org.datavec.spark.transform.utils.SparkUtils.writeStringToFile(launchArgsPath,lp.toString(),sc);  //Write a copy of  the launch arguments to file

        List<SparkTest> testsToRun = new ArrayList<>();
        for (Integer np : numParams) {
            for (Integer ds : dataSize) {
                for (Integer mbs : miniBatchSizePerWorker) {
                    for (DataLoadingMethod dataLoadingMethod : dataLoadingMethods) {
                        for(RepartitionStrategy rs : repartitionStrategy ) {
                            for(Integer numDSObjs : numDataSetObjects) {
                                for (Integer avgFreq : averagingFrequency) {

                                    if (dataLoadingMethod == DataLoadingMethod.CSV) {
                                        for (CsvCompressionCodec c : csvCompressionCodecs) {
                                            switch (TestType.valueOf(testType)) {
                                                case MLP:
                                                    testsToRun.add(
                                                            new MLPTest.Builder()
                                                                    .numDataSetObjects(numDSObjs)
                                                                    .paramsSize(np)
                                                                    .dataSize(ds)
                                                                    .dataLoadingMethod(dataLoadingMethod)
                                                                    .averagingFrequency(avgFreq)
                                                                    .minibatchSizePerWorker(mbs)
                                                                    .saveUpdater(saveUpdater)
                                                                    .repartition(repartition)
                                                                    .repartitionStrategy(rs)
                                                                    .workerPrefetchNumBatches(workerPrefetchNumBatches)
                                                                    .csvCompressionCodec(c)
                                                                    .csvCoalesceSize(csvCoalesceSize)
                                                                    .build());
                                                    break;
                                                case RNN:
                                                    testsToRun.add(
                                                            new RNNTest.Builder()
                                                                    .numDataSetObjects(numDSObjs)
                                                                    .paramsSize(np)
                                                                    .dataSize(ds)
                                                                    .dataLoadingMethod(dataLoadingMethod)
                                                                    .averagingFrequency(avgFreq)
                                                                    .minibatchSizePerWorker(mbs)
                                                                    .saveUpdater(saveUpdater)
                                                                    .repartition(repartition)
                                                                    .repartitionStrategy(rs)
                                                                    .workerPrefetchNumBatches(workerPrefetchNumBatches)
                                                                    .timeSeriesLength(rnnTimeSeriesLength)
                                                                    .csvCompressionCodec(c)
                                                                    .csvCoalesceSize(csvCoalesceSize)
                                                                    .build());
                                                    break;
                                                case CNN:
                                                    throw new UnsupportedOperationException("CNN tests not yet implemented");
                                                default:
                                                    throw new RuntimeException("Unknown test type: " + testType);
                                            }
                                        }
                                    } else {
                                        switch (TestType.valueOf(testType)) {
                                            case MLP:
                                                testsToRun.add(
                                                        new MLPTest.Builder()
                                                                .numDataSetObjects(numDSObjs)
                                                                .paramsSize(np)
                                                                .dataSize(ds)
                                                                .dataLoadingMethod(dataLoadingMethod)
                                                                .averagingFrequency(avgFreq)
                                                                .minibatchSizePerWorker(mbs)
                                                                .saveUpdater(saveUpdater)
                                                                .repartition(repartition)
                                                                .repartitionStrategy(rs)
                                                                .workerPrefetchNumBatches(workerPrefetchNumBatches)
                                                                .csvCoalesceSize(csvCoalesceSize)
                                                                .build());
                                                break;
                                            case RNN:
                                                testsToRun.add(
                                                        new RNNTest.Builder()
                                                                .numDataSetObjects(numDSObjs)
                                                                .paramsSize(np)
                                                                .dataSize(ds)
                                                                .dataLoadingMethod(dataLoadingMethod)
                                                                .averagingFrequency(avgFreq)
                                                                .minibatchSizePerWorker(mbs)
                                                                .saveUpdater(saveUpdater)
                                                                .repartition(repartition)
                                                                .repartitionStrategy(rs)
                                                                .workerPrefetchNumBatches(workerPrefetchNumBatches)
                                                                .timeSeriesLength(rnnTimeSeriesLength)
                                                                .csvCoalesceSize(csvCoalesceSize)
                                                                .build());
                                                break;
                                            case CNN:
                                                throw new UnsupportedOperationException("CNN tests not yet implemented");
                                            default:
                                                throw new RuntimeException("Unknown test type: " + testType);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }



        Configuration config = new Configuration();
        FileSystem fileSystem = FileSystem.get(URI.create(tempPath), config);

        int test = 0;
        for (SparkTest sparkTest : testsToRun) {

            int numDSObjects = sparkTest.getNumDataSetObjects();

            List<Integer> intList = new ArrayList<>();
            for(int i = 0; i< numDSObjects; i++ ) intList.add(i);
            JavaRDD<Integer> intRDD = sc.parallelize(intList);

            long testStartTime = System.currentTimeMillis();

            String dataDir = tempPath + (tempPath.endsWith("/") ? "" : "/") + test + "/";

            boolean exists = fileSystem.exists(new Path(tempPath));
            if (exists) {
                if (skipExisting) {
                    log.info("Temporary directory exists; skipping test. {}", tempPath);
                    continue;
                }
                log.info("Temporary directory exists; attempting to delete. {}", tempPath);
                fileSystem.delete(new Path(tempPath), true);
            }

            //Step 1: generate data
            long startGenerateExport = System.currentTimeMillis();
            JavaRDD<DataSet> trainData = null;
            JavaRDD<String> stringPaths = null;
            switch (sparkTest.getDataLoadingMethod()) {
                case SparkBinaryFiles:
                    log.info("Generating/exporting data at directory: {}", dataDir);
                    JavaRDD<DataSet> data = intRDD.map(new GenerateDataFunction(sparkTest));
                    data.foreachPartition(new DataSetExportFunction(new URI(dataDir)));
                    break;
                case Parallelize:
                    List<DataSet> tempList = new ArrayList<>();
                    for (int i = 0; i < numDSObjects; i++) {
                        tempList.add(sparkTest.getSyntheticDataSet());
                    }
                    trainData = sc.parallelize(tempList);
                    trainData.cache();
                    break;
                case StringPath:
                    log.info("Generating/exporting data at directory: {}", dataDir);
                    JavaRDD<DataSet> data2 = intRDD.map(new GenerateDataFunction(sparkTest));
                    data2.foreachPartition(new DataSetExportFunction(new URI(dataDir)));

                    FileSystem hdfs = FileSystem.get(URI.create(tempPath), config);

                    RemoteIterator<LocatedFileStatus> fileIter = hdfs.listFiles(new org.apache.hadoop.fs.Path(dataDir), false);

                    List<String> paths = new ArrayList<>();
                    while(fileIter.hasNext()){
                        String path = fileIter.next().getPath().toString();
                        paths.add(path);
                    }

                    stringPaths = sc.parallelize(paths);
                    stringPaths.cache();

                    break;
                case CSV:
                    log.info("Generating/exporting CSV test data at directory: {}", dataDir);
                    //Generate some CSV data. Dimensions:
                    //Number of lines/examples: number of DataSet objects * minibatch size
                    //Number of values in each: 2*dataSize (in this test: input/output is the same size)
                    int numLines = numDSObjects * sparkTest.getMinibatchSizePerWorker();
                    int valuesPerLine = 2 * sparkTest.getDataSize();
                    intList = new ArrayList<>();
                    for(int i=0; i<numLines; i++ ){
                        intList.add(i);
                    }

                    intRDD = sc.parallelize(intList);
                    JavaRDD<String> csvData = intRDD.map(new GenerateCsvDataFunction(valuesPerLine));
                    if(sparkTest.getCsvCoalesceSize() > 0){
                        csvData = csvData.coalesce(sparkTest.getCsvCoalesceSize());
                    } else {
                        csvData.coalesce(sc.defaultParallelism());
                    }

                    Configuration c = sc.hadoopConfiguration();
                    if(sparkTest.getCsvCompressionCodec() == CsvCompressionCodec.None){
                        //Default compression codec as specified in hadoop configuration will be used normally, depending
                        // on the cluster configuration. For example, we might get Snappy output from the saveAsTextFile
                        // operation, even if we set test configuration to "None". This isn't what we want, so we'll override
                        // the appropriate configuration options here.
                        c.set("mapreduce.map.output.compress","false");
                        c.set("mapred.output.compress", "false");

                        csvData.saveAsTextFile(dataDir);
                    } else {
                        Class<? extends CompressionCodec> ccClass = sparkTest.getCsvCompressionCodec().getCodec();
                        c.set("mapreduce.map.output.compress","true");
                        c.set("mapred.output.compress", "true");
                        c.setClass("mapreduce.map.output.compress.codec", ccClass, CompressionCodec.class);
                        c.set("mapred.output.compression.codec",ccClass.getName());

                        csvData.saveAsTextFile(dataDir, ccClass);
                    }


                    break;
                case SequenceFile:
                    log.info("Generating SequenceFile for Data at directory: {}", dataDir);

                    //Need: DataSet as a sequence file
                    JavaRDD<DataSet> data3 = intRDD.map(new GenerateDataFunction(sparkTest));
                    JavaPairRDD<Text,BytesWritable> pairRDD = data3.mapToPair(new ToSequenceFilePairFunction());
                    pairRDD.saveAsHadoopFile(dataDir, Text.class, BytesWritable.class, SequenceFileOutputFormat.class);

                    JavaPairRDD<Text,BytesWritable> sequenceFile = sc.sequenceFile(dataDir, Text.class, BytesWritable.class);
                    trainData = sequenceFile.map(new FromSequenceFilePairFunction());

                    break;
                default:
                    throw new RuntimeException("Unknown data loading method: " + sparkTest.getDataLoadingMethod());
            }

            long endGenerateExport = System.currentTimeMillis();


            //Step 2: Train network for 1 epoch
            MultiLayerConfiguration netConf = sparkTest.getConfiguration();

            int dataSetObjectSize = (sparkTest.getDataLoadingMethod() == DataLoadingMethod.CSV ? 1 : sparkTest.getMinibatchSizePerWorker());
            TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(dataSetObjectSize)
                    .averagingFrequency(sparkTest.getAveragingFrequency())
                    .batchSizePerWorker(sparkTest.getMinibatchSizePerWorker())
                    .saveUpdater(sparkTest.isSaveUpdater())
                    .repartionData(sparkTest.getRepartition())
                    .repartitionStrategy(sparkTest.getRepartitionStrategy())
                    .workerPrefetchNumBatches(sparkTest.getWorkerPrefetchNumBatches())
                    .build();

            SparkDl4jMultiLayer net = new SparkDl4jMultiLayer(sc, netConf, tm);
            net.setCollectTrainingStats(true);

            long startFit = System.currentTimeMillis();
            switch (sparkTest.getDataLoadingMethod()) {
                case SparkBinaryFiles:
                    net.fit(dataDir);
                    break;
                case Parallelize:
                    net.fit(trainData);
                    break;
                case StringPath:
                    net.fitPaths(stringPaths);
                    break;
                case CSV:
                    JavaRDD<String> rawLines = sc.textFile(dataDir, sc.defaultParallelism());
                    JavaRDD<List<Writable>> writables = rawLines.map(new StringToWritablesFunction(new CSVRecordReader(0,",")));

                    int featuresSize = sparkTest.getDataSize();
                    JavaRDD<DataSet> sizeOneDataSets = writables.map(new DataVecDataSetFunction(featuresSize,2*featuresSize-1,-1, true, null, null));

                    net.fit(sizeOneDataSets);
                    break;
                case SequenceFile:
                    net.fit(trainData);
                    break;
                default:
                    throw new RuntimeException("Unknown data loading method: " + sparkTest.getDataLoadingMethod());
            }
            long endFit = System.currentTimeMillis();


            //Step 3: record results
            //(a) Configuration (as yaml)
            //(b) Times
            //(c) Spark stats HTML
            //(d) Spark stats raw data/times
            String baseTestOutputDir = resultPath + (resultPath.endsWith("/") ? "" : "/") + System.currentTimeMillis() + "_" + testType + "_" + test + "/";

            String yamlConf = sparkTest.toYaml();
            String yamlPath = baseTestOutputDir + "testConfig.yml";
            SparkUtils.writeStringToFile(yamlPath, yamlConf, sc);

            StringBuilder sb = new StringBuilder();
            sb.append("Data generate/export time: ").append(endGenerateExport - startGenerateExport).append("\n");
            sb.append("Fit time: ").append(endFit - startFit).append("\n");
            sb.append("Spark default parallelism: ").append(sc.defaultParallelism()).append("\n");
            int nParamsActual = sparkTest.getNumParams();
            sb.append("Actual # parameters: ").append(nParamsActual).append("\n");
            double paramsSizeMB = 4 * nParamsActual / (1024.0 * 1024.0); //Float: 4 bytes per element
            sb.append("Parameters size (MB): ").append(String.format("%.3f",paramsSizeMB)).append("\n");
            DataSet dsTemp = sparkTest.getSyntheticDataSet();
            int dataSetSizeNumElements = dsTemp.getFeatureMatrix().length() + dsTemp.getLabels().length();
            if(dsTemp.getFeaturesMaskArray() != null) dataSetSizeNumElements += dsTemp.getFeaturesMaskArray().length();
            if(dsTemp.getLabelsMaskArray() != null) dataSetSizeNumElements += dsTemp.getLabelsMaskArray().length();
            sb.append("Number values per DataSet object: ").append(dataSetSizeNumElements).append("\n");
            double sizeMB = 4 * dataSetSizeNumElements / (1024.0 * 1024.0); //Float: 4 bytes per element
            sb.append("DataSet data size (MB): ").append(String.format("%.3f",sizeMB)).append("\n");
            String testStats = sb.toString();
            String timesOutputPath = baseTestOutputDir + "testStats.txt";
            SparkUtils.writeStringToFile(timesOutputPath, testStats, sc);


            String statsHtmlOutputPath = baseTestOutputDir + "SparkStats.html";
            SparkTrainingStats sts = net.getSparkTrainingStats();
            StatsUtils.exportStatsAsHtml(sts, statsHtmlOutputPath, sc);

            String statsRawOutputPath = baseTestOutputDir + "SparkStats.txt";
            String rawStats = sts.statsAsString();
            SparkUtils.writeStringToFile(statsRawOutputPath, rawStats, sc);


            //Finally: where necessary, delete the temporary data
            if(sparkTest.getDataLoadingMethod() == DataLoadingMethod.SparkBinaryFiles ){
                log.info("Deleting temporary files at path: {}", tempPath);
                fileSystem.delete(new Path(tempPath), true);
            }

            test++;
            log.info("Completed test {} of {}. Total test time: {} ms", test, testsToRun.size(), (System.currentTimeMillis()-testStartTime));
        }


        log.info("***** COMPLETE *****");
    }

}

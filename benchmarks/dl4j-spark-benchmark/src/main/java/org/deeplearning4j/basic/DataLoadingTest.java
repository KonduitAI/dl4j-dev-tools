package org.deeplearning4j.basic;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;

/**
 * Performance debugging: data loading etc
 *
 * This test attempts to measure the overhead of generating and loading data in some different ways on Spark.
 */
public class DataLoadingTest {

    public static Logger log = LoggerFactory.getLogger(DataLoadingTest.class);

    @Parameter(names = "-useSparkLocal", description = "Whether to use spark local (if false: use spark submit)", arity = 1)
    protected boolean useSparkLocal = false;

    @Parameter(names="-numDataSetObjects", description = "Number of test files (DataSet objects)")
    protected int numTestFiles = 10000;

    @Parameter(names="-tempPath", description = "Path to the test directory (typically HDFS), in which to generate data", required = true)
    protected String tempPath;

    @Parameter(names="-resultPath", description = "Path to the base output directory. Results will be placed in a subdirectory. For example, HDFS or S3", required = true)
    protected String resultPath;


    public static void main(String[] args) throws Exception {
        new DataLoadingTest().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
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

        String launchArgsPath = resultPath + (resultPath.endsWith("/") ? "" : "/") + System.currentTimeMillis() + "_launchConf.txt";
        //Log the launch configuration
        String f = "%-40s\t%s\n";
        StringBuilder lp = new StringBuilder();
        lp.append("Launching job with args:\n");
        lp.append(String.format(f,"launchArgsPath",launchArgsPath));
        lp.append(String.format(f,"useSparkLocal",useSparkLocal));
        lp.append(String.format(f,"numDataSetObjects",numTestFiles));
        lp.append(String.format(f,"tempPath",tempPath));
        lp.append(String.format(f,"resultPath",resultPath));
        log.info(lp.toString());

        SparkConf conf = new SparkConf();
        conf.setAppName("DataLoadingTest");
        if(useSparkLocal) conf.setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //Write launch args to file:
        SparkUtils.writeStringToFile(launchArgsPath,lp.toString(),sc);  //Write a copy of  the launch arguments to file

        Configuration config = new Configuration();
//        conf.set("fs.hdfs.impl",org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());
//        conf.set("fs.file.impl",org.apache.hadoop.fs.LocalFileSystem.class.getName());
        FileSystem  hdfs = FileSystem.get(URI.create(tempPath), config);
        boolean exists = hdfs.exists(new Path(tempPath));
        if(exists){
            log.info("Temporary directory exists; attempting to delete. {}", tempPath);
            hdfs.delete(new Path(tempPath), true);
        }



        //Create some basic data
        List<Integer> list = new ArrayList<>();
        for( int i=0; i<numTestFiles; i++ ){
            list.add(i);
        }



        long startParallelize = System.currentTimeMillis();
        JavaRDD<Integer> rddi = sc.parallelize(list);
        long endParallelize = System.currentTimeMillis();

        JavaRDD<byte[]> rddDataSet = rddi.map(new MapIntDataSetFunction());
        rddDataSet.cache();

        long startCount = System.currentTimeMillis();
        rddDataSet.count();
        long endCount = System.currentTimeMillis();

        long startExport = System.currentTimeMillis();
        rddDataSet.foreachPartition(new ByteArrayExportFunction(new URI(tempPath)));
        long endExport = System.currentTimeMillis();


        JavaPairRDD<String,PortableDataStream> pds = sc.binaryFiles(tempPath);
        pds.cache();

        int pdsNPartitions = pds.partitions().size();

        long startCountPDS = System.currentTimeMillis();
        pds.count();
        long endCountPDS = System.currentTimeMillis();


        //Load data again but coalesce first
        pds = sc.binaryFiles(tempPath);
        pds = pds.coalesce(sc.defaultParallelism());

        long startCountPDS2 = System.currentTimeMillis();
        pds.count();
        long endCountPDS2 = System.currentTimeMillis();

        //Load data again but repartition first
        pds = sc.binaryFiles(tempPath);
        pds = pds.repartition(sc.defaultParallelism());

        long startCountPDS3 = System.currentTimeMillis();
        pds.count();
        long endCountPDS3 = System.currentTimeMillis();


        JavaPairRDD<String,String> asString = sc.wholeTextFiles(tempPath);
        long startText = System.currentTimeMillis();
        asString.count();
        long endText = System.currentTimeMillis();


        long startList = System.currentTimeMillis();
        RemoteIterator<LocatedFileStatus> iter = FileSystem.get(sc.hadoopConfiguration()).listFiles(new Path(tempPath), true);
        long endList = System.currentTimeMillis();

        List<URI> list2 = new ArrayList<>();
        long startProcessList = System.currentTimeMillis();
        while(iter.hasNext()){
            LocatedFileStatus lfs = iter.next();
            URI uri = lfs.getPath().toUri();
            list2.add(uri);
        }
        long endProcessList = System.currentTimeMillis();

        StringBuilder sb = new StringBuilder();
        sb.append("-numDataSetObjects = ").append(numTestFiles).append("\n");
        sb.append("-tempPath = ").append(tempPath).append("\n");
        sb.append("-resultPath = ").append(resultPath).append("\n");
        sb.append("Default parallelism: ").append(sc.defaultParallelism()).append("\n");
        sb.append("Parallelize time: ").append(endParallelize-startParallelize).append("\n");
        sb.append("Count + create data times: ").append(endCount - startCount).append("\n");
        sb.append("Export time: ").append(endExport - startExport).append("\n");
        sb.append("PDS initial number of partitions: ").append(pdsNPartitions).append("\n");
        sb.append("PDS count 1: ").append(endCountPDS-startCountPDS).append("\n");
        sb.append("PDS count (w/ coalesce): ").append(endCountPDS2-startCountPDS2).append("\n");
        sb.append("PDS count (w/ repartition): ").append(endCountPDS3-startCountPDS3).append("\n");
        sb.append("WholeTextFiles count: ").append(endText-startText).append("\n");
        sb.append("List files time: ").append(endList-startList).append("\n");
        sb.append("Process files list: ").append(endProcessList - startProcessList).append("\n");

        String str = sb.toString();
        String filename = "DataLoadingTest_" + System.currentTimeMillis() + "_results.txt";
        String outPath = resultPath + (resultPath.endsWith("/") ? "" : "/" ) + filename;

        SparkUtils.writeStringToFile(outPath, str, sc);

        sc.stop();
        Thread.sleep(4000);


        log.info("----- DONE -----");
    }


    private static class MapIntDataSetFunction implements Function<Integer,byte[]> {
        @Override
        public byte[] call(Integer integer) throws Exception {
            return new byte[32*1000*2*4];
        }
    }

}

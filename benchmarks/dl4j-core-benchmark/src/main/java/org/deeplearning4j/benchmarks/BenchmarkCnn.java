package org.deeplearning4j.benchmarks;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.ArrayUtils;
import org.deeplearning4j.datasets.iterator.impl.BenchmarkDataSetIterator;
import org.deeplearning4j.models.ModelSelector;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.utils.DTypeUtils;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Map;

/**
 * Benchmarks popular CNN models using the CIFAR-10 dataset.
 */
@Slf4j
public class BenchmarkCnn extends BaseBenchmark {

    public static boolean EXIT_ON_COMPLETION = true;

    // values to pass in from command line when compiled, esp running remotely
    @Option(name = "--modelType", usage = "Model type (e.g. ALEXNET, VGG16, RESNET50, or CNN).", aliases = "-model")
    public static ModelType modelType = ModelType.RESNET50PRE;
    @Option(name="--numLabels",usage="Train batch size.",aliases = "-labels")
    public static int numLabels = 1000;
    @Option(name="--totalIterations",usage="Train batch size.",aliases = "-iterations")
    public static int totalIterations = 200;
    @Option(name="--batchSize",usage="Train batch size.",aliases = "-batch")
    public static int batchSize = 32;
    @Option(name="--gcWindow",usage="Set Garbage Collection window in milliseconds.",aliases = "-gcwindow")
    public static int gcWindow = 5000;
    @Option(name="--profile",usage="Run profiler and print results",aliases = "-profile")
    public static boolean profile = false;
    @Option(name="--cacheMode",usage="Cache mode setting for net")
    public static CacheMode cacheMode = CacheMode.DEVICE;
    @Option(name="--workspaceMode", usage="Workspace mode for net")
    public static WorkspaceMode workspaceMode = WorkspaceMode.SINGLE;   //Single == ENABLED in 1.0.0-beta
    @Option(name="--updater", usage="Updater for net")
    public static Updater updater = Updater.NONE;
    @Option(name="--usePW", usage="Use parallel wrapper")
    public static boolean usePW = false;
    @Option(name="--pwNumThreads", usage="Number of ParallelWrappe threads. If set to -1, use number of GPUs")
    public static int pwNumThreads = -1;
    @Option(name="--pwAvgFreq", usage="Parallel Wrapper averaging frequency")
    public static int pwAvgFreq = 5;
    @Option(name="--pwPrefetchBuffer", usage="Parallel Wrapper averaging frequency")
    public static int pwPrefetchBuffer = 2;
    @Option(name="--datatype", usage="ND4J DataType - FLOAT, DOUBLE, HALF")
    public static String datatype = "FLOAT";
    @Option(name="--memoryListener", usage="Add MemoryReportingListener")
    public static boolean memoryListener = false;

    private String datasetName  = "SIMULATEDCNN";
    private int seed = 42;

    public void run(String[] args) throws Exception {
        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
            System.exit(1);
        }

        log.info("Using DataType {}", datatype);
        DTypeUtils.setDataType(datatype);

        log.info("Building models for "+modelType+"....");
        networks = ModelSelector.select(modelType, null, numLabels, seed, iterations, workspaceMode, cacheMode, updater);

        for (Map.Entry<ModelType, TestableModel> net : networks.entrySet()) {
            int[][] inputShape = net.getValue().metaData().getInputShape();
            String description = datasetName + " " + batchSize + "x" + inputShape[0][0] + "x" + inputShape[0][1] + "x" + inputShape[0][2];
            log.info("Selected: " + net.getKey().toString() + " " + description);

            log.info("Preparing benchmarks for {} iterations, {} labels, updater: {}, workspace: {}, cache mode: {}",
                    totalIterations, numLabels, updater, workspaceMode, cacheMode);

            int[] iterShape = ArrayUtils.addAll(new int[]{batchSize}, inputShape[0]);
            DataSetIterator iter = new BenchmarkDataSetIterator(iterShape, numLabels, totalIterations);

            new BaseBenchmark.Benchmark()
                    .net(net)
                    .description(description)
                    .numLabels(numLabels)
                    .batchSize(batchSize)
                    .seed(seed)
                    .datasetName(datasetName)
                    .iter(iter)
                    .modelType(modelType)
                    .profile(profile)
                    .gcWindow(gcWindow)
                    .occasionalGCFreq(0)
                    .usePW(usePW)
                    .pwNumThreads(pwNumThreads)
                    .pwAvgFreq(pwAvgFreq)
                    .pwPrefetchBuffer(pwPrefetchBuffer)
                    .memoryListener(memoryListener)
                    .execute();
        }

        if(EXIT_ON_COMPLETION) {
            System.exit(0);
        }
    }

    public static void main(String[] args) throws Exception {

//        // optimized for Titan X
//        CudaEnvironment.getInstance().getConfiguration()
//                .setMaximumBlockSize(768)
//                .setMinimumBlockSize(768);

        new BenchmarkCnn().run(args);
    }
}

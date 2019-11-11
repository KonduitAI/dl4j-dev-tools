package ai.skymind;

import ai.skymind.util.TimedScoreListener;
import ai.skymind.util.Utils;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicLong;

@Slf4j
public class DL4JTestRun {

    @Option(name = "--modelClass", usage = "Model class", required = true)
    public static String modelClass;
    @Option(name = "--dataClass", usage = "Data pipeline class", required = true)
    public static String dataClass;
    @Option(name = "--runtimeSec", usage = "Maximum runtime (seconds)")
    public static int runtimeSec = 3600;    //1 hour
    @Option(name = "--periodicGC", usage = "Periodic GC frequency (<= 0 is disabled - default)")
    public static int periodicGC = 0;
    @Option(name = "--useHelpers", usage = "Whether to use MKL-DNN/cuDNN or not")
    public static boolean useHelpers = false;
    @Option(name = "--maxIters", usage = "Maximum number of iterations")
    public static int maxIters = Integer.MAX_VALUE;

    public static void main(String[] args) throws Exception {
        new DL4JTestRun().run(args);
    }

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

        log.info("Model class: {}", modelClass);
        log.info("Data class: {}", dataClass);
        log.info("Runtime: {} seconds", runtimeSec);
        log.info("Periodic GC: {}", (periodicGC <= 0 ? "disabled" : periodicGC + " ms"));
        log.info("Use helpers: {}", useHelpers);

        if(!useHelpers){
            Utils.disableMKLDNN();
        }

        Nd4j.create(1);

        Utils.logMemoryConfig();
        AtomicLong[] bytes = Utils.startMemoryLoggingThread(30000);

        BenchmarkModel m = (BenchmarkModel) Class.forName(modelClass).newInstance();
        Pipeline p = (Pipeline) Class.forName(dataClass).newInstance();

        Model model = m.getModel();
        boolean mln = model instanceof MultiLayerNetwork;
        model.setListeners(new TimedScoreListener(60000));

        log.info("Num params: {}", model.numParams());

        if(periodicGC > 0) {
            Nd4j.getMemoryManager().togglePeriodicGc(true);
            Nd4j.getMemoryManager().setAutoGcWindow(periodicGC);
        }

        if(!useHelpers){
            Utils.removeHelpers(model);
        }

        long start = System.currentTimeMillis();
        long end = start + runtimeSec * 1000L;
        switch (p.type()){
            case DATASET_ITERATOR:
                DataSetIterator iter = p.getIterator();
                while(System.currentTimeMillis() < end){    //TODO eventually add cutting short for iterator
                    if(mln){
                        ((MultiLayerNetwork)model).fit(iter);
                        if(((MultiLayerNetwork) model).getIterationCount() >= maxIters)
                            break;  //TODO break before epoch is done if iter limit is hit
                    } else {
                        ((ComputationGraph)model).fit(iter);
                        if(((ComputationGraph) model).getIterationCount() >= maxIters)
                            break;  //TODO break before epoch is done if iter limit is hit
                    }
                }
                break;
            case MDS_ITERATOR:
                MultiDataSetIterator mdsIter = p.getMdsIterator();
                while(System.currentTimeMillis() < end){    //TODO eventually add cutting short for iterator
                    if(mln){
                        ((MultiLayerNetwork)model).fit(mdsIter);
                        if(((MultiLayerNetwork) model).getIterationCount() >= maxIters)
                            break;  //TODO break before epoch is done if iter limit is hit
                    } else {
                        ((ComputationGraph)model).fit(mdsIter);
                        if(((ComputationGraph) model).getIterationCount() >= maxIters)
                            break;  //TODO break before epoch is done if iter limit is hit
                    }
                }
                break;
            case INDARRAYS:
                int iterCount = 0;
                long lastReport = System.currentTimeMillis();
                while(System.currentTimeMillis() < end && iterCount < maxIters){
                    INDArray[] next = p.getFeatures();
                    if(mln){
                        ((MultiLayerNetwork)model).output(next[0]);
                    } else {
                        ((ComputationGraph)model).output(next);
                    }
                    if(++iterCount % 1000 == 0 && System.currentTimeMillis() > lastReport + 60000L ){
                        log.info("Num iters: {}", iterCount);
                        lastReport = System.currentTimeMillis();
                    }
                }
                break;
            default:
                throw new RuntimeException("Unknown data type: " + p.type());
        }

        log.info("Completed. Max observed bytes = {}, max physical bytes = {}", bytes[0].get(), bytes[1].get());
    }

}

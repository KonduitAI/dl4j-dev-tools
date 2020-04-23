package org.deeplearning4j.listeners;

/**
 * Created by justin on 3/25/17.
 */

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Variation of PerformanceListener that allows collection of statistics.
 *
 * @author Justin Long (@crockpotveggies)
 */
public class BenchmarkListener implements TrainingListener {
    private static final Logger logger = LoggerFactory.getLogger(org.deeplearning4j.optimize.listeners.PerformanceListener.class);
    private ThreadLocal<Double> samplesPerSec = new ThreadLocal<>();
    private ThreadLocal<Double> batchesPerSec = new ThreadLocal<>();
    private ThreadLocal<Long> lastTime = new ThreadLocal<>();
    private ThreadLocal<AtomicLong> iterationCount = new ThreadLocal<>();

    private BenchmarkReport benchmarkReport;

    private String device;

    public BenchmarkListener(BenchmarkReport benchmarkReport) {
        this.benchmarkReport = benchmarkReport;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        // we update lastTime on every iteration
        // just to simplify things
        boolean isFirstIter = false;
        if (lastTime.get() == null) {
            lastTime.set(System.currentTimeMillis());
            isFirstIter = true;
        }

        if (samplesPerSec.get() == null)
            samplesPerSec.set(0.0);

        if (batchesPerSec.get() == null)
            batchesPerSec.set(0.0);

        if (iterationCount.get() == null)
            iterationCount.set(new AtomicLong(0));

        if(!isFirstIter) {

            long currentTime = System.currentTimeMillis();

            long timeSpent = currentTime - lastTime.get();
            float timeSec = timeSpent / 1000f;

            INDArray input;
            if (model instanceof ComputationGraph) {
                // for comp graph (with multidataset
                ComputationGraph cg = (ComputationGraph) model;
                INDArray[] inputs = cg.getInputs();

                if (inputs != null && inputs.length > 0)
                    input = inputs[0];
                else
                    input = model.input();
            } else {
                input = model.input();
            }

            long numSamples = input.size(0);

            samplesPerSec.set((double) (numSamples / timeSec));
            batchesPerSec.set((double) (1 / timeSec));

            long tId = Thread.currentThread().getId();
            benchmarkReport.addIterations(tId, 1);
            benchmarkReport.addIterationTime(tId, timeSpent);
            if (!Double.isInfinite(samplesPerSec.get())) {
                benchmarkReport.addSamplesSec(tId, samplesPerSec.get());
            }
            if (!Double.isInfinite(batchesPerSec.get())) {
                benchmarkReport.addBatchesSec(tId, batchesPerSec.get());
            }
        }

        lastTime.set(System.currentTimeMillis());
    }

    @Override
    public void onEpochStart(Model model) {

    }

    @Override
    public void onEpochEnd(Model model) {

    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {

    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {

    }

    @Override
    public void onGradientCalculation(Model model) {

    }

    @Override
    public void onBackwardPass(Model model) {

    }
}

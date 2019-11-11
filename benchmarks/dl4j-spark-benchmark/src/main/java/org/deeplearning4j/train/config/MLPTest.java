package org.deeplearning4j.train.config;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by Alex on 23/07/2016.
 */
public class MLPTest extends BaseSparkTest {

    protected MLPTest(Builder builder) {
        super(builder);
    }

    @Override
    public MultiLayerConfiguration getConfiguration() {
        //With 3 layers, same input/output size, we have:
        //  L = layer size
        //  D = input/output size
        //  X = total number of parameters
        // (D+1)L + (L+1)L + (L+1)D = X
        // Layer size wanted for X parameters:
        // L = -(D+1) + sqrt((D+1)^2 - D - X)

        int layerSize = calcLayerSize();

        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP)
                .learningRate(0.1)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(dataSize).nOut(layerSize).build())
                .layer(1, new DenseLayer.Builder().nIn(layerSize).nOut(layerSize).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")
                        .nIn(layerSize).nOut(dataSize).build())
                .pretrain(false).backprop(true)
                .build();

        return mlc;
    }

    private int calcLayerSize() {
        int d = dataSize;
        int x = paramsSize;
        double l = -(d + 1) + Math.sqrt((d + 1) * (d + 1) - d + x);
        return (int) Math.ceil(l);
    }


    @Override
    public DataSet getSyntheticDataSet() {
        INDArray labels = Nd4j.zeros(minibatchSizePerWorker, dataSize);
        for (int i = 0; i < minibatchSizePerWorker; i++) {
            labels.putScalar(i, rng.nextInt(dataSize), 1.0);
        }
        return new DataSet(Nd4j.rand(minibatchSizePerWorker, dataSize), labels);
    }

    public static class Builder extends BaseSparkTest.Builder<Builder>{

        public MLPTest build(){
            return new MLPTest(this);
        }

    }
}

package org.deeplearning4j.train.config;

import lombok.Getter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by Alex on 23/07/2016.
 */
@Getter
public class RNNTest extends BaseSparkTest {

    protected int timeSeriesLength;
    protected boolean useTruncatedBPTT;
    protected int tbpttLength;

    public RNNTest(Builder builder){
        super(builder);

        this.timeSeriesLength = builder.timeSeriesLength;
        this.useTruncatedBPTT = builder.useTruncatedBPTT;
        this.tbpttLength = builder.tbpttLength;
    }


    @Override
    public MultiLayerConfiguration getConfiguration() {
        //3 layers, with same input/output size, we have:
        // L = LSTM layer size
        // D = input/output size
        // X = total number of parameters

        //For GravesLSTM layers:
//        int nParams = nLast * (4*nL)   //"input" weights
//                + nL * (4 * nL + 3) //recurrent weights
//                + 4*nL;             //bias

        //First layer: 4DL + L(4L+3) + 4L
        //Second layer: 4L^2 + L(4L+3) + 4L
        //Third layer: LD + D
        //To find L, solve: 12L^2 + (14+5D)L + (D-X) = 0
        //1/24 * (-(14+5D) + sqrt((14+5D)^2 - 48*(D-X))

        double l = 1/24.0 * (-(14+5*dataSize) + Math.sqrt((14+5*dataSize)*(14+5*dataSize) - 48*(dataSize - paramsSize)));

        int layerSize = (int)Math.ceil(l);

        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP)
                .learningRate(0.1)
                .activation("tanh")
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(dataSize).nOut(layerSize).build())
                .layer(1, new GravesLSTM.Builder().nIn(layerSize).nOut(layerSize).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")
                        .nIn(layerSize).nOut(dataSize).build())
                .pretrain(false).backprop(true)
                .backpropType(useTruncatedBPTT ? BackpropType.TruncatedBPTT : BackpropType.Standard)
                .tBPTTBackwardLength(tbpttLength).tBPTTForwardLength(tbpttLength)
                .build();

        return mlc;
    }

    @Override
    public DataSet getSyntheticDataSet() {
        INDArray labels = Nd4j.zeros(minibatchSizePerWorker, dataSize, timeSeriesLength);
        for (int i = 0; i < minibatchSizePerWorker; i++) {
            for( int j=0; j<timeSeriesLength; j++ ) {
                labels.putScalar(i, rng.nextInt(dataSize), j, 1.0);
            }
        }
        return new DataSet(Nd4j.rand(minibatchSizePerWorker, dataSize, timeSeriesLength), labels);
    }


    public static class Builder extends BaseSparkTest.Builder<Builder> {
        protected int timeSeriesLength = 100;
        protected boolean useTruncatedBPTT = true;
        protected int tbpttLength = 50;

        public Builder timeSeriesLength(int timeSeriesLength){
            this.timeSeriesLength = timeSeriesLength;
            return this;
        }

        public Builder useTruncatedBPTT(boolean useTruncatedBPTT){
            this.useTruncatedBPTT = useTruncatedBPTT;
            return this;
        }

        public Builder tbpttLength(int tbpttLength){
            this.tbpttLength = tbpttLength;
            return this;
        }

        public RNNTest build(){
            return new RNNTest(this);
        }
    }
}

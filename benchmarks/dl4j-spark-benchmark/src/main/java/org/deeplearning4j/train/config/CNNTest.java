package org.deeplearning4j.train.config;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Alex on 23/07/2016.
 */
public class CNNTest extends BaseSparkTest {

    protected CNNTest(Builder builder) {
        super(builder);
    }

    @Override
    public MultiLayerConfiguration getConfiguration() {
        //With 3 layers, same input/output size, we have:
        //  L = layer size
        //  D = input/output size
        //  X = total number of parameters

        //Convolution layer number of parameters: nIn * nOut * kernel[0] * kernel[1] + nOut;
        //Dense layer parameters: depends on image size...

        throw new RuntimeException("Not yet implemented");
    }


    @Override
    public DataSet getSyntheticDataSet() {
        INDArray labels = Nd4j.zeros(minibatchSizePerWorker, dataSize);
        for (int i = 0; i < minibatchSizePerWorker; i++) {
            labels.putScalar(i, rng.nextInt(dataSize), 1.0);
        }
        return new DataSet(Nd4j.rand(minibatchSizePerWorker, dataSize*dataSize*3), labels);     //width * height * input depth
    }

    public static class Builder extends BaseSparkTest.Builder<Builder>{

        public CNNTest build(){
            return new CNNTest(this);
        }

    }
}

package org.deeplearning4j.models.cnn;

import org.deeplearning4j.models.ModelMetaData;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * VGG-16
 */
public class VGG16 implements TestableModel {

    private int[] inputShape = new int[]{3,224,224};
    private int numLabels;
    private long seed;
    private int iterations;
    private WorkspaceMode workspaceMode;
    private CacheMode cacheMode;
    private Updater updater;

    public VGG16(int numLabels, long seed, int iterations, WorkspaceMode workspaceMode, CacheMode cacheMode, Updater updater) {
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
        this.workspaceMode = workspaceMode;
        this.cacheMode = cacheMode;
        this.updater = updater;
    }

    public MultiLayerConfiguration conf() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NONE)
                .activation(Activation.RELU)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .cacheMode(cacheMode)
                .list()
                .layer(0, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nIn(inputShape[0])
                        .nOut(64)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(1, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(64)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(2, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(3, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(128)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(4, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(128)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(5, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build())
                .layer(6, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(256)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(7, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(256)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(8, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(256)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(9, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(10, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(11, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(12, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(13, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(14, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(15, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(16, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1).padding(1, 1)
                        .nOut(512)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .build())
                .layer(17, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
//                .layer(18, new DenseLayer.Builder().nOut(4096).dropOut(0.5)
//                        .build())
//                .layer(19, new DenseLayer.Builder().nOut(4096).dropOut(0.5)
//                        .build())
                .layer(18, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX) // radial basis function required
                        .build())
                .setInputType(InputType.convolutionalFlat(inputShape[2],inputShape[1],inputShape[0]))
                .build();

        return conf;
    }

    public MultiLayerNetwork init(){
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;
    }

    public ModelMetaData metaData(){
        return new ModelMetaData(
            new int[][]{inputShape},
            1,
            ModelType.CNN
        );
    }

}
package org.deeplearning4j.models.cnn;

import org.deeplearning4j.VersionSpecificModels;
import org.deeplearning4j.models.ModelMetaData;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DebugCNN implements TestableModel {
    private int[] inputShape = new int[] { 3, 512, 512 };

    private WorkspaceMode workspaceMode;
    private CacheMode cacheMode;
    private Updater updater;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    public DebugCNN(WorkspaceMode workspaceMode, CacheMode cacheMode, Updater updater) {
        this.cacheMode = cacheMode;
        this.workspaceMode = workspaceMode;
        this.updater = updater;
    }


    public ComputationGraph init(){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(updater.getIUpdaterWithDefaultConfig())
                .convolutionMode(ConvolutionMode.Same)
                .graphBuilder()
                .addInputs("in")
                .addLayer("l0", new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(2,2)
                        .activation(Activation.RELU)
                        .nOut(128)
                        .build(), "in")
                .addLayer("l1", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build(), "l0")
                .addLayer("l2", new OutputLayer.Builder()
                        .nOut(1000).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build(), "l1")
                .setOutputs("l2")
                .setInputTypes(InputType.convolutional(inputShape[1], inputShape[2], inputShape[0]))
                .build();
        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();
        return cg;
    }

    public ModelMetaData metaData(){
        return new ModelMetaData(
                new int[][]{inputShape},
                1000,
                ModelType.CNN
        );
    }

}

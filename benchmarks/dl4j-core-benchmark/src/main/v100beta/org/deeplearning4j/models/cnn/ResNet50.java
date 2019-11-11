package org.deeplearning4j.models.cnn;


import org.deeplearning4j.models.ModelMetaData;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * ResNet50
 *
 * Dl4j's implementation of ResNet50
 *
 * References: TODO
 */
public class ResNet50  implements TestableModel {
    private int[] inputShape = new int[] { 3, 224, 224 };
    private int numLabels = 1000;
    private long seed = 42;
    private int iterations = 90;
    private WorkspaceMode workspaceMode;
    private CacheMode cacheMode;
    private Updater updater;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    public ResNet50(int numLabels, long seed, int iterations, WorkspaceMode workspaceMode, CacheMode cacheMode, Updater updater) {
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
        this.cacheMode = cacheMode;
        this.workspaceMode = workspaceMode;
        this.updater = updater;
    }

    private void identityBlock(ComputationGraphConfiguration.GraphBuilder graph, int[] kernelSize, int[] filters,
                               String stage, String block, String input) {
        String convName = "res" + stage + block + "_branch";
        String batchName = "bn" + stage + block + "_branch";
        String activationName = "act" + stage + block + "_branch";
        String shortcutName = "short" + stage + block + "_branch";

        graph.addLayer(convName + "2a",
                new ConvolutionLayer.Builder(new int[] {1, 1}).nOut(filters[0]).cudnnAlgoMode(cudnnAlgoMode)
                        .build(),
                input)
                .addLayer(batchName + "2a", new BatchNormalization(), convName + "2a")
                .addLayer(activationName + "2a",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        batchName + "2a")

                .addLayer(convName + "2b", new ConvolutionLayer.Builder(kernelSize).nOut(filters[1])
                                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).convolutionMode(ConvolutionMode.Same).build(),
                        activationName + "2a")
                .addLayer(batchName + "2b", new BatchNormalization(), convName + "2b")
                .addLayer(activationName + "2b",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        batchName + "2b")

                .addLayer(convName + "2c",
                        new ConvolutionLayer.Builder(new int[] {1, 1}).nOut(filters[2])
                                .cudnnAlgoMode(cudnnAlgoMode).build(),
                        activationName + "2b")
                .addLayer(batchName + "2c", new BatchNormalization(), convName + "2c")

                .addVertex(shortcutName, new ElementWiseVertex(ElementWiseVertex.Op.Add), batchName + "2c",
                        input)
                .addLayer(convName, new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        shortcutName);
    }

    private void convBlock(ComputationGraphConfiguration.GraphBuilder graph, int[] kernelSize, int[] filters,
                           String stage, String block, String input) {
        convBlock(graph, kernelSize, filters, stage, block, new int[] {2, 2}, input);
    }

    private void convBlock(ComputationGraphConfiguration.GraphBuilder graph, int[] kernelSize, int[] filters,
                           String stage, String block, int[] stride, String input) {
        String convName = "res" + stage + block + "_branch";
        String batchName = "bn" + stage + block + "_branch";
        String activationName = "act" + stage + block + "_branch";
        String shortcutName = "short" + stage + block + "_branch";

        graph.addLayer(convName + "2a", new ConvolutionLayer.Builder(new int[] {1, 1}, stride).nOut(filters[0]).build(),
                input)
                .addLayer(batchName + "2a", new BatchNormalization(), convName + "2a")
                .addLayer(activationName + "2a",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        batchName + "2a")

                .addLayer(convName + "2b",
                        new ConvolutionLayer.Builder(kernelSize).nOut(filters[1])
                                .convolutionMode(ConvolutionMode.Same).build(),
                        activationName + "2a")
                .addLayer(batchName + "2b", new BatchNormalization(), convName + "2b")
                .addLayer(activationName + "2b",
                        new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        batchName + "2b")

                .addLayer(convName + "2c",
                        new ConvolutionLayer.Builder(new int[] {1, 1}).nOut(filters[2]).build(),
                        activationName + "2b")
                .addLayer(batchName + "2c", new BatchNormalization(), convName + "2c")

                // shortcut
                .addLayer(convName + "1",
                        new ConvolutionLayer.Builder(new int[] {1, 1}, stride).nOut(filters[2]).build(),
                        input)
                .addLayer(batchName + "1", new BatchNormalization(), convName + "1")


                .addVertex(shortcutName, new ElementWiseVertex(ElementWiseVertex.Op.Add), batchName + "2c",
                        batchName + "1")
                .addLayer(convName, new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        shortcutName);
    }

    public ComputationGraphConfiguration.GraphBuilder graphBuilder() {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .activation(Activation.IDENTITY)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
//                .weightInit(new TruncatedNormalDistribution(0.0, 0.5))
                .weightInit(WeightInit.XAVIER)  //Use enum for 0.9.1 compatibility; doesn't impact performance...
                .l1(1e-7)
                .l2(5e-5)
                .miniBatch(true)
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .cudnnAlgoMode(cudnnAlgoMode)
                .convolutionMode(ConvolutionMode.Truncate)
                .graphBuilder();


        graph.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
                // stem
                .addLayer("stem-zero", new ZeroPaddingLayer.Builder(3, 3).build(), "input")
                .addLayer("stem-cnn1",
                        new ConvolutionLayer.Builder(new int[] {7, 7}, new int[] {2, 2}).nOut(64)
                                .build(),
                        "stem-zero")
                .addLayer("stem-batch1", new BatchNormalization(), "stem-cnn1")
                .addLayer("stem-act1", new ActivationLayer.Builder().activation(Activation.RELU).build(),
                        "stem-batch1")
                .addLayer("stem-maxpool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,
                        new int[] {3, 3}, new int[] {2, 2}).build(), "stem-act1");

        convBlock(graph, new int[] {3, 3}, new int[] {64, 64, 256}, "2", "a", new int[] {2, 2}, "stem-maxpool1");
        identityBlock(graph, new int[] {3, 3}, new int[] {64, 64, 256}, "2", "b", "res2a_branch");
        identityBlock(graph, new int[] {3, 3}, new int[] {64, 64, 256}, "2", "c", "res2b_branch");

        convBlock(graph, new int[] {3, 3}, new int[] {128, 128, 512}, "3", "a", "res2c_branch");
        identityBlock(graph, new int[] {3, 3}, new int[] {128, 128, 512}, "3", "b", "res3a_branch");
        identityBlock(graph, new int[] {3, 3}, new int[] {128, 128, 512}, "3", "c", "res3b_branch");
        identityBlock(graph, new int[] {3, 3}, new int[] {128, 128, 512}, "3", "d", "res3c_branch");

        convBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "a", "res3d_branch");
        identityBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "b", "res4a_branch");
        identityBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "c", "res4b_branch");
        identityBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "d", "res4c_branch");
        identityBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "e", "res4d_branch");
        identityBlock(graph, new int[] {3, 3}, new int[] {256, 256, 1024}, "4", "f", "res4e_branch");

        convBlock(graph, new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "a", "res4f_branch");
        identityBlock(graph, new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "b", "res5a_branch");
        identityBlock(graph, new int[] {3, 3}, new int[] {512, 512, 2048}, "5", "c", "res5b_branch");

        graph.addLayer("avgpool",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3}).build(),
                "res5c_branch")
                // TODO add flatten/reshape layer here
                .addLayer("output",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(numLabels).activation(Activation.SOFTMAX).build(),
                        "avgpool")
                .setOutputs("output");

        return graph;
    }

    public ComputationGraph init(){
        ComputationGraphConfiguration.GraphBuilder conf = graphBuilder();
        ComputationGraph network = new ComputationGraph(conf.build());
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
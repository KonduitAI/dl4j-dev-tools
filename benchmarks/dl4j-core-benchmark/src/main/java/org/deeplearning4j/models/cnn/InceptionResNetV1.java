package org.deeplearning4j.models.cnn;

import org.deeplearning4j.models.ModelMetaData;
import org.deeplearning4j.models.ModelType;
import org.deeplearning4j.models.TestableModel;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * A variant of the original FaceNet model that relies on embeddings and triplet loss.
 * Reference: https://arxiv.org/abs/1503.03832
 * Also based on the OpenFace implementation: http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf
 *
 * Revised and consolidated version by @crockpotveggies
 */
public class InceptionResNetV1 implements TestableModel {

    private int[] inputShape = new int[]{3,160,160};
    private long seed;
    private int iterations;
    private int numClasses;
    private WorkspaceMode workspaceMode;
    private CacheMode cacheMode;
    private Updater updater;

    public InceptionResNetV1(int outputNum, long seed, int iterations, WorkspaceMode workspaceMode, CacheMode cacheMode, Updater updater) {
      this.seed = seed;
      this.numClasses = outputNum;
      this.iterations = iterations;
      this.workspaceMode = workspaceMode;
      this.cacheMode = cacheMode;
      this.updater = updater;
    }

    public ComputationGraph init() {
        int embeddingSize = 128;
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder("input1");

        graph
            .addInputs("input1")
            .setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
            // Logits
            .addLayer("bottleneck", new DenseLayer.Builder().nIn(5376).nOut(embeddingSize).build(), "avgpool")
            // Embeddings
            .addVertex("embeddings", new L2NormalizeVertex(new int[]{1}, 1e-10), "bottleneck")
            // Output
            .addLayer("outputLayer", new CenterLossOutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .alpha(0.9).lambda(1e-4)
                .nIn(embeddingSize)
                .nOut(numClasses)
            .build(), "embeddings")
            .setOutputs("outputLayer");

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }

    public ComputationGraphConfiguration.GraphBuilder graphBuilder(String input) {

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .activation(Activation.RELU)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(updater)
            .weightInit(WeightInit.DISTRIBUTION)
//            .weightInit(new NormalDistribution(0.0, 0.5))
            .weightInit(WeightInit.XAVIER)  //Use enum for 0.9.1 compatibility; doesn't impact performance...
            .l2(5e-5)
            .miniBatch(true)
            .convolutionMode(ConvolutionMode.Truncate)
            .trainingWorkspaceMode(workspaceMode)
            .inferenceWorkspaceMode(workspaceMode)
            .cacheMode(cacheMode)
            .graphBuilder();


        graph
            // stem
            .addLayer("stem-cnn1", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(inputShape[0]).nOut(32).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), input)
            .addLayer("stem-batch1", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(32).nOut(32).build(), "stem-cnn1")
            .addLayer("stem-cnn2", new ConvolutionLayer.Builder(new int[]{3,3}).nIn(32).nOut(32).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "stem-batch1")
            .addLayer("stem-batch2", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(32).nOut(32).build(), "stem-cnn2")
            .addLayer("stem-cnn3", new ConvolutionLayer.Builder(new int[]{3,3}).convolutionMode(ConvolutionMode.Same).nIn(32).nOut(64).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "stem-batch2")
            .addLayer("stem-batch3", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(64).nOut(64).build(), "stem-cnn3")

            .addLayer("stem-pool4", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2}).build(), "stem-batch3")

            .addLayer("stem-cnn5", new ConvolutionLayer.Builder(new int[]{1,1}).nIn(64).nOut(80).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "stem-pool4")
            .addLayer("stem-batch5", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(80).nOut(80).build(), "stem-cnn5")
            .addLayer("stem-cnn6", new ConvolutionLayer.Builder(new int[]{3,3}).nIn(80).nOut(128).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "stem-batch5")
            .addLayer("stem-batch6", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(128).nOut(128).build(), "stem-cnn6")
            .addLayer("stem-cnn7", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(128).nOut(192).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "stem-batch6")
            .addLayer("stem-batch7", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(192).nOut(192).build(), "stem-cnn7");


        // 5xInception-resnet-A
        InceptionResNetHelper.inceptionV1ResA(graph, "resnetA", 5, 0.17, "stem-batch7");


        // Reduction-A
        graph
            // 3x3
            .addLayer("reduceA-cnn1", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(192).nOut(192).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "resnetA")
            .addLayer("reduceA-batch1", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(192).nOut(192).build(), "reduceA-cnn1")
            // 1x1 -> 3x3 -> 3x3
            .addLayer("reduceA-cnn2", new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(192).nOut(128).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "resnetA")
            .addLayer("reduceA-batch2", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(128).nOut(128).build(), "reduceA-cnn2")
            .addLayer("reduceA-cnn3", new ConvolutionLayer.Builder(new int[]{3,3}).convolutionMode(ConvolutionMode.Same).nIn(128).nOut(128).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "reduceA-batch2")
            .addLayer("reduceA-batch3", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(128).nOut(128).build(), "reduceA-cnn3")
            .addLayer("reduceA-cnn4", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(128).nOut(192).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "reduceA-batch3")
            .addLayer("reduceA-batch4", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(192).nOut(192).build(), "reduceA-cnn4")
            // maxpool
            .addLayer("reduceA-pool5", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2}).build(), "resnetA")
            // -->
            .addVertex("reduceA", new MergeVertex(), "reduceA-batch1", "reduceA-batch4", "reduceA-pool5");


        // 10xInception-resnet-B
        InceptionResNetHelper.inceptionV1ResB(graph, "resnetB", 10, 0.10, "reduceA");


        // Reduction-B
        graph
            // 3x3 pool
            .addLayer("reduceB-pool1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3,3}, new int[]{2,2}).build(), "resnetB")
            // 1x1 -> 3x3
            .addLayer("reduceB-cnn2", new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(576).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "resnetB")
            .addLayer("reduceB-batch1", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256).build(), "reduceB-cnn2")
            .addLayer("reduceB-cnn3", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(256).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "reduceB-batch1")
            .addLayer("reduceB-batch2", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256).build(), "reduceB-cnn3")
            // 1x1 -> 3x3
            .addLayer("reduceB-cnn4", new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(576).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "resnetB")
            .addLayer("reduceB-batch3", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256).build(), "reduceB-cnn4")
            .addLayer("reduceB-cnn5", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(256).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "reduceB-batch3")
            .addLayer("reduceB-batch4", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256).build(), "reduceB-cnn5")
            // 1x1 -> 3x3 -> 3x3
            .addLayer("reduceB-cnn6", new ConvolutionLayer.Builder(new int[]{1,1}).convolutionMode(ConvolutionMode.Same).nIn(576).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "resnetB")
            .addLayer("reduceB-batch5", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256).build(), "reduceB-cnn6")
            .addLayer("reduceB-cnn7", new ConvolutionLayer.Builder(new int[]{3,3}).convolutionMode(ConvolutionMode.Same).nIn(256).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "reduceB-batch5")
            .addLayer("reduceB-batch6", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256).build(), "reduceB-cnn7")
            .addLayer("reduceB-cnn8", new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{2,2}).nIn(256).nOut(256).cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE).build(), "reduceB-batch6")
            .addLayer("reduceB-batch7", new BatchNormalization.Builder(false).decay(0.995).eps(0.001).nIn(256).nOut(256).build(), "reduceB-cnn8")
            // -->
            .addVertex("reduceB", new MergeVertex(), "reduceB-pool1", "reduceB-batch2", "reduceB-batch4", "reduceB-batch7");


        // 10xInception-resnet-C
        InceptionResNetHelper.inceptionV1ResC(graph, "resnetC", 5, 0.20, "reduceB");

        // Average pooling
        graph.addLayer("avgpool", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{1,1}).build(), "resnetC");

        return graph;
    }

    public ModelMetaData metaData(){
        return new ModelMetaData(
                new int[][]{inputShape},
                1,
                ModelType.CNN
        );
    }

}
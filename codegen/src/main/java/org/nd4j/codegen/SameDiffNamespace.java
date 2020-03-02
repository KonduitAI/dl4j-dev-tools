package org.nd4j.codegen;

import org.nd4j.codegen.api.NamespaceOps;
import org.nd4j.codegen.ops.*;

public enum SameDiffNamespace {
    BITWISE,
    NEURALNETWORK,
    RANDOM,
    IMAGE,
    CNN,
    RNN,
    MATH,
    BASE,
    LOSS,
    VALIDATION,
    LINALG;


    public static SameDiffNamespace fromString(String in){
        switch (in.toLowerCase()){
            case "bitwise":
                return BITWISE;
            case "nn":
            case "neuralnetwork":
                return NEURALNETWORK;
            case "random":
                return RANDOM;
            case "math":
                return MATH;
            case "image":
                return IMAGE;
            case "cnn":
                return CNN;
            case "rnn":
                return RNN;
            case "base":
                return BASE;
            case "loss":
                return LOSS;
            case "validation":
                return VALIDATION;
            case "linalg":
                return LINALG;
            default:
                return null;
        }
    }

    public String javaClassName(){
        switch (this){
            case BITWISE:
                return "SDBitwise";
            case NEURALNETWORK:
                return "SDNN";
            case RANDOM:
                return "SDRandom";
            case IMAGE:
                return "SDImage";
            case CNN:
                return "SDCNN";
            case RNN:
                return "SDRNN";
            case MATH:
                return "SDMath";
            /*case BASE:
                return "SDOps";*/
            case LOSS:
                return "SDLoss";
            case VALIDATION:
                return "SDValidation";
            case LINALG:
                return "SDLinalg";
        }
        throw new IllegalStateException("No java class name defined for: " + this);
    }

    public NamespaceOps getNamespace(){
        switch (this){
            case BITWISE:
                return BitwiseKt.Bitwise();
            case RANDOM:
                return RandomKt.Random();
            case MATH:
                return MathKt.Math();
            case IMAGE:
                return ImageKt.SDImage();
            case CNN:
                return CNNKt.SDCNN();
            case RNN:
                return RNNKt.SDRNN();
            case NEURALNETWORK:
                return NeuralNetworkKt.NN();
            case BASE:
                return SDBaseOpsKt.SDBaseOps();
            case LOSS:
                return SDLossKt.SDLoss();
            case LINALG:
                return LinalgKt.Linalg();
        }
        throw new IllegalStateException("No namespace definition available for: " + this);
    }
}
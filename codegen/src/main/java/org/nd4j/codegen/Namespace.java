package org.nd4j.codegen;

import org.nd4j.codegen.api.NamespaceOps;
import org.nd4j.codegen.ops.BitwiseKt;
import org.nd4j.codegen.ops.NeuralNetworkKt;
import org.nd4j.codegen.ops.RandomKt;
import org.nd4j.codegen.ops.MathKt;
import org.nd4j.codegen.ops.ImageKt;


public enum Namespace {
    BITWISE,
    NEURALNETWORK,
    RANDOM,
    IMAGE,
    MATH;


    public static Namespace fromString(String in){
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
            default:
                return null;
        }
    }

    public String javaClassName(){
        switch (this){
            case BITWISE:
                return "NDBitwise";
            case NEURALNETWORK:
                return "NDNN";
            case RANDOM:
                return "NDRandom";
            case IMAGE:
                return "NDImage";
            case MATH:
                return "NDMath";
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
            case NEURALNETWORK:
                return NeuralNetworkKt.NN();
        }
        throw new IllegalStateException("No namespace definition available for: " + this);
    }
}

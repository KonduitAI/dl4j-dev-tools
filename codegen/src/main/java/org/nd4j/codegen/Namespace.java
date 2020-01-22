package org.nd4j.codegen;

import org.nd4j.codegen.api.NamespaceOps;
import org.nd4j.codegen.ops.*;

public enum Namespace {
    BITWISE,
    NEURALNETWORK,
    RANDOM,
    MATH,
    BASE;

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
            case "base":
                return BASE;
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
            case MATH:
                return "NDMath";
            case BASE:
                return "NDBase";
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
            case NEURALNETWORK:
                return NeuralNetworkKt.NN();
            case BASE:
                return SDBaseOpsKt.SDBaseOps();
        }
        throw new IllegalStateException("No namespace definition available for: " + this);
    }
}

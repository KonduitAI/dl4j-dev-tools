package org.nd4j.codegen;

import org.nd4j.codegen.api.NamespaceOps;
import org.nd4j.codegen.ops.BitwiseKt;
import org.nd4j.codegen.ops.NeuralNetworkKt;
import org.nd4j.codegen.ops.RandomKt;
import org.nd4j.codegen.ops.SDMathKt;

public enum Namespace {
    BITWISE,
    NEURALNETWORK,
    RANDOM,
    MATH;

    public static Namespace fromString(String in){
        switch (in.toLowerCase()){
            case "bitwise":
                return BITWISE;
            case "neuralnetwork":
                return NEURALNETWORK;
            case "random":
                return RANDOM;
            case "math":
                return MATH;
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
                return SDMathKt.SDMath();
        }
        throw new IllegalStateException("No namespace definition available for: " + this);
    }
}

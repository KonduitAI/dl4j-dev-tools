package org.deeplearning4j.train.functions;

import org.apache.spark.api.java.function.Function;

import java.util.Random;

/**
 * Created by Alex on 23/07/2016.
 */

public class GenerateCsvDataFunction implements Function<Integer,String> {

    private final int numValues;
    private final Random r;

    public GenerateCsvDataFunction(int numValues){
        this.numValues = numValues;
        this.r = new Random();
    }

    @Override
    public String call(Integer v1) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append(r.nextDouble());
        for( int i=1; i<numValues; i++ ){
            sb.append(",").append(r.nextDouble());
        }
        return sb.toString();
    }
}

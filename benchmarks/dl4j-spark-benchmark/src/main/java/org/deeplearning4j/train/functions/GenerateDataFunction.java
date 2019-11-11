package org.deeplearning4j.train.functions;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.train.config.SparkTest;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by Alex on 23/07/2016.
 */
@AllArgsConstructor
public class GenerateDataFunction implements Function<Integer,DataSet> {
    private final SparkTest sparkTest;

    @Override
    public DataSet call(Integer v1) throws Exception {
        return sparkTest.getSyntheticDataSet();
    }
}

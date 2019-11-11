package org.deeplearning4j.train.config;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.train.DataLoadingMethod;
import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;

/**
 * Created by Alex on 23/07/2016.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use= JsonTypeInfo.Id.NAME, include= JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = MLPTest.class, name = "MLPTest")
})
public interface SparkTest extends Serializable {

    MultiLayerConfiguration getConfiguration();

    int getNumDataSetObjects();

    int getNumParams();

    DataSet getSyntheticDataSet();

    int getMinibatchSizePerWorker();

    int getDataSize();

    int getParamsSize();

    boolean isSaveUpdater();

    Repartition getRepartition();

    RepartitionStrategy getRepartitionStrategy();

    int getWorkerPrefetchNumBatches();

    int getAveragingFrequency();

    DataLoadingMethod getDataLoadingMethod();

    CsvCompressionCodec getCsvCompressionCodec();

    int getCsvCoalesceSize();

    String toJson();

    String toYaml();

}

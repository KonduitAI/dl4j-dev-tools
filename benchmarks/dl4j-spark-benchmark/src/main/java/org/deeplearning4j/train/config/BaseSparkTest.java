package org.deeplearning4j.train.config;

import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.PropertyAccessor;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;
import lombok.Data;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.train.DataLoadingMethod;

import java.util.Random;

/**
 * Created by Alex on 23/07/2016.
 */
@Data
@JsonIgnoreProperties({"rng"})
public abstract class BaseSparkTest implements SparkTest {

    protected int minibatchSizePerWorker;
    protected int numDataSetObjects;
    protected int dataSize;
    protected int paramsSize;
    protected Random rng = new Random();
    protected boolean saveUpdater;
    protected Repartition repartition;
    protected RepartitionStrategy repartitionStrategy;
    protected int workerPrefetchNumBatches;
    protected int averagingFrequency;
    protected DataLoadingMethod dataLoadingMethod;
    protected CsvCompressionCodec csvCompressionCodec;
    protected int csvCoalesceSize;


    protected BaseSparkTest(Builder builder){
        this.numDataSetObjects = builder.numDataSetObjects;
        this.minibatchSizePerWorker = builder.minibatchSizePerWorker;
        this.dataSize = builder.dataSize;
        this.paramsSize = builder.paramsSize;
        this.saveUpdater = builder.saveUpdater;
        this.repartition = builder.repartition;
        this.repartitionStrategy = builder.repartitionStrategy;
        this.workerPrefetchNumBatches = builder.workerPrefetchNumBatches;
        this.averagingFrequency = builder.averagingFrequency;
        this.dataLoadingMethod = builder.dataLoadingMethod;
        this.csvCompressionCodec = builder.csvCompressionCodec;
        this.csvCoalesceSize = builder.csvCoalesceSize;
    }

    @Override
    public int getNumParams(){
        MultiLayerNetwork net = new MultiLayerNetwork(getConfiguration());
        net.init();
        return net.numParams();
    }

    @Override
    public String toJson(){
        try{
            return getObjectMapper(new JsonFactory()).writeValueAsString(this);
        } catch(Exception e){
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toYaml(){
        try{
            return getObjectMapper(new YAMLFactory()).writeValueAsString(this);
        } catch(Exception e){
            throw new RuntimeException(e);
        }
    }

    protected ObjectMapper getObjectMapper(JsonFactory factory) {
        ObjectMapper om = new ObjectMapper(factory);
        om.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        om.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        om.enable(SerializationFeature.INDENT_OUTPUT);
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.NONE);
        om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
        return om;
    }


    @SuppressWarnings("unchecked")
    public static abstract class Builder<T extends Builder<T>>{

        protected int numDataSetObjects = 2000;
        protected int minibatchSizePerWorker = 32;
        protected int dataSize = 128;
        protected int paramsSize = 100_000;
        protected boolean saveUpdater = true;
        protected Repartition repartition = Repartition.Always;
        protected RepartitionStrategy repartitionStrategy = RepartitionStrategy.SparkDefault;
        protected int workerPrefetchNumBatches = 2;
        protected int averagingFrequency = 5;
        protected DataLoadingMethod dataLoadingMethod = DataLoadingMethod.SparkBinaryFiles;
        protected CsvCompressionCodec csvCompressionCodec;
        protected int csvCoalesceSize;


        protected Random rng = new Random();

        public T numDataSetObjects(int numDataSetObjects){
            this.numDataSetObjects = numDataSetObjects;
            return (T)this;
        }

        public T minibatchSizePerWorker(int minibatchSizePerWorker){
            this.minibatchSizePerWorker = minibatchSizePerWorker;
            return (T)this;
        }

        public T paramsSize(int paramsSize){
            this.paramsSize = paramsSize;
            return (T)this;
        }

        public T dataSize(int dataSize){
            this.dataSize = dataSize;
            return (T)this;
        }

        public T saveUpdater(boolean saveUpdater){
            this.saveUpdater = saveUpdater;
            return (T)this;
        }

        public T repartition(Repartition repartition){
            this.repartition = repartition;
            return (T)this;
        }

        public T repartitionStrategy(RepartitionStrategy repartitionStrategy){
            this.repartitionStrategy = repartitionStrategy;
            return (T)this;
        }

        public T workerPrefetchNumBatches(int workerPrefetchNumBatches){
            this.workerPrefetchNumBatches = workerPrefetchNumBatches;
            return (T)this;
        }

        public T averagingFrequency(int averagingFrequency){
            this.averagingFrequency = averagingFrequency;
            return (T)this;
        }

        public T dataLoadingMethod(DataLoadingMethod dataLoadingMethod){
            this.dataLoadingMethod = dataLoadingMethod;
            return (T)this;
        }

        public T csvCompressionCodec(CsvCompressionCodec csvCompressionCodec){
            this.csvCompressionCodec = csvCompressionCodec;
            return (T)this;
        }

        public T csvCoalesceSize(int csvCoalesceSize){
            this.csvCoalesceSize = csvCoalesceSize;
            return (T)this;
        }

    }

}

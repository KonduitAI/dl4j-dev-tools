package org.deeplearning4j.train.misc;

import com.beust.jcommander.IStringConverter;
import org.deeplearning4j.spark.api.Repartition;
import org.deeplearning4j.spark.api.RepartitionStrategy;
import org.deeplearning4j.train.DataLoadingMethod;

/**
 * Created by Alex on 24/07/2016.
 */
public class EnumConverters {
    public static class RepartitionEnumConverter implements IStringConverter<Repartition>{
        @Override
        public Repartition convert(String s) {
            return Repartition.valueOf(s);
        }
    }

    public static class RepartitionStrategyEnumConverter implements IStringConverter<RepartitionStrategy>{
        @Override
        public RepartitionStrategy convert(String s) {
            return RepartitionStrategy.valueOf(s);
        }
    }

    public static class DataLoadingMethodEnumConverter implements IStringConverter<DataLoadingMethod>{
        @Override
        public DataLoadingMethod convert(String s) {
            return DataLoadingMethod.valueOf(s);
        }
    }
}

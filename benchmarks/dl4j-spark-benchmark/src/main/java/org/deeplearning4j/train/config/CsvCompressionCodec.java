package org.deeplearning4j.train.config;

import org.apache.hadoop.io.compress.*;

/**
 * Created by Alex on 12/08/2016.
 */
public enum CsvCompressionCodec {
    //Details from: http://www.slideshare.net/Hadoop_Summit/singh-kamat-june27425pmroom210c slide 7 and slide
    //Name          Splittable          Java/Native         Compression         Speed
    None,       //      Y               -                       None
    Deflate,    //      N               Y/Y                     High            Medium
    GZip,       //      N               Y/Y                     High            Medium
    BZip,       //      Y               Y/Y                     High            Low
    LZ4,        //      N               N/Y                     Low             High
    Snappy;     //      N               N/Y                     Low             High

    public Class<? extends CompressionCodec> getCodec(){

        switch (this){
            case None:
                return null;
            case Deflate:
                return DefaultCodec.class;
            case GZip:
                return GzipCodec.class;
            case BZip:
                return BZip2Codec.class;
            case LZ4:
                return Lz4Codec.class;
            case Snappy:
                return SnappyCodec.class;
            default:
                throw new RuntimeException();
        }
    }
}

package org.deeplearning4j.train;

/**
 * Created by Alex on 23/07/2016.
 */
public enum DataLoadingMethod {

    SparkBinaryFiles,
    Parallelize,
    StringPath,
    CSV,
    SequenceFile
}

package org.nd4j.codegen.api

enum class DataType {
    FLOATING_POINT, // Any floating point data type
    INT, // any integer data type
    NUMERIC, // any floating point or integer data type
    BOOL, // boolean data type
    DATA_TYPE, // tensor data type
    CONDITION, // A condition
    LOSS_REDUCE // Loss reduction mode
}
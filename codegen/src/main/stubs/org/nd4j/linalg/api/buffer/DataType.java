package org.nd4j.linalg.api.buffer;

public enum DataType {
    DOUBLE,
    FLOAT,
    HALF,
    LONG,
    INT,
    SHORT,
    UBYTE,
    BYTE,
    BOOL,
    UTF8,
    COMPRESSED,
    BFLOAT16,
    UINT16,
    UINT32,
    UINT64,
    UNKNOWN;

    public boolean isIntType(){
        return true;
    }

    public boolean isFPType(){
        return true;
    }
}

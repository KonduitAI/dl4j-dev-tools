package org.deeplearning4j.utils;

import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Method;

public class DTypeUtils {

    private DTypeUtils(){ }

    public static void setDataType(String datatype){
        Class<?> c = null;
        try{
            //Snashots
            c = Class.forName("org.nd4j.linalg.api.buffer.DataType");
        } catch (Throwable e){ }
        if(c == null) {
            try {
                //1.0.0-beta3 and earlier
                c = Class.forName("org.nd4j.linalg.api.buffer.DataBuffer$Type");
            } catch (Throwable e) { }
        }
        if(c == null){
            throw new RuntimeException("Could not find DataBuffer$Type");
        }
        try {
            Method m = Nd4j.class.getMethod("setDataType", c);
            m.invoke(null, Enum.valueOf((Class<Enum>) c, datatype));
        } catch (Exception e){
            throw new RuntimeException(e);
        }
    }

    public static String getDefaultDtype(){
        try{
            Object o = Nd4j.class.getMethod("dataType").invoke(null,(Object[])null);
            return String.valueOf(o);
        } catch (Throwable e){
            return "-";
        }
    }

}

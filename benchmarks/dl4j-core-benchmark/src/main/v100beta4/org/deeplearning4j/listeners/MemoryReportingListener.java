package org.deeplearning4j.listeners;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.utils.StringUtils;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

@Slf4j
public class MemoryReportingListener extends BaseTrainingListener {

    private int nDevices = -1;
    private String format;
    private NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    @Override
    public void iterationDone(Model model, int iteration, int epoch){

        if(nDevices < 0){

            nDevices = nativeOps.getAvailableDevices();
            StringBuilder sb = new StringBuilder();
            //jvmTotal javacppCurr javacppPhys [dev0 curr ... devN curr]
            sb.append("%-4s %20s%20s%20s");
            Object[] arr = new String[4+nDevices];
            arr[0] = "Iter";
            arr[1] = "JVM Total";
            arr[2] = "Jcpp Physical Bytes";
            arr[3] = "Jcpp Current Bytes";
            for( int i=0; i<nDevices; i++ ){
                sb.append("%20s");
                arr[4+i] = "Device " + i + " Current";
            }
            format = sb.toString();


            log.info(String.format(format, arr));
        }

        long jvmTotal = Runtime.getRuntime().totalMemory();
        long javacppCurrPhys = Pointer.physicalBytes();
        long javacppCurrBytes = Pointer.totalBytes();

        Object[] arr = new String[4 + nDevices];
        arr[0] = String.valueOf(iteration);
        arr[1] = f(jvmTotal);
        arr[2] = f(javacppCurrPhys);
        arr[3] = f(javacppCurrBytes);
        for( int i=0; i<nDevices; i++ ){
            long memUsed = getDeviceTotal(nativeOps, i) - getDeviceFreeMemory(nativeOps, i);
            arr[4+i] = f(memUsed);
        }

        log.info(String.format(format, arr));
    }

    private Map<Integer, Pointer> devPointers = new HashMap<>();

    private long getDeviceTotal(NativeOps nativeOps, int i){
        try {
            //Beta4 or later API
            Method m = NativeOps.class.getMethod("getDeviceTotalMemory", int.class);
            return (Long)m.invoke(nativeOps, i);
        } catch (Throwable t){
            //Must be beta3 or earlier
            try {
                Method m = NativeOps.class.getMethod("getDeviceTotalMemory", Pointer.class);
                return (Long) m.invoke(nativeOps, getDevicePointer(i));
            } catch (Throwable t2){
                throw new RuntimeException(t2);
            }
        }
    }

    private long getDeviceFreeMemory(NativeOps nativeOps, int i){
        try {
            //Beta4 or later API
            Method m = NativeOps.class.getMethod("getDeviceFreeMemory", int.class);
            return (Long)m.invoke(nativeOps, i);
        } catch (Throwable t){
            //Must be beta3 or earlier
            try {
                Method m = NativeOps.class.getMethod("getDeviceFreeMemory", Pointer.class);
                return (Long) m.invoke(nativeOps, getDevicePointer(i));
            } catch (Throwable t2){
                throw new RuntimeException(t2);
            }
        }
    }

    private Pointer getDevicePointer(int device) {
        if (devPointers.containsKey(device)) {
            return devPointers.get(device);
        }
        try {
            Class<?> c = Class.forName("org.nd4j.jita.allocator.pointers.CudaPointer");
            Constructor<?> constructor = c.getConstructor(long.class);
            Pointer p = (Pointer) constructor.newInstance((long) device);
            devPointers.put(device, p);
            return p;
        } catch (Throwable t) {
            devPointers.put(device, null); //Stops attempting the failure again later...
            return null;
        }
    }

    private static String f(long bytes){
        String s = StringUtils.TraditionalBinaryPrefix.long2String(bytes, "B", 2);
        String format = "%10s";
        s = String.format(format, s);
        if(bytes >= 1024){
            s += " (" + bytes + ")";
        }
        return s;
    }
}

package org.deeplearning4j.memory;

import lombok.Data;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.utils.StringUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;
import oshi.SystemInfo;
import oshi.hardware.HardwareAbstractionLayer;
import oshi.software.os.OperatingSystem;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.text.DecimalFormat;
import java.time.LocalDateTime;
import java.util.*;

/**
 * Reporting for BenchmarkListener.
 *
 * @author Justin Long (@crockpotveggies)
 */
@Data
public class MemoryBenchmarkReport {

    private String name;
    private String description;
    private MemoryTest memoryTest;
    private String dType;
    private WorkspaceMode workspaceMode;
    private List<String> devices = new ArrayList<>();
    private String backend;
    private String cpuCores;
    private String blasVendor;
    private String modelSummary;
    private String cudaVersion;
    private String cudnnVersion;
    private boolean periodicGCEnabled;
    private int periodicGCFreq;
    private int occasionalGCFreq;
    private long numParams;
    private int numLayers;
    private List<Integer> minibatchSizes;
    private long bytesMaxBeforeInit;
    private long bytesMaxPostInit;
    private Map<Integer, Object> bytesForMinibatchInference;
    private Map<Integer, Object> bytesForMinibatchTrain;

    public MemoryBenchmarkReport(String name, String description, MemoryTest memoryTest) {
        this.name = name;
        this.description = description;
        this.memoryTest = memoryTest;
        dType = Nd4j.dataType().toString();

        Properties env = Nd4j.getExecutioner().getEnvironmentInformation();
        backend = env.get("backend").toString();
        cpuCores = env.get("cores").toString();
        blasVendor = env.get("blas.vendor").toString();

        // if CUDA is present, add GPU information
        try {
            List devicesList = (List) env.get("cuda.devicesInformation");
            Iterator deviceIter = devicesList.iterator();
            while (deviceIter.hasNext()) {
                Map dev = (Map) deviceIter.next();
                devices.add(dev.get("cuda.deviceName") + " " + dev.get("cuda.deviceMajor") + " " + dev.get("cuda.deviceMinor") + " " + dev.get("cuda.totalMemory"));
            }
        } catch (Exception e) {
            SystemInfo sys = new SystemInfo();
            devices.add(sys.getHardware().getProcessor().getName());
        }

        // also get CUDA version
        try {
            Field f = Class.forName("org.bytedeco.javacpp.cuda").getField("__CUDA_API_VERSION");
            int version = f.getInt(null);
            this.cudaVersion = Integer.toString(version);
        } catch (Exception e) {
            this.cudaVersion = "n/a";
        }

        // if cuDNN is present, let's get that info
        try {
            Method m = Class.forName("org.bytedeco.javacpp.cudnn").getDeclaredMethod("cudnnGetVersion");
            long version = (long) m.invoke(null);
            this.cudnnVersion = Long.toString(version);
        } catch (Exception e) {
            this.cudnnVersion = "n/a";
        }
    }

    public void setModel(Model model) {
        this.numParams = model.numParams();

        if (model instanceof MultiLayerNetwork) {
            this.modelSummary = ((MultiLayerNetwork) model).summary();
            this.numLayers = ((MultiLayerNetwork) model).getnLayers();
        }
        if (model instanceof ComputationGraph) {
            this.modelSummary = ((ComputationGraph) model).summary();
            this.numLayers = ((ComputationGraph) model).getNumLayers();
        }
    }


    public List<String> devices() {
        return devices;
    }

    public String getModelSummary() {
        return modelSummary;
    }

    public String toString() {
        DecimalFormat df = new DecimalFormat("#.##");

        SystemInfo sys = new SystemInfo();
        OperatingSystem os = sys.getOperatingSystem();
        HardwareAbstractionLayer hardware = sys.getHardware();



        List<Object[]> t = new ArrayList<>();
        t.add(new String[]{"Name", name});
        t.add(new String[]{"Description", description});
        t.add(new String[]{"Memory Test Type", memoryTest.toString()});
        t.add(new String[]{"Workspace mode", workspaceMode.toString()});
        t.add(new String[]{"Data type", dType.toString()});
        t.add(new String[]{"Operating System",
                os.getManufacturer() + " " +
                        os.getFamily() + " " +
                        os.getVersion().getVersion()});
        t.add(new String[]{"Devices", devices().get(0)});
        t.add(new String[]{"CPU Cores", cpuCores});
        t.add(new String[]{"Backend", backend});
        t.add(new String[]{"BLAS Vendor", blasVendor});
        t.add(new String[]{"CUDA Version", cudaVersion});
        t.add(new String[]{"CUDNN Version", cudnnVersion});
        t.add(new String[] { "Periodic GC enabled", String.valueOf(periodicGCEnabled) });
        if(periodicGCEnabled){
            t.add(new String[] { "Periodic GC frequency", String.valueOf(periodicGCFreq) });
        }
        t.add(new String[] { "Occasional GC Freq", String.valueOf(occasionalGCFreq) });
        t.add(new String[]{"Total Params", "" + numParams});
        t.add(new String[]{"Total Layers", Integer.toString(numLayers)});
        t.add(new String[]{"Bytes before init", formatBytes(bytesMaxBeforeInit)});
        t.add(new String[]{"Bytes post init", formatBytes(bytesMaxPostInit)});
        t.add(new String[]{"Tested minibatch sizes", minibatchSizes.toString()});
        if(memoryTest == MemoryTest.INFERENCE){
            t.add(new String[]{"Memory use vs minibatch (inference)", ""});
            for(Map.Entry<Integer,Object> e : bytesForMinibatchInference.entrySet()){
                String str;
                if(e.getValue() instanceof Number){
                    Long l = ((Number) e.getValue()).longValue();
                    str = formatBytes(l);
                } else {
                    str = e.getValue().toString();
                }
                t.add(new String[]{"  Minibatch " + e.getKey(), str});
            }
        } else if(memoryTest == MemoryTest.TRAINING){
            t.add(new String[]{"Memory use vs minibatch (training)", ""});
            for(Map.Entry<Integer,Object> e : bytesForMinibatchTrain.entrySet()){
                String str;
                if(e.getValue() instanceof Number){
                    Long l = ((Number) e.getValue()).longValue();
                    str = formatBytes(l);
                } else {
                    str = e.getValue().toString();
                }
                t.add(new String[]{"  Minibatch " + e.getKey(), str});
            }
        } else {
            throw new RuntimeException(memoryTest.toString());
        }

        final Object[][] table = t.toArray(new Object[t.size()][0]);

        StringBuilder sb = new StringBuilder();

        for (final Object[] row : table) {
            sb.append(String.format("%28s %45s\n", row));
        }

        return sb.toString();
    }

    private static String formatBytes(long bytes){
        return bytes + " - " + StringUtils.TraditionalBinaryPrefix.long2String(bytes, null, 2);
    }

}

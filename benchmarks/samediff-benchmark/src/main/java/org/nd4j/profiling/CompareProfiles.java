package org.nd4j.profiling;

import org.nd4j.autodiff.listeners.profiler.ProfilingListener;
import org.nd4j.autodiff.listeners.profiler.comparison.ProfileAnalyzer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class CompareProfiles {

    public static void main(String[] args) {

        File modelsRootDir = new File("~/TF_Graphs/");
        File tfProfileRootDir = new File("~/DL4J/dl4j-dev-tools/import-tests/profiling");
        File sdProfileOutputDir = new File("~/DL4J/dl4j-dev-tools/import-tests/profiling/sd");

        //Available tests (so far): mobilenetv2, inception_resnet_v2, faster_rcnn_resnet101_coco
        String testName = "mobilenetv2";
        int batch = 32; //1 or 32
        String tfVersion = "1.15.0";

        int warmup = 3;
        int runs = 10;

        File modelFile;
        List<String> outputNames;
        Map<String, INDArray> ph;
        String dirName;
        switch (testName){
            case "mobilenetv2":
                modelFile = new File(modelsRootDir, "mobilenet_v2_1.0_224_frozen.pb");
                outputNames = Collections.singletonList("MobilenetV2/Predictions/Reshape_1:0");
                ph = Collections.singletonMap("input:0", Nd4j.rand(DataType.FLOAT, batch, 224, 224, 3));
                dirName = "mobilenet_v2_1.0_224_batch" + batch + "_tf-" + tfVersion;
                break;
            case "inception_resnet_v2":
                modelFile = new File(modelsRootDir, "mobilenet_v2_1.0_224_frozen.pb");
                outputNames = Collections.singletonList("InceptionResnetV2/AuxLogits/Logits/BiasAdd:0");
                ph = Collections.singletonMap("input:0", Nd4j.rand(DataType.FLOAT, batch, 299, 299, 3));
                dirName = "inception_resnet_v2_batch" + batch + "_tf-" + tfVersion;
                break;
            case "faster_rcnn_resnet101_coco":
                modelFile = new File(modelsRootDir, "mobilenet_v2_1.0_224_frozen.pb");
                outputNames = Arrays.asList("detection_boxes:0", "detection_scores:0", "num_detections:0", "detection_classes:0");
                ph = Collections.singletonMap("input:0", Nd4j.rand(DataType.FLOAT, batch, 600, 600, 3));
                dirName = "faster_rcnn_resnet101_coco_batch" + batch + "_tf-" + tfVersion;
                break;
            default:
                throw new RuntimeException("Unknown/not implemented test name: " + testName);
        }

        File dir = new File(tfProfileRootDir, dirName);

        SameDiff sd = SameDiff.importFrozenTF(modelFile);

        for( int i=0; i<warmup; i++ ){
            Map<String,INDArray> m = sd.output(ph, outputNames);
            for(INDArray arr : m.values()){
                arr.close();
            }
        }


        File sdDir = new File(sdProfileOutputDir, testName + "_" + System.currentTimeMillis());
        ProfilingListener l = ProfilingListener.builder(sdDir)
                .recordAll()
                .warmup(0)
                .build();
        sd.setListeners(l);


        for( int i=0; i<runs; i++ ){
            Map<String,INDArray> m = sd.output(ph, outputNames);
            for(INDArray arr : m.values()){
                arr.close();
            }
        }

        //Now, compare profiles:
        String s = ProfileAnalyzer.compareProfiles(sdDir, dir, ProfileAnalyzer.ProfileFormat.SAMEDIFF, ProfileAnalyzer.ProfileFormat.TENSORFLOW,
                true, true, "sd", "tf", ProfileAnalyzer.SortBy.PROFILE1_PC);

        System.out.println(s);
    }

}

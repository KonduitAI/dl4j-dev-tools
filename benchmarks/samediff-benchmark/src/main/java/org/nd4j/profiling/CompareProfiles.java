package org.nd4j.profiling;

import lombok.extern.slf4j.Slf4j;
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

/**
 * SameDiff profile comparison tool - typically used for SameDiff vs. TensorFlow import, but could be adapted for
 * SameDiff vs. SameDiff (say CPU vs. CUDA).
 *
 * Note it is set up to use the output of dl4j-dev-tools/import-tests/profiling/zoo_model_profiling.py
 *
 * The corresponding frozen TensorFlow models are loaded here and profiled, before printing out the comparison results
 *
 * @author Alex Black
 */
@Slf4j
public class CompareProfiles {

    public static void main(String[] args) {

        File modelsRootDir = new File("/home/alex/TF_Graphs/");             //Directory where the frozen .pb TensorFlow files are located
        File tfProfileRootDir = new File("/home/alex/profiling");           //Directory where the TensorFlow profiles were written (by zoo_model_profiling.py)
        File sdProfileOutputDir = new File("/home/alex/profiling/sd");      //Directory where the SameDiff profiles should be written

        //Available tests (so far): mobilenetv2, inception_resnet_v2, faster_rcnn_resnet101_coco
        String testName = "mobilenetv2";
        int batch = 32; //1 or 32, to match zoo_model_profiling.py
        String tfVersion = "1.15.0";

        //Number of forward pass iterations
        int warmup = 3;
        int runs = 10;

        File modelFile;
        List<String> outputNames;
        Map<String, INDArray> ph;
        String dirName;
        switch (testName){
            case "mobilenetv2":
                modelFile = new File(modelsRootDir, "mobilenet_v2_1.0_224_frozen.pb");
                outputNames = Collections.singletonList("MobilenetV2/Predictions/Reshape_1");
                ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 224, 224, 3));
                dirName = "mobilenet_v2_1.0_224_batch" + batch + "_tf-" + tfVersion;
                break;
            case "inception_resnet_v2":
                modelFile = new File(modelsRootDir, "mobilenet_v2_1.0_224_frozen.pb");
                outputNames = Collections.singletonList("InceptionResnetV2/AuxLogits/Logits/BiasAdd");
                ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 299, 299, 3));
                dirName = "inception_resnet_v2_batch" + batch + "_tf-" + tfVersion;
                break;
            case "faster_rcnn_resnet101_coco":
                modelFile = new File(modelsRootDir, "mobilenet_v2_1.0_224_frozen.pb");
                outputNames = Arrays.asList("detection_boxes", "detection_scores", "num_detections", "detection_classes");
                ph = Collections.singletonMap("input:0", Nd4j.rand(DataType.FLOAT, batch, 600, 600, 3));
                dirName = "faster_rcnn_resnet101_coco_batch" + batch + "_tf-" + tfVersion;
                break;
            default:
                throw new RuntimeException("Unknown/not implemented test name: " + testName);
        }

        File dir = new File(tfProfileRootDir, dirName);

        SameDiff sd = SameDiff.importFrozenTF(modelFile);

        log.info("Starting warmup - {} iterations", warmup);
        for( int i=0; i<warmup; i++ ){
            Map<String,INDArray> m = sd.output(ph, outputNames);
            for(INDArray arr : m.values()){
                if(arr.closeable())
                    arr.close();
            }
        }


        //Set up SameDiff profiling listener and JSON output
        File sdDir = new File(sdProfileOutputDir, testName + "_" + System.currentTimeMillis());
        sdDir.mkdirs();
        File sdProfileFile = new File(sdDir, "profile.json");
        ProfilingListener l = ProfilingListener.builder(sdProfileFile)
                .recordAll()
                .warmup(0)
                .build();
        sd.setListeners(l);


        log.info("Starting profiling - {} iterations", runs);
        for( int i=0; i<runs; i++ ){
            Map<String,INDArray> m = sd.output(ph, outputNames);
            for(INDArray arr : m.values()){
                if(arr.closeable())
                    arr.close();
            }
        }

        //Now, compare profiles:
        String s = ProfileAnalyzer.compareProfiles(sdDir, dir, ProfileAnalyzer.ProfileFormat.SAMEDIFF, ProfileAnalyzer.ProfileFormat.TENSORFLOW,
                true, true, "sd", "tf", ProfileAnalyzer.SortBy.RATIO);

        System.out.println(s);


        //And print raw profile info
        System.out.println("============================================================================");
        System.out.println(" ----- TensorFlow Profile Summary -----");
        ProfileAnalyzer.summarizeProfileDirectory(dir, ProfileAnalyzer.ProfileFormat.TENSORFLOW);

        System.out.println("============================================================================");
        System.out.println(" ----- SameDiff Profile Summary -----");
        ProfileAnalyzer.summarizeProfileDirectory(sdDir, ProfileAnalyzer.ProfileFormat.SAMEDIFF);
    }
}

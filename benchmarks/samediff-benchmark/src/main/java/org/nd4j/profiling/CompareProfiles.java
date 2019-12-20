/* ******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
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

//        File modelsRootDir = new File("/home/alex/TF_Graphs/");             //Directory where the frozen .pb TensorFlow files are located
//        File tfProfileRootDir = new File("/home/alex/profiling");           //Directory where the TensorFlow profiles were written (by zoo_model_profiling.py)
//        File sdProfileOutputDir = new File("/home/alex/profiling/sd");      //Directory where the SameDiff profiles should be written

        File modelsRootDir = new File("/home/alex/TF_Graphs/");             //Directory where the frozen .pb TensorFlow files are located
        File tfProfileRootDir = new File("/home/alex/TF_Graphs/gpu_profiling");           //Directory where the TensorFlow profiles were written (by zoo_model_profiling.py)
        File sdProfileOutputDir = new File("/home/alex/TF_Graphs/gpu_profiling/sd");      //Directory where the SameDiff profiles should be written

        //Available tests (so far): mobilenetv2, inception_resnet_v2, faster_rcnn_resnet101_coco
        String testName = "densenet";
//        String testName = "squeezenet";
//        String testName = "nasnet_mobile";
//        String testName = "inception_v4_2018_04_27";
//        String testName = "inception_resnet_v2";
//        String testName = "mobilenetv1";
//        String testName = "mobilenetv2";
//        String testName = "ssd_mobilenet";
//        String testName = "faster_rcnn_resnet101_coco";
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
            case "densenet":
                modelFile = new File(modelsRootDir, "densenet_2018_04_27/densenet.pb");
                outputNames = Arrays.asList("ArgMax", "softmax_tensor");
                ph = Collections.singletonMap("Placeholder", Nd4j.rand(DataType.FLOAT, batch, 224, 224, 3));
                dirName = "densenet_2018_04_27_batch" + batch + "_tf-" + tfVersion;
                break;
            case "squeezenet":
                modelFile = new File(modelsRootDir, "squeezenet_2018_04_27/squeezenet.pb");
                outputNames = Arrays.asList("ArgMax", "softmax_tensor");
                ph = Collections.singletonMap("Placeholder", Nd4j.rand(DataType.FLOAT, batch, 224, 224, 3));
                dirName = "squeezenet_2018_04_27_batch" + batch + "_tf-" + tfVersion;
                break;
            case "nasnet_mobile":
                modelFile = new File(modelsRootDir, "nasnet_mobile_2018_04_27/nasnet_mobile.pb");
                outputNames = Arrays.asList("final_layer/predictions");
                ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 224, 224, 3));
                dirName = "nasnet_mobile_2018_04_27_batch" + batch + "_tf-" + tfVersion;
                break;
            case "inception_v4_2018_04_27":
                modelFile = new File(modelsRootDir, "inception_v4_2018_04_27/inception_v4.pb");
                outputNames = Arrays.asList("InceptionV4/Logits/Predictions");
                ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 299, 299, 3));
                dirName = "inception_v4_2018_04_27_batch" + batch + "_tf-" + tfVersion;
                break;
            case "inception_resnet_v2":
                modelFile = new File(modelsRootDir, "inception_resnet_v2.pb");
                outputNames = Collections.singletonList("InceptionResnetV2/AuxLogits/Logits/BiasAdd");
                ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 299, 299, 3));
                dirName = "inception_resnet_v2_batch" + batch + "_tf-" + tfVersion;
                break;
            case "mobilenetv1":
                modelFile = new File(modelsRootDir, "mobilenet_v1_0.5_128/mobilenet_v1_0.5_128_frozen.pb");
                outputNames = Collections.singletonList("MobilenetV1/Predictions/Reshape_1");
                ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 128, 128, 3));
                dirName = "mobilenet_v1_0.5_128_batch" + batch + "_tf-" + tfVersion;
                break;
            case "mobilenetv2":
                modelFile = new File(modelsRootDir, "mobilenet_v2_1.0_224_frozen.pb");
                outputNames = Collections.singletonList("MobilenetV2/Predictions/Reshape_1");
                ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 224, 224, 3));
                dirName = "mobilenet_v2_1.0_224_batch" + batch + "_tf-" + tfVersion;
                break;
            case "ssd_mobilenet":
                modelFile = new File(modelsRootDir, "ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb");
                outputNames = Arrays.asList("detection_boxes", "detection_scores", "num_detections", "detection_classes");
                ph = Collections.singletonMap("input_tensor", Nd4j.rand(DataType.FLOAT, batch, 320, 320, 3));
                dirName = "ssd_mobilenet_v1_coco_2018_01_28_batch" + batch + "_tf-" + tfVersion;
                break;
            case "faster_rcnn_resnet101_coco":
                modelFile = new File(modelsRootDir, "faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb");
                outputNames = Arrays.asList("detection_boxes", "detection_scores", "num_detections", "detection_classes");
                ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 600, 600, 3));
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
        log.info("SameDiff profiling - output path: {}", sdProfileFile.getAbsolutePath());
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

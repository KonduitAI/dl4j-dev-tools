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
import org.nd4j.autodiff.samediff.SDVariable;
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
 * SameDiff profile generation - generate and export profiles, for CPU vs. GPU profiling
 *
 * @author Alex Black
 */
@Slf4j
public class RunModelProfiles {

    public static void main(String[] args) {

        File modelsRootDir = new File("/home/alex/TF_Graphs/");                     //Directory where the frozen .pb TensorFlow files are located
        File profileOutputDir = new File("/home/alex/profiling/cpu_vs_gpu");        //Directory where the SameDiff profiles should be written

        //Available tests (so far): mobilenetv2, inception_resnet_v2, faster_rcnn_resnet101_coco
        for(String testName : new String[]{
            "densenet", "squeezenet", "nasnet_mobile", /*"inception_v4_2018_04_27",*/ "inception_resnet_v2", "mobilenetv1",
                "mobilenetv2", /*"ssd_mobilenet", "faster_rcnn_resnet101_coco" */}){
            System.gc();
            int batch = 32;

            //Number of forward pass iterations
            int warmup = 3;
            int runs = 10;

            File modelFile;
            List<String> outputNames;
            Map<String, INDArray> ph;
            String dirName;
            switch (testName) {
                case "densenet":
                    modelFile = new File(modelsRootDir, "densenet_2018_04_27/densenet.pb");
                    outputNames = Arrays.asList("ArgMax", "softmax_tensor");
                    ph = Collections.singletonMap("Placeholder", Nd4j.rand(DataType.FLOAT, batch, 224, 224, 3));
                    break;
                case "squeezenet":
                    modelFile = new File(modelsRootDir, "squeezenet_2018_04_27/squeezenet.pb");
                    outputNames = Arrays.asList("ArgMax", "softmax_tensor");
                    ph = Collections.singletonMap("Placeholder", Nd4j.rand(DataType.FLOAT, batch, 224, 224, 3));
                    break;
                case "nasnet_mobile":
                    modelFile = new File(modelsRootDir, "nasnet_mobile_2018_04_27/nasnet_mobile.pb");
                    outputNames = Arrays.asList("final_layer/predictions");
                    ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 224, 224, 3));
                    break;
                case "inception_v4_2018_04_27":
                    modelFile = new File(modelsRootDir, "inception_v4_2018_04_27/inception_v4.pb");
                    outputNames = Arrays.asList("InceptionV4/Logits/Predictions");
                    ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 299, 299, 3));
                    break;
                case "inception_resnet_v2":
                    modelFile = new File(modelsRootDir, "inception_resnet_v2.pb");
                    outputNames = Collections.singletonList("InceptionResnetV2/AuxLogits/Logits/BiasAdd");
                    ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 299, 299, 3));
                    break;
                case "mobilenetv1":
                    modelFile = new File(modelsRootDir, "mobilenet_v1_0.5_128/mobilenet_v1_0.5_128_frozen.pb");
                    outputNames = Collections.singletonList("MobilenetV1/Predictions/Reshape_1");
                    ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 128, 128, 3));
                    break;
                case "mobilenetv2":
                    modelFile = new File(modelsRootDir, "mobilenet_v2_1.0_224_frozen.pb");
                    outputNames = Collections.singletonList("MobilenetV2/Predictions/Reshape_1");
                    ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 224, 224, 3));
                    break;
                case "ssd_mobilenet":
                    modelFile = new File(modelsRootDir, "ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb");
                    outputNames = Arrays.asList("detection_boxes", "detection_scores", "num_detections", "detection_classes");
                    ph = Collections.singletonMap("input_tensor", Nd4j.rand(DataType.FLOAT, batch, 320, 320, 3));
                    break;
                case "faster_rcnn_resnet101_coco":
                    modelFile = new File(modelsRootDir, "faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb");
                    outputNames = Arrays.asList("detection_boxes", "detection_scores", "num_detections", "detection_classes");
                    ph = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, batch, 600, 600, 3));
                    break;
                default:
                    throw new RuntimeException("Unknown/not implemented test name: " + testName);
            }

            SameDiff sd = SameDiff.importFrozenTF(modelFile);

            log.info("Starting warmup - {} iterations", warmup);
            for (int i = 0; i < warmup; i++) {
                Map<String, INDArray> m = sd.output(ph, outputNames);
                for (INDArray arr : m.values()) {
                    if (arr.closeable())
                        arr.close();
                }
            }


            //Set up SameDiff profiling listener and JSON output
            String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
            File outDir = new File(profileOutputDir, backend);
            File sdDir = new File(outDir, testName);
            sdDir.mkdirs();
            File sdProfileFile = new File(sdDir, "profile.json");
            log.info("SameDiff profiling - output path: {}", sdProfileFile.getAbsolutePath());
            ProfilingListener l = ProfilingListener.builder(sdProfileFile)
                    .recordAll()
                    .warmup(0)
                    .build();
            sd.setListeners(l);


            log.info("Starting profiling - {} iterations", runs);
            for (int i = 0; i < runs; i++) {
                Map<String, INDArray> m = sd.output(ph, outputNames);
                for (INDArray arr : m.values()) {
                    if (arr.closeable())
                        arr.close();
                }
            }

            for(SDVariable v : sd.variables()){
                try{
                    v.getArr().close();
                } catch (Throwable t){ }
            }

            System.gc();
        }
    }
}

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
public class CompareExistingProfiles {

    public static void main(String[] args) {

        String name1 = "AVX2";
        String name2 = "x86";

//        File profile1 = new File("/home/alex/profiling/sd/mobilenetv1_1576814270138/profile.json");     //AVX2
//        File profile2 = new File("/home/alex/profiling/sd/mobilenetv1_1576814736342/profile.json");     //x86

        File profile1 = new File("/home/alex/profiling/sd/inception_v4_2018_04_27_1576814180331/profile.json");     //AVX2
        File profile2 = new File("/home/alex/profiling/sd/inception_v4_2018_04_27_1576815483951/profile.json");     //x86

        boolean profile1IsDir = false;
        boolean profile2IsDir = false;

        ProfileAnalyzer.ProfileFormat f1 = ProfileAnalyzer.ProfileFormat.SAMEDIFF;
        ProfileAnalyzer.ProfileFormat f2 = ProfileAnalyzer.ProfileFormat.SAMEDIFF;


        //Now, compare profiles:
        String s = ProfileAnalyzer.compareProfiles(profile1, profile2, f1, f2,
                profile1IsDir, profile2IsDir, name1, name2, ProfileAnalyzer.SortBy.RATIO);

        System.out.println(s);


        //And print raw profile info
        System.out.println("============================================================================");
        System.out.println(" ----- " + name1 + " Profile Summary -----");
        if(profile1IsDir) {
            ProfileAnalyzer.summarizeProfileDirectory(profile1, f1);
        } else {
            ProfileAnalyzer.summarizeProfile(profile1, f1);
        }

        System.out.println("============================================================================");
        System.out.println(" ----- " + name2 + " Profile Summary -----");
        if(profile2IsDir) {
            ProfileAnalyzer.summarizeProfileDirectory(profile2, f2);
        } else {
            ProfileAnalyzer.summarizeProfile(profile2, f2);
        }
    }
}

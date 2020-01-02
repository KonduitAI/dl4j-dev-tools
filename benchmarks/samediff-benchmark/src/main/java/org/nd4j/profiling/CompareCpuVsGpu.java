package org.nd4j.profiling;

import org.nd4j.autodiff.listeners.profiler.comparison.Config;
import org.nd4j.autodiff.listeners.profiler.comparison.OpStats;
import org.nd4j.autodiff.listeners.profiler.comparison.ProfileAnalyzer;
import org.nd4j.linalg.function.BiFunction;

import java.io.File;
import java.util.HashSet;
import java.util.Set;

public class CompareCpuVsGpu {

    public static void main(String[] args) {

        String testName = "densenet";
//        String testName = "squeezenet";
//        String testName = "nasnet_mobile";
//        String testName = "inception_v4_2018_04_27";
//        String testName = "inception_resnet_v2";
//        String testName = "mobilenetv1";
//        String testName = "mobilenetv2";
//        String testName = "ssd_mobilenet";
//        String testName = "faster_rcnn_resnet101_coco";

        File profileOutputDir = new File("/home/alex/profiling/cpu_vs_gpu");        //Directory where the SameDiff profiles should be written

        File p1 = new File(profileOutputDir, "CUDA/" + testName + "/profile.json");
        File p2 = new File(profileOutputDir, "CPU/" + testName + "/profile.json");

        //Now, compare profiles:
        final Set<String> skipOps = new HashSet<>();
        skipOps.add("identity");

        Config c = Config.builder()
                .profile1(p1)
                .profile2(p2)
                .profile1Format(ProfileAnalyzer.ProfileFormat.SAMEDIFF)
                .profile2Format(ProfileAnalyzer.ProfileFormat.SAMEDIFF)
                .profile1IsDir(false)
                .profile2IsDir(false)
                .p1Name("CUDA")
                .p2Name("CPU")
                .sortBy(ProfileAnalyzer.SortBy.RATIO)
                .filter(new BiFunction<OpStats, OpStats, Boolean>() {
                    @Override
                    public Boolean apply(OpStats opStats, OpStats opStats2) {
                        //True to keep, false to remove
                        if(opStats != null && skipOps.contains(opStats.getOpName()) || opStats2 != null && skipOps.contains(opStats2.getOpName())){
                            return false;
                        }
                        return true;
                    }
                })
                .build();

        String s = ProfileAnalyzer.compareProfiles(c);

        System.out.println(s);


        //And print raw profile info
        System.out.println("============================================================================");
        System.out.println(" ----- CUDA Profile Summary -----");
        ProfileAnalyzer.summarizeProfile(p1, ProfileAnalyzer.ProfileFormat.SAMEDIFF);

        System.out.println("============================================================================");
        System.out.println(" ----- SameDiff Profile Summary -----");
        ProfileAnalyzer.summarizeProfile(p2, ProfileAnalyzer.ProfileFormat.SAMEDIFF);


    }

}

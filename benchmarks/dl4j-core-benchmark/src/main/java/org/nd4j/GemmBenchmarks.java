package org.nd4j;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

/**
 * Created by Alex on 31/07/2017.
 */
@Slf4j
public class GemmBenchmarks {

    @AllArgsConstructor
    @Data
    private static class TestCase {
        private final int a;
        private final int b;
        private final int c;
        private final char xO;
        private final char yO;
        private final char zO;
        private final int nTests;
        private final boolean reuseArrays;
        private final boolean gcFirst;

        @Override
        public String toString(){
            return a + "," + b + "," + c + "," + xO + yO + zO + "," + nTests + "," + reuseArrays + "," + gcFirst;
        }

        public static String getHeader(){
            return "a,b,c,orders,nTests,reuseArrays,gcFirst";
        }
    }


    public static void main(String[] args) throws Exception {

        System.out.println("PATH:" + System.getenv("PATH"));

        Properties p = Nd4j.getExecutioner().getEnvironmentInformation();
        System.out.println(p);
        Object os = p.get("os");
        Object blasThreads = p.get("blas.threads");
        Object backend = p.get("backend");
        Object ompThreads = p.get("omp.threads");
        Object vendor = p.get("blas.vendor");

        String env = os + "," + backend + "," + vendor + "," + blasThreads;
        String envH = "os,backend,vendor,blasThreads";


        List<TestCase> testCases = getTestCases();


        System.out.println(envH + "," + TestCase.getHeader() + ",meanNs,medianNs,p90Ns");
        for(TestCase tc : testCases ){

            if(tc.isGcFirst()){
                System.gc();
                Thread.sleep(1000L);
            }

            INDArray x = null;
            INDArray y = null;
            INDArray z = null;

            if(tc.isReuseArrays()){
                x = getArray(tc, "x");
                y = getArray(tc, "y");
                z = getArray(tc, "z");
            }

            long[] times = new long[tc.getNTests()];
            for(int i = 0; i<tc.getNTests(); i++ ){
                if(!tc.isReuseArrays()){
                    x = getArray(tc, "x");
                    y = getArray(tc, "y");
                    z = getArray(tc, "z");
                }

                long start = System.nanoTime();
                if(tc.getZO() == 'f'){
                    Nd4j.gemm(x, y, z, false, false, 1.0, 0.0); //C = 1.0 * AxB + 0.0 * C
                } else if(tc.getZO() == 'c'){
                    //Need to transpose to get result in C order
                    //Normally we have Z = X*Y, with Z in f order
                    //Use the fact that Z^T (f order) = (X*Y)^T = Y^T * X*T, and finally that Z^T (f order) == Z (c order)
                    Nd4j.gemm(y, x, z.transpose(), true, true, 1.0, 0.0);
                } else {
                    throw new RuntimeException();
                }
                long end = System.nanoTime();

                times[i] = end - start;
            }

            Arrays.sort(times);
            double mean = Arrays.stream(times).asDoubleStream().sum() / times.length;
            double median;
            if(times.length % 2 == 0){
                median = (times[times.length/2 - 1] + times[times.length/2]) / 2.0;
            } else {
                median = times[times.length/2];
            }
            int p90pos = (int)(0.9 * times.length);
            double p90 = times[p90pos];

//            log.info("{}, mean = {}, median = {}, p90 = {}", tc, mean, median, p90);
            System.out.println(env + "," + tc + "," + mean + "," + median + "," + p90);
        }
    }

    private static INDArray getArray(TestCase tc, String array){
        switch (array.toLowerCase()){
            case "x":
                return Nd4j.create(new int[]{tc.a, tc.b}, tc.xO);
            case "y":
                return Nd4j.create(new int[]{tc.b, tc.c}, tc.yO);
            case "z":
                return Nd4j.create(new int[]{tc.a, tc.c}, tc.zO);
            default:
                throw new RuntimeException();
        }
    }


    private static List<TestCase> getTestCases(){

        List<TestCase> l = new ArrayList<>();

//        int[] sizesA = new int[]{16, 64, 256, 1024, 4096, 16384};
        int[] sizesA = new int[]{16, 64, 256, 1024, 4096};
        int[] sizesB = sizesA;
        int[] sizesC = sizesA;

        char[] o = {'c','f'};
        for( char xo : o){
            for( char yo : o ){
                for( char zo : o ){
                    for(int a : sizesA ){
                        for(int b : sizesB){
                            for(int c : sizesC){
                                l.add(new TestCase(a, b, c, xo, yo, zo, 100, true, true));
                            }
                        }
                    }
                }
            }
        }

        return l;
    }

}

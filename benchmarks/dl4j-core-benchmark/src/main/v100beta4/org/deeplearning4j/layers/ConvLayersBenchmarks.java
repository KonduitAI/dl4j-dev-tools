//package org.deeplearning4j.layers;
//
//import lombok.AllArgsConstructor;
//import lombok.Data;
//import org.deeplearning4j.nn.conf.ConvolutionMode;
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
//import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
//import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
//import org.deeplearning4j.util.ConvolutionUtils;
//import org.nd4j.linalg.activations.Activation;
//import org.nd4j.linalg.api.buffer.DataType;
//import org.nd4j.linalg.api.memory.MemoryWorkspace;
//import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
//import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
//import org.nd4j.linalg.api.memory.enums.LearningPolicy;
//import org.nd4j.linalg.api.memory.enums.ResetPolicy;
//import org.nd4j.linalg.api.memory.enums.SpillPolicy;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.api.ops.DynamicCustomOp;
//import org.nd4j.linalg.api.ops.OpContext;
//import org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D;
//import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
//import org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling2D;
//import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.util.ArrayUtil;
//import org.nd4j.nativeblas.Nd4jCpu;
//
//import java.text.DecimalFormat;
//import java.util.Arrays;
//import java.util.List;
//
//public class ConvLayersBenchmarks {
//
//    private static final int WARMUP = 30;
//    private static final int ITERS = 100;
//
//
//    public static void main(String[] args) {
//
//
//        List<long[]> convSizes = Arrays.asList(new long[]{32, 3, 64, 64}, new long[]{64, 128, 16, 16}, new long[]{128, 512, 64, 64});
//
////        for(boolean permutedIn : new boolean[]{false, true}) {    //TODO add support for permuted inputs, as might occur in DL4J
//        for(boolean permutedIn : new boolean[]{false}) {
//            for (long[] inShape : convSizes) {
//                INDArray in;
//                if(permutedIn){
//                    //NHWC -> NCHW
//                    in = Nd4j.create(DataType.FLOAT, inShape[0], inShape[2], inShape[3], inShape[1]).permute(0,3,1,2);
//                } else {
//                    //Standard case
//                    in = Nd4j.create(DataType.FLOAT, inShape);
//                }
//
//                //Conv2d - kernel 2, stride 1, same
//                conv2d_NCHW(WARMUP, inShape, 2, 1, true);
//                Timings t = conv2d_NCHW(ITERS, inShape, 2, 1, true);
//                System.out.println("Conv2d, shape=" + Arrays.toString(inShape) + ", k=2, s=1, same");
//                System.out.println(t);
//
//                //Conv2d - kernel 3, stride 2, same
//                conv2d_NCHW(WARMUP, inShape, 3, 2, true);
//                t = conv2d_NCHW(ITERS, inShape, 3, 2, true);
//                System.out.println("Conv2d, shape=" + Arrays.toString(inShape) + ", k=3, s=2, same");
//                System.out.println(t);
//
//                System.out.println("-----------------");
//
//                //Pooling2d
//                conv2d_NCHW(WARMUP, inShape, 2, 1, true);
//                t = pool2d_NCHW(ITERS, inShape, 2, 1, true, true);
//                System.out.println("maxPool2d, shape=" + Arrays.toString(inShape) + ", k=2, s=1, same");
//                System.out.println(t);
//
//                conv2d_NCHW(WARMUP, inShape, 3, 2, true);
//                t = pool2d_NCHW(ITERS, inShape, 3, 2, true, true);
//                System.out.println("maxPool2d, shape=" + Arrays.toString(inShape) + ", k=3, s=2, same");
//                System.out.println(t);
//
//                conv2d_NCHW(WARMUP, inShape, 2, 1, true);
//                t = pool2d_NCHW(ITERS, inShape, 2, 1, true, false);
//                System.out.println("avgPool2d, shape=" + Arrays.toString(inShape) + ", k=2, s=1, same");
//                System.out.println(t);
//
//                conv2d_NCHW(WARMUP, inShape, 3, 2, true);
//                t = pool2d_NCHW(ITERS, inShape, 3, 2, true, false);
//                System.out.println("avgPool2d, shape=" + Arrays.toString(inShape) + ", k=3, s=2, same");
//                System.out.println(t);
//
//
//                System.out.println("-----------------");
//
//                //Batch norm
//
//
//                //LRN
//            }
//        }
//    }
//
//    @Data
//    @AllArgsConstructor
//    public static class Timings {
//        public final int numRuns;
//        public final long cppOp;
//        public final long opMKLDNN;
//        public final long dl4jNoHelper;
//
//        @Override
//        public String toString(){
//            return "n=" + numRuns + ", cpp=" + df.format(cppOp / (double)numRuns)
//                    + "ms, mkldnn=" + df.format(opMKLDNN / (double)numRuns)
//                    + "ms, dl4jLayer=" + df.format(dl4jNoHelper / (double)numRuns) + "ms";
//        }
//    }
//
//    private static final DecimalFormat df = new DecimalFormat("0.00");
//
//    private static Timings conv2d_NCHW(int numRuns, long[] inShape, int k, int s, boolean same){
//        int[] kernel = {k,k};
//        int[] strides = {s,s};
//        int[] pad = {0,0};
//        int[] dilation = {1,1};
//
//        int inH = (int)inShape[2];
//        int inW = (int)inShape[3];
//
//        int nIn = (int) inShape[1];
//        //Weights: conv2d op expects [kH, kW, iC, oC] weights... DL4J conv uses [oC, iC, kH, kW]
//        INDArray weights = Nd4j.create(DataType.FLOAT, k, k, nIn, nIn);
//        INDArray bias = Nd4j.create(DataType.FLOAT, nIn);
//        INDArray input = Nd4j.createUninitialized(DataType.FLOAT, inShape);
//
//
//
//        int[] outSize;
//        ConvolutionMode convolutionMode = same ? ConvolutionMode.Same : ConvolutionMode.Truncate;
//        if (same) {
//            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation); //Also performs validation
//            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {inH, inW}, kernel, strides, dilation);
//        } else {
//            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode, dilation); //Also performs validation
//        }
//        INDArray out = Nd4j.createUninitialized(DataType.FLOAT, inShape[0], nIn, outSize[0], outSize[1]);
//
//        long mkldnn = 0;
//        long opNoMKLDNN = 0;
//        for(boolean mkl : new boolean[]{false, true}) {
//            Nd4jCpu.Environment.getInstance().setUseMKLDNN(mkl);
//            OpContext context = Nd4j.getExecutioner().buildContext();
//            context.setIArguments(kernel[0], kernel[1],
//                    strides[0], strides[1],
//                    pad[0], pad[1],
//                    dilation[0], dilation[1],
//                    same ? 1 : 0,  //Same mode
//                    0   //0=NCHW
//            );
//            INDArray[] inputsArr = new INDArray[]{input, weights, bias};
//            context.getInputArrays().clear();
//            for (int i = 0; i < inputsArr.length; i++) {
//                context.setInputArray(i, inputsArr[i]);
//            }
//            context.setOutputArray(0, out);
//            Conv2D op = new Conv2D();
//            long start = System.currentTimeMillis();
//            for (int i = 0; i < numRuns; i++) {
//                Nd4j.exec(op, context);
//            }
//            if(mkl){
//                mkldnn = System.currentTimeMillis() - start;
//            } else {
//                opNoMKLDNN = System.currentTimeMillis() - start;
//            }
//        }
//
//        //DL4J layer:
//        ConvolutionLayer conf = new ConvolutionLayer.Builder()
//                .convolutionMode(convolutionMode)
//                .nIn(nIn)
//                .nOut(nIn)
//                .kernelSize(k,k)
//                .stride(s,s)
//                .activation(Activation.IDENTITY)
//                .build();
//        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
//        nnc.setLayer(conf);
//
//        long nParams = conf.initializer().numParams(conf);
//        INDArray pView = Nd4j.create(DataType.FLOAT, 1, nParams);
//
//        Nd4jCpu.Environment.getInstance().setUseMKLDNN(false);  //Disable MKLDNN so no helper
//        org.deeplearning4j.nn.layers.convolution.ConvolutionLayer layer = (org.deeplearning4j.nn.layers.convolution.ConvolutionLayer) conf.instantiate(nnc, null, 0, pView, false, DataType.FLOAT);
//        if(layer.getHelper() != null)
//            throw new RuntimeException();
//
//        LayerWorkspaceMgr mgr = LayerWorkspaceMgr.builder()
//                .defaultWorkspace("workspace", WS_CONFIG)
//                .build();
//
//        long start = System.currentTimeMillis();
//        for( int i=0; i<numRuns; i++ ) {
//            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(WS_CONFIG, "workspace")) {
//                layer.activate(input, true, mgr);
//            }
//        }
//        long layerTime = System.currentTimeMillis() - start;
//
//        input.close();
//        weights.close();
//        bias.close();
//        out.close();
//        pView.close();
//
//        return new Timings(numRuns, opNoMKLDNN, mkldnn, layerTime);
//    }
//
//    private static Timings pool2d_NCHW(int numRuns, long[] inShape, int k, int s, boolean same, boolean max){
//        int[] kernel = {k,k};
//        int[] strides = {s,s};
//        int[] pad = {0,0};
//        int[] dilation = {1,1};
//
//        int inH = (int)inShape[2];
//        int inW = (int)inShape[3];
//
//        int nIn = (int) inShape[1];
//        INDArray input = Nd4j.createUninitialized(DataType.FLOAT, inShape);
//
//
//
//        int[] outSize;
//        ConvolutionMode convolutionMode = same ? ConvolutionMode.Same : ConvolutionMode.Truncate;
//        if (same) {
//            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation); //Also performs validation
//            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {inH, inW}, kernel, strides, dilation);
//        } else {
//            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode, dilation); //Also performs validation
//        }
//        INDArray out = Nd4j.createUninitialized(DataType.FLOAT, inShape[0], nIn, outSize[0], outSize[1]);
//
//
//        long mkldnn = 0;
//        long opNoMKLDNN = 0;
//        for(boolean mkl : new boolean[]{false, true}) {
//            Nd4jCpu.Environment.getInstance().setUseMKLDNN(mkl);
//            OpContext context = Nd4j.getExecutioner().buildContext();
//            context.setIArguments(
//                    kernel[0], kernel[1],
//                    strides[0], strides[1],
//                    pad[0], pad[1],
//                    dilation[0], dilation[1],
//                    same ? 1 : 0,
//                    0,  //Extra - not used?
//                    0); //0 = NCHW
//
//            context.setInputArray(0, input);
//            context.setOutputArray(0, out);
//            DynamicCustomOp op;
//            if (max) {
//                op = new MaxPooling2D();
//            } else {
//                op = new AvgPooling2D();
//            }
//            Nd4jCpu.Environment.getInstance().setUseMKLDNN(false);
//            long start = System.currentTimeMillis();
//            for (int i = 0; i < numRuns; i++) {
//                Nd4j.exec(op, context);
//            }
//            if(mkl){
//                mkldnn = System.currentTimeMillis() - start;
//            } else {
//                opNoMKLDNN = System.currentTimeMillis() - start;
//            }
//        }
//
//        //DL4J layer:
//        SubsamplingLayer conf = new SubsamplingLayer.Builder()
//                .convolutionMode(convolutionMode)
//                .kernelSize(k,k)
//                .stride(s,s)
//                .build();
//        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
//        nnc.setLayer(conf);
//
//
//        Nd4jCpu.Environment.getInstance().setUseMKLDNN(false);  //Disable MKLDNN so no helper
//        org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer layer = (org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer) conf.instantiate(nnc, null, 0, null, false, DataType.FLOAT);
//        if(layer.getHelper() != null)
//            throw new RuntimeException();
//
//        LayerWorkspaceMgr mgr = LayerWorkspaceMgr.builder()
//                .defaultWorkspace("workspace", WS_CONFIG)
//                .build();
//
//
//        long start = System.currentTimeMillis();
//        for( int i=0; i<numRuns; i++ ) {
//            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(WS_CONFIG, "workspace")) {
//                layer.activate(input, true, mgr);
//            }
//        }
//        long layerTime = System.currentTimeMillis() - start;
//
//        input.close();
//
//        return new Timings(numRuns, opNoMKLDNN, mkldnn, layerTime);
//    }
//
//
//
//    protected static final WorkspaceConfiguration WS_CONFIG = WorkspaceConfiguration.builder()
//            .initialSize(0)
//            .overallocationLimit(0.05)
//            .policyLearning(LearningPolicy.FIRST_LOOP)
//            .policyReset(ResetPolicy.BLOCK_LEFT)
//            .policySpill(SpillPolicy.REALLOCATE)
//            .policyAllocation(AllocationPolicy.OVERALLOCATE)
//            .build();
//}

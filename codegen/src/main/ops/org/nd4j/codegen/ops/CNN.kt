package org.nd4j.codegen.ops

import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*

fun SDCNN() =  Namespace("SDCNN"){
    val namespaceJavaPackage = "org.nd4j.linalg.api.ops.impl.layers.convolution"


    val conv1DConfig = Config("Conv1DConfig"){
        Arg(LONG, "k"){ description = "Kernel"; defaultValue=-1L}
        Arg(LONG, "s"){ description = "stride"; defaultValue=1}
        Arg(LONG, "p"){ description = "padding"; defaultValue=0}
        Arg(LONG, "d"){ description = "dilation"; defaultValue=1}
        Arg(BOOL, "isSameMode"){ description = "Same mode"; defaultValue=true}
        Arg(STRING, "dataFormat"){ description = "Data format"; defaultValue="NCW"}
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig"
    }


    val pooling2DConfig = Config("Pooling2DConfig"){
        Arg(INT, "kH"){ description = "Kernel height"; defaultValue=-1}
        Arg(INT, "kW"){ description = "Kernel width"; defaultValue=-1}
        Arg(INT, "sH"){ description = "Stride along height dimension"; defaultValue=1};
        Arg(INT, "sW"){ description = "Stride along width dimension"; defaultValue=1};
        Arg(INT, "pH"){ description = "Padding along height dimension"; defaultValue=0};
        Arg(INT, "pW"){ description = "Padding along width dimension"; defaultValue=0};
        Arg(INT, "dH"){ description = "Dilation along height dimension"; defaultValue=1};
        Arg(INT, "dW"){ description = "Dilation along width dimension"; defaultValue=1};
        Arg(BOOL, "isSameMode"){ description = "Same mode"; defaultValue=true}
        Arg(STRING, "dataFormat"){ description = "Data format"; defaultValue="nchw"}
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig"
    }

    val pooling3DConfig = Config("Pooling3DConfig"){
        Arg(INT, "kD"){ description = "Kernel depth"; defaultValue=-1}
        Arg(INT, "kW"){ description = "Kernel width"; defaultValue=-1}
        Arg(INT, "kH"){ description = "Kernel height"; defaultValue=-1};
        Arg(INT, "sD"){ description = "Stride depth"; defaultValue=1};
        Arg(INT, "sW"){ description = "Stride width"; defaultValue=1};
        Arg(INT, "sH"){ description = "Stride height"; defaultValue=1};
        Arg(INT, "pD"){ description = "Padding depth"; defaultValue=0};
        Arg(INT, "pW"){ description = "Padding width"; defaultValue=0};
        Arg(INT, "pH"){ description = "Padding height"; defaultValue=0};
        Arg(INT, "dD"){ description = "Dilation depth"; defaultValue=1};
        Arg(INT, "dW"){ description = "Dilation width"; defaultValue=1};
        Arg(INT, "dH"){ description = "Dilation height"; defaultValue=1};
        Arg(BOOL, "isSameMode"){ description = "Same mode"; defaultValue=true}
        Arg(BOOL, "isNCDHW"){ description = "isNCDHW"; defaultValue=true}
        Arg(STRING, "dataFormat"){ description = "Data format"; defaultValue="nchw"}
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig"
    }


    Op("avgPooling2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "AvgPooling2D"
        Input(NUMERIC, "input") { description = "the input to average pooling 2d operation - 4d CNN (image) activations in NCHW format\n" +
                "                        (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        useConfig(pooling2DConfig)

        Output(NUMERIC, "output"){ description = "Result after applying average pooling on the input" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D Convolution layer operation - average pooling 2d
            """.trimIndent()
        }
    }

    Op("avgPooling3d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "AvgPooling3D"
        Input(NUMERIC, "input") { description = "the input to average pooling 3d operation - 5d activations in NCDHW format\n" +
                "                        (shape [minibatch, channels, depth, height, width]) or NDHWC format\n" +
                "                        (shape [minibatch, depth, height, width, channels])" }
        useConfig(pooling3DConfig)

        Output(NUMERIC, "output"){ description = "after applying average pooling on the input" }

        Doc(Language.ANY, DocScope.ALL){
        """
         3D convolution layer operation - average pooling 3d 
        """.trimIndent()
        }
    }

    Op("batchToSpace") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "BatchToSpace"
        Input(NUMERIC, "x") { description = "Input variable. 4d input" }
        Input(NUMERIC, "blocks") { description = "Block size, in the height/width dimension" }
        Input(NUMERIC, "crops") { description = "Optional 2d int[] array: values [[crop top, crop bottom], [crop left, crop right]]" }

        Output(NUMERIC, "output"){ description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Convolution 2d layer batch to space operation on 4d input.
             Reduces input batch dimension by rearranging data into a larger spatial dimensions
            """.trimIndent()
        }
        }

    Op("col2Im") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Col2Im"

        Input(NUMERIC, "in") { description = "Input - rank 6 input with shape [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth]" }
        Input(NUMERIC, "config") { description = "Convolution configuration for the col2im operation" }

        Output(NUMERIC, "output"){ description = "Col2Im output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             col2im operation for use in 2D convolution operations. Outputs a 4d array with shape
             [minibatch, inputChannels, height, width]
            """.trimIndent()
        }
    }

    Op("conv1d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Conv1D"
        Input(NUMERIC, "input") { description = "the inputs to conv1d" }
        Input(NUMERIC, "weights") { description = "weights for conv1d op - rank 3 array with shape [kernelSize, inputChannels, outputChannels]" }
        useConfig(conv1DConfig)
        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #conv1d(String, SDVariable, SDVariable, SDVariable, Conv1DConfig)}, no bias.
     
""".trimIndent()
        }
    }

    Op("conv1d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Conv1D"
        Input(NUMERIC, "input") { description = "the inputs to conv1d" }
        Input(NUMERIC, "weights") { description = "weights for conv1d op - rank 3 array with shape [kernelSize, inputChannels, outputChannels]" }
        Input(NUMERIC, "bias") { description = "bias for conv1d op - rank 1 array with shape [outputChannels]. May be null." }
        useConfig(conv1DConfig)

        Output(NUMERIC, "output"){ description = "result of conv1d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Conv1d operation.
            """.trimIndent()
        }
    }

    Op("conv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Conv2D"
        Input(NUMERIC, "layerInput") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format" }
        Input(NUMERIC, "weights") { description = "Weights for the convolution operation. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, outputChannels]" }
        Input(NUMERIC, "config") { description = " Conv2DConfig configuration" }

        Output(NUMERIC, "output"){ description = "result of conv2d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             See {@link #conv2d(String, SDVariable, SDVariable, SDVariable, Conv2DConfig)}, no bias.
            """.trimIndent()
        }
    }

    Op("conv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Conv2D"
        Input(NUMERIC, "layerInput") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format" }
        Input(NUMERIC, "weights") { description = "Weights for the convolution operation. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, outputChannels]" }
        Input(NUMERIC, "bias") { description = "Optional 1D bias array with shape [outputChannels]. May be null." }
        Input(NUMERIC, "config") { description = " Conv2DConfig configuration" }

        Output(NUMERIC, "output"){ description = "result of conv2d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D Convolution operation with optional bias
            """.trimIndent()
        }
    }

    Op("conv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Conv2D"
        Input(NUMERIC, "inputs") { description = "inputs an array with either 2 elements (layerInput, weights) or 3 elements (layerInput, weights, bias) as\n" +
                "               described in {@link #conv2d(SDVariable, SDVariable, SDVariable, Conv2DConfig)}" }
        Input(NUMERIC, "config") { description = "onfig Conv2DConfig configuration" }

        Output(NUMERIC, "output"){ description = "result of convolution 2d operation" }

        Doc(Language.ANY, DocScope.ALL){
            """
         2D Convolution operation with optional bias
            
        """.trimIndent()
        }
    }

    Op("conv3d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Conv3D"
        Input(NUMERIC, "input") { description = "the input to average pooling 3d operation - 5d activations in NCDHW format\n" +
                "(shape [minibatch, channels, depth, height, width]) or NDHWC format\n" +
                "(shape [minibatch, depth, height, width, channels])" }
        Input(NUMERIC, "weights") { description = " Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels]." }
        Input(NUMERIC, "conv3DConfig") { description = "3DConfig the configuration" }

        Output(NUMERIC, "output"){ description = "Conv3d output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             See {@link #conv3d(String, SDVariable, SDVariable, SDVariable, Conv3DConfig)}, no bias.
            """.trimIndent()
        }
    }

    Op("conv3d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Conv3D"
        Input(NUMERIC, "input") { description = "the input to average pooling 3d operation - 5d activations in NCDHW format\n" +
                "(shape [minibatch, channels, depth, height, width]) or NDHWC format\n" +
                "(shape [minibatch, depth, height, width, channels])" }
        Input(NUMERIC, "weights") { description = " Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels]." }
        Input(NUMERIC, "bias") { description = " Optional 1D bias array with shape [outputChannels]. May be null." }
        Input(NUMERIC, "conv3DConfig") { description = "3DConfig the configuration" }

        Output(NUMERIC, "output"){ description = "Conv3d output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Convolution 3D operation with optional bias 
            """.trimIndent()
        }
    }

    Op("deconv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DeConv2D"
        Input(NUMERIC, "layerInput") { description = "the input to deconvolution 2d operation - 4d CNN (image) activations in NCHW format\n" +
                "(shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }

        Input(NUMERIC, "weights") { description = "Weights for the 2d deconvolution operation. 4 dimensions with format [inputChannels, outputChannels, kernelHeight, kernelWidth]" }
        Input(NUMERIC, "weights") { description = "Weights for the 2d deconvolution operation. 4 dimensions with format [inputChannels, outputChannels, kernelHeight, kernelWidth]" }

        Output(NUMERIC, "output"){ description = "result of deconv2d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             See {@link #deconv2d(String, SDVariable, SDVariable, SDVariable, DeConv2DConfig)}, no bias.
            """.trimIndent()
        }
    }

    Op("deconv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DeConv2D"
        Input(NUMERIC, "layerInput") { description = "the input to deconvolution 2d operation - 4d CNN (image) activations in NCHW format\n" +
                "(shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        Input(NUMERIC, "weights") { description = "Weights for the 2d deconvolution operation. 4 dimensions with format [inputChannels, outputChannels, kernelHeight, kernelWidth]" }
        Input(NUMERIC, "bias") { description = "Optional 1D bias array with shape [outputChannels]. May be null." }
        Input(NUMERIC, "weights") { description = "Weights for the 2d deconvolution operation. 4 dimensions with format [inputChannels, outputChannels, kernelHeight, kernelWidth]" }

        Output(NUMERIC, "output"){ description = "result of deconv2d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D deconvolution operation with optional bias
            """.trimIndent()
        }
    }

    Op("deconv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DeConv2D"
        Input(NUMERIC, "inputs") { description = "Inputs to the deconvolution 2d operation - input array of length 2 (layerInput, weights)\n" +
                "                       or length 3 (layerInput, weights, bias) as described in {@link #deconv2d(SDVariable[], DeConv2DConfig)}" }
        Input(NUMERIC, "deconv2DConfig") { description = "deconv2DConfig the configuration" }

        Output(NUMERIC, "output"){ description = "result of deconv2d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D deconvolution operation with or without optional bias 
            """.trimIndent()
        }
    }

    Op("deconv3d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DeConv3D"
        Input(NUMERIC, "input") { description = "Input array - shape [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)" }
        Input(NUMERIC, "weights") { description = "Weights array - shape [kD, kH, kW, oC, iC]" }
        Input(NUMERIC, "config") { description = "Configuration" }

        Output(NUMERIC, "output"){ description = "result of 3D CNN deconvolution operation" }

        Doc(Language.ANY, DocScope.ALL){
            """
             See {@link #deconv3d(String, SDVariable, SDVariable, SDVariable, DeConv3DConfig)}, no bias.
            """.trimIndent()
        }
    }

    Op("deconv3d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DeConv3D"
        Input(NUMERIC, "input") { description = "Input array - shape [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)" }
        Input(NUMERIC, "weights") { description = "Weights array - shape [kD, kH, kW, oC, iC]" }
        Input(NUMERIC, "bias") { description = "Bias array - optional, may be null. If non-null, must have shape [outputChannels]" }
        Input(NUMERIC, "config") { description = "Configuration" }

        Output(NUMERIC, "output"){ description = "result of 3D CNN deconvolution operation" }

        Doc(Language.ANY, DocScope.ALL){
            """
             3D CNN deconvolution operation with or without optional bias
            """.trimIndent()
        }
    }

    Op("depthToSpace") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DepthToSpace"
        Input(NUMERIC, "x") { description = "the input to depth to space pooling 2d operation - 4d activations in NCHW format\n" +
                "                   (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        Input(NUMERIC, "blockSize") { description = "Block size, in the height/width dimension" }
        Input(NUMERIC, "dataFormat") { description = "Data format: NCHW or NHWC" }

        Output(NUMERIC, "output"){ description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Convolution 2d layer batch to space operation on 4d input.<br>
             Reduces input channels dimension by rearranging data into a larger spatial dimensions<br>
             Example: if input has shape [mb, 8, 2, 2] and block size is 2, then output size is [mb, 8/(2*2), 2*2, 2*2]
             = [mb, 2, 4, 4]
            """.trimIndent()
        }
    }

    Op("depthWiseConv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DepthwiseConv2D"
        Input(NUMERIC, "layerInput") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format" }
        Input(NUMERIC, "depthWeights") { description = "Depth-wise conv2d weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier]" }
        Input(NUMERIC, "config") { description = "Conv2DConfig configuration" }

        Output(NUMERIC, "output"){ description = "result of depthwise conv2d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             See {@link #depthWiseConv2d(String, SDVariable, SDVariable, SDVariable, Conv2DConfig)}, no bias.
            """.trimIndent()
        }
    }

    Op("depthWiseConv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DepthwiseConv2D"
        Input(NUMERIC, "layerInput") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format" }
        Input(NUMERIC, "depthWeights") { description = "Depth-wise conv2d weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier]" }
        Input(NUMERIC, "bias") { description = "Optional 1D bias array with shape [outputChannels]. May be null." }
        Input(NUMERIC, "config") { description = "Conv2DConfig configuration" }

        Output(NUMERIC, "output"){ description = "result of depthwise conv2d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Depth-wise 2D convolution operation with optional bias 
            """.trimIndent()
        }
    }

    Op("depthWiseConv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DepthwiseConv2D"
        Input(NUMERIC, "inputs") { description = "the inputs to depth-wise conv2d. An array with either 2 elements (layerInput, depthWeights)\n" +
                "                          or 3 elements (layerInput, depthWeights, bias) as described in\n" +
                "                          {@link #depthWiseConv2d(SDVariable, SDVariable, SDVariable, Conv2DConfig)}" }
        Input(NUMERIC, "depthConv2DConfig") { description = "depthConv2DConfig the configuration" }

        Output(NUMERIC, "output"){ description = "result of depthwise conv2d op" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Depth-wise convolution 2D operation.
            """.trimIndent()
        }
    }

    Op("dilation2D") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "Dilation2D"
        Input(NUMERIC, "df") { description = "" }
        Input(NUMERIC, "weights") { description = "df" }
        Input(NUMERIC, "strides") { description = "weights" }
        Input(NUMERIC, "rates") { description = "strides" }
        Input(NUMERIC, "isSameMode") { description = "isSameMode" }

        Output(NUMERIC, "output"){ description = "Computed the grayscale dilation of 4-D input and 3-D filters tensors." }

        Doc(Language.ANY, DocScope.ALL){
            """
             TODO doc string
            """.trimIndent()
        }
    }

    Op("extractImagePatches") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.image"
        javaOpClass = "ExtractImagePatches"
        Input(NUMERIC, "input") { description = "Input array. Must be rank 4, with shape [minibatch, height, width, channels]" }
        Input(NUMERIC, "kH") { description = "Kernel height" }
        Input(NUMERIC, "kW") { description = "Kernel width" }
        Input(NUMERIC, "sH") { description = "Stride height" }
        Input(NUMERIC, "sW") { description = "Stride width" }
        Input(NUMERIC, "rH") { description = "Rate height" }
        Input(NUMERIC, "rW") { description = "Rate width" }
        Input(NUMERIC, "sameMode") { description = "If true: use same mode padding. If false" }

        Output(NUMERIC, "output"){ description = "The result is a 4D tensor which is indexed by batch, row, and column." }

        Doc(Language.ANY, DocScope.ALL){
            """
             Extract image patches 
            """.trimIndent()
        }
    }

    Op("im2Col") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "Im2col"
        Input(NUMERIC, "in") { description = "Input - rank 4 input with shape [minibatch, inputChannels, height, width]" }
        Input(NUMERIC, "config") { description = "config Convolution configuration for the im2col operation" }

        Output(NUMERIC, "output"){ description = "Im2Col output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             im2col operation for use in 2D convolution operations. Outputs a 6d array with shape
             [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth]   
            """.trimIndent()
        }
    }

    Op("localResponseNormalization") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "LocalResponseNormalization"
        Input(NUMERIC, "input") { description = "the inputs to lrn" }
        Input(NUMERIC, "lrnConfig") { description = "lrnConfig the configuration" }

        Output(NUMERIC, "output"){ description = "Result after Local Response Normalization"}

        Doc(Language.ANY, DocScope.ALL){
            """
             2D convolution layer operation - local response normalization
            """.trimIndent()
        }
    }

    Op("maxPooling2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "MaxPooling2D"
        Input(NUMERIC, "input") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format\n" +
                "                        (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        Input(NUMERIC, "pooling2DConfig") { description = "pooling2DConfig the configuration" }

        Output(NUMERIC, "output"){ description = "Result after applying max pooling on the input" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D Convolution layer operation - max pooling 2d 
            """.trimIndent()
        }
    }

    Op("maxPooling3d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "MaxPooling3D"
        Input(NUMERIC, "input") { description = "the input to average pooling 3d operation - 5d activations in NCDHW format\n" +
                "                        (shape [minibatch, channels, depth, height, width]) or NDHWC format\n" +
                "                        (shape [minibatch, depth, height, width, channels])" }
        Input(NUMERIC, "pooling3DConfig") { description = "pooling3DConfig the configuration" }

        Output(NUMERIC, "output"){ description = "Result after applying max pooling on the input" }

        Doc(Language.ANY, DocScope.ALL){
            """
             3D convolution layer operation - max pooling 3d operation.
            """.trimIndent()
        }
    }

    Op("separableConv2d") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "separableConv2d"
        Input(NUMERIC, "layerInput") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format\n" +
                "                     (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        Input(NUMERIC, "depthWeights") { description = "Separable conv2d depth weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier]" }
        Input(NUMERIC, "pointWeights") { description = "Point weights, rank 4 with format [1, 1, inputChannels*depthMultiplier, outputChannels]. May be null" }
        Input(NUMERIC, "config") { description = "Conv2DConfig configuration" }

        Output(NUMERIC, "output"){ description = "result of separable convolution 2d operation" }

        Doc(Language.ANY, DocScope.ALL){
            """
             See {@link #separableConv2d(String, SDVariable, SDVariable, SDVariable, SDVariable, Conv2DConfig)}, no bias.
            """.trimIndent()
        }
    }

    Op("separableConv2d") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "layerInput") { description = "the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format\n" +
                "                     (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        Input(NUMERIC, "depthWeights") { description = "Separable conv2d depth weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier]" }
        Input(NUMERIC, "pointWeights") { description = "Point weights, rank 4 with format [1, 1, inputChannels*depthMultiplier, outputChannels]. May be null" }
        Input(NUMERIC, "bias") { description = "Optional bias, rank 1 with shape [outputChannels]. May be null." }
        Input(NUMERIC, "config") { description = "Conv2DConfig configuration" }

        Output(NUMERIC, "output"){ description = "result of separable convolution 2d operation" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Separable 2D convolution operation with optional bias 
            """.trimIndent()
        }
    }

    Op("sconv2d") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "inputs") { description = "the inputs to separable conv2 operation. Should be length 3 (layerInput, depthWeights, pointWeights)\n" +
                "                     or length 4 (layerInput, depthWeights, pointWeights, bias) as described in {@link #separableConv2d(SDVariable, SDVariable, SDVariable, SDVariable, Conv2DConfig)}" }
        Input(NUMERIC, "conv2DConfig") { description = "conv2DConfig the configuration" }

        Output(NUMERIC, "output"){ description = "result of separable convolution 2d operation" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Separable 2D convolution operation with/without optional bias
            """.trimIndent()
        }
    }

    Op("spaceToBatch") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "Input variable. 4d input" }
        Input(NUMERIC, "blocks") { description = "Block size, in the height/width dimension" }
        Input(NUMERIC, "padding") { description = "Optional 2d int[] array for padding the result: values [[pad top, pad bottom], [pad left, pad right]]" }

        Output(NUMERIC, "output"){ description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Convolution 2d layer space to batch operation on 4d input.
             Increases input batch dimension by rearranging data from spatial dimensions into batch dimension 
            """.trimIndent()
        }
    }

    Op("spaceToDepth") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "the input to depth to space pooling 2d operation - 4d activations in NCHW format\n" +
                "                   (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])" }
        Input(NUMERIC, "blockSize") { description = " Block size, in the height/width dimension" }
        Input(NUMERIC, "dataFormat") { description = "Data format: \"NCHW\" or \"NHWC\"" }

        Output(NUMERIC, "output"){ description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Convolution 2d layer space to depth operation on 4d input.<br>
             Increases input channels (reduced spatial dimensions) by rearranging data into a larger channels dimension<br>
             Example: if input has shape [mb, 2, 4, 4] and block size is 2, then output size is [mb, 8/(2*2), 2*2, 2*2]
             = [mb, 2, 4, 4] 
            """.trimIndent()
        }
    }

    Op("upsampling2d") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "Input in NCHW format" }
        Input(NUMERIC, "scale") { description = "The scale for both height and width dimensions." }

        Output(NUMERIC, "output"){ description = "Upsampled input"}

        Doc(Language.ANY, DocScope.ALL){
            """
             See {@link #upsampling2d(String, SDVariable, boolean, int, int)},
             scale is used for both height and width dimensions. 
            """.trimIndent()
        }
    }

    Op("upsampling2d") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "Input in NCHW format" }
        Input(NUMERIC, "nchw") { description = "If true: input is in NCHW (minibatch, channels, height, width) format. False: NHWC format" }
        Input(NUMERIC, "scaleH") { description = "Scale to upsample in height dimension" }
        Input(NUMERIC, "scaleW") { description = "Scale to upsample in width dimension" }

        Output(NUMERIC, "output"){ description = "Upsampled input" }

        Doc(Language.ANY, DocScope.ALL){
            """
             2D Convolution layer operation - Upsampling 2d 
            """.trimIndent()
        }
    }
}
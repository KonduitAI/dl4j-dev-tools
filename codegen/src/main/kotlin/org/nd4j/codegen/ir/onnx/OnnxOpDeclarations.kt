package org.nd4j.codegen.ir.onnx

import onnx.Onnx
import org.nd4j.codegen.ir.ArgDescriptor
import org.nd4j.codegen.ir.AttributeMappingRule
import org.nd4j.codegen.ir.nd4jOpDescriptors
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.codegen.ir.tensorflow.TensorflowMappingProcess
import org.nd4j.codegen.ir.tensorflow.defineTensorflowSingleTransform
import org.nd4j.codegen.ir.tensorflow.tensorflowOpRegistry

val onnxOpRegistry = OpMappingRegistry<Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto>("onnx")
val names = mapOf(
        "Abs" to "abs",
        "Acos" to "acos",
        "Acosh" to "acosh",
        "Asin" to "asin",
        "Asinh" to "asinh",
        "Atan" to "atan",
        "Atanh" to "atanh",
        "Ceil" to "ceil",
        "Cos" to "cos",
        "Cosh" to "cosh",
        "Erf" to "erf",
        "Exp" to "exp",
        "Floor" to "floor",
        "Identity" to "identity",
        "IsNaN" to "isnan",
        "Log" to "log",
        "LogSoftmax" to "log_softmax",
        "Mod" to "mod",
        "Neg" to "neg",
        "Relu" to "relu",
        "Round" to "round",
        "Sigmoid" to "sigmoid",
        "Sign" to "sign",
        "Sin" to "sin",
        "Sinh" to "sinh",
        "Softmax" to "softmax",
        "Softsign" to "softsign",
        "Sqrt" to "sqrt",
        "Tan" to "tan",
        "Tanh" to "tanh"

)

val pairWiseNames = mapOf(
        "Add" to "add",
        "And" to "boolean_and",
        "Div" to "divide",
        "Equal" to "equals",
        "Greater" to "greater",
        "GreaterOrEqual" to "greater_equal",
        "Less" to "less",
        "LessOrEqual" to "less_equal",
        "Mul" to "multiply",
        "Not" to "not",
        "Or" to "or",
        "Sub" to "subtract",
        "Xor" to "xor"
)


//Adagrad
//Adam


//unmapped: select_last_index
val argMax = OnnxMappingProcess(
        opName = "argmax",
        inputFrameworkOpName = "ArgMax",
        tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims","dimensions" to "axis"))),
        opMappingRegistry = onnxOpRegistry
)

//unmapped: select_last_index
val argMin = OnnxMappingProcess(
        opName = "argmin",
        inputFrameworkOpName = "ArgMin",
        tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims","dimensions" to "axis"))),
        opMappingRegistry = onnxOpRegistry
)


//TODO:
/**
 * input: "X"
input: "Y"
output: "Z"
name: "ArrayFeatureExtractor"
op_type: "ArrayFeatureExtractor"
attribute {
name: "X-types"
strings: "int64"
strings: "string"
strings: "double"
strings: "float"
strings: "int32"
type: STRINGS
}
attribute {
name: "Y-types"
strings: "int64"
type: STRINGS
}
doc_string: "\n    Select elements of the input tensor based on the indices passed.<br>\n    The indices are applied to the last axes of the tensor.\n"
--

 */

//Note:  weight formats are NCHW in ONNX
val avgPool = OnnxMappingProcess(
        inputFrameworkOpName = "AveragePool",
        opName = "avgpool2d",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(
                argDescriptorConstant(argDescriptorConstants = listOf(ArgDescriptor {
                    name = "isNCHW"
                    boolValue = true
                })),
                stringContainsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "auto_pad",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH",inputFrameworkAttributeName = "dilations",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW",inputFrameworkAttributeName = "dilations",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "pH",inputFrameworkAttributeName = "padding",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "pW",inputFrameworkAttributeName = "padding",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH",inputFrameworkAttributeName = "kernel_shape",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW",inputFrameworkAttributeName = "kernel_shape",targetValue = "NCHW",trueIndex = 3,falseIndex = 2)))

val batchNorm = OnnxMappingProcess(
        opName = "batchnorm",
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "BatchNormalization",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X","mean" to "mean","variance" to "var","gamma" to "scale"))),
        attributeMappingRules = listOf(valueMappings(mapOf("epsilon" to "epsilon")))
)
//TODO: Binarizer
//TODO: Bitshift
//TODO: Cast
//TODO: CastMap
//TODO: CategoryMapper
//TODO: Celu
//TODO: Clip
//TODO: Compress
val concat = OnnxMappingProcess(
        opName = "concat",
        inputFrameworkOpName = "Concat",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "inputs"))),
        attributeMappingRules = listOf(valueMappings(mapOf("concatDimension" to "axis")))

)
//TODO: ConcatFromSequence
//TODO: Constant
//TODO: ConstantOfShape
//TODO: ConvInteger
//TODO: ConvTranspose
val cumSum = OnnxMappingProcess(
        opName = "cumsum",
        inputFrameworkOpName = "CumSum",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x"))),
        attributeMappingRules = listOf(valueMappings(mapOf("exclusive" to "exclusive","reverse" to "reverse")), ndarrayToIntList(ndarrayNameToAttributeName = mutableMapOf("axis" to "axis")))
)

val depthToSpace = OnnxMappingProcess(
        opName = "depth_to_space",
        inputFrameworkOpName = "DepthToSpace",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMappings(mapOf("block_size" to "block_size")),
                stringEqualsRule("isNWHC", inputFrameworkAttributeName = "data_format", valueToTest = "NWHC")),
        opMappingRegistry = onnxOpRegistry
)

//TODO: DequantizeLinear
//TODO: Det
//TODO: DictVectorizer
//TODO: Dropout: Note https://github.com/eclipse/deeplearning4j/issues/5650
val dropout = OnnxMappingProcess(
        opName = "dropout_inverted",
        inputFrameworkOpName = "Dropout",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        opMappingRegistry = onnxOpRegistry
)
//TODO: DynamicQuantizeLinear
//TODO: Einsum
//TODO: Expand
//TODO: EyeLike
//TODO: FeatureVectorizer
//TODO: Flatten
val gru = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "GRU",
        opName = "gruCell",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "x","ruWeight" to "R","cWeight" to "W","cBias" to "B"))),
        attributeMappingRules = listOf()
)

val gather = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Gather",
        opName = "gather",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("indices" to "indices","input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("dimensions" to "axis")))
)
//TODO: GatherElements
val gatherNd = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "GatherND",
        opName = "gather_nd",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("indices" to "indices","input" to "data")))
)


val gemm = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Gemm",
        opName = "mmul",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "A","y" to "B"))),
        attributeMappingRules = listOf(valueMappings(mapOf("alpha" to "alpha","beta" to "beta","transposeX" to "transA","transposeY" to "transB")))
)
//TODO: GlobalAveragePool
//TODO: GlobalLpPool
//TODO: GlobalMaxPool
//TODO: Gradient
//TODO: GraphCall
val hardSigmoid = OnnxMappingProcess(
        opName =  "hard_sigmoid",
        inputFrameworkOpName = "HardSigmoid",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "X")))
)


//        "IsInf" to "isinf",

//TODO: map is-negative,is-positive
val isInf = OnnxMappingProcess(
        opName = "isinf",
        inputFrameworkOpName = "IsInf",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X")))
)

//TODO: Hardmax
//TODO: If
//TODO: Imputer
//TODO: InstanceNormalization
val lrn = OnnxMappingProcess(
        opName = "lrn",
        inputFrameworkOpName = "LRN",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = listOf(valueMappings(mapOf("alpha" to "alpha","beta" to "beta","bias" to "bias","depth" to "size")))

)

//TODO: Need to figure out how to map LSTMLayerCOnfig DirectionMode to the strings in direction (forward, reverse, or bidirectional)
//TODO: Need to map gateAct, cellAct, outAct enums to strings in activations. Valid activation functions include:
/*

        TANH,
        RELU,
        SIGMOID,
        AFFINE,
        LEAKY_RELU,
        THRESHHOLD_RELU,
        SCALED_TAHN,
        HARD_SIGMOID,
        ELU,
        SOFTSIGN,
        SOFTPLUS
 */

//Onnx supports;

/**
 * Relu(x)                - max(0, x)

Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

Sigmoid(x)             - 1/(1 + e^{-x})

(NOTE: Below are optional)

Affine(x)              - alpha*x + beta

LeakyRelu(x)           - x if x >= 0 else alpha * x

ThresholdedRelu(x)     - x if x >= alpha else 0

ScaledTanh(x)          - alpha*Tanh(beta*x)

HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

Softsign(x)            - x/(1 + |x|)

Softplus(x)            - log(1 + e^x)
 */
//Need to figure out how to pass in activation alpha and beta as lists
//These are only used with some of the activations. The alpha/beta vary by activation function.

//TODO: Figure out if hidden_size is relevant
//TODO: figure out if input_forget is relevant

//TODO: Use listNumberToListNumber for float activations (alphas/betas) for model import
//TODO: Note we may *Not* support custom alphas/betas like what onnx does here
//We support: S
/**
 *  const auto gateActHasAlpha = gateAct == 3 || gateAct == 4 || gateAct == 5 || gateAct == 6 || gateAct == 8;
const auto cellActHasAlpha = cellAct == 3 || cellAct == 4 || cellAct == 5 || cellAct == 6 || cellAct == 8;
const auto outActHasAlpha  = outAct  == 3 || outAct  == 4 || outAct  == 5 || outAct  == 6 || outAct  == 8;
const auto gateActHasBeta  = gateAct == 3 || gateAct == 6;
const auto cellActHasBeta  = cellAct == 3 || cellAct == 6;
const auto outActHasBeta   = outAct  == 3 || outAct  == 6;

uint count = 1;
const auto cellClip = T_ARG(0);                                     // cell clipping value, if it = 0 then do not apply clipping
const auto gateAlpha = gateActHasAlpha ? T_ARG(count++) : 0;
const auto gateBeta  = gateActHasBeta  ? T_ARG(count++) : 0;
const auto cellAlpha = cellActHasAlpha ? T_ARG(count++) : 0;
const auto cellBeta  = cellActHasBeta  ? T_ARG(count++) : 0;
const auto outAlpha  = outActHasAlpha  ? T_ARG(count++) : 0;
const auto outBeta   = outActHasBeta   ? T_ARG(count++) : 0;
 */
/**
 *  const auto gateAct       = INT_ARG(2);    // activation for input (i), forget (f) and output (o) gates
const auto cellAct       = INT_ARG(3);    // activation for cell state (c)
const auto outAct        = INT_ARG(4);    // activation for output (h)

 */

val lstm = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "LSTM",
        opName = "lstmLayer",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "x" to "x",
                "W" to "iWeights",
                "iRWeights" to "R",
                "iBias" to "B",
                "maxTSLength" to "sequence_lens",
                "yLast" to "initial_h",
                "cLast" to "initial_c"))),
        attributeMappingRules =  listOf(valueMappings(mapOf("cellClip" to "clip")),
                stringToIndex(outputAttributeValue = "directionMode",
                        inputAttributeValue = "direction",
                        listOfValues = listOf("forward","reverse","bidirectional")),
                listAttributeValueLookup(outputAttributeValue = "gateAlpha",inputAttributeValue = "activation_alpha",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "cellAlpha",inputAttributeValue = "activation_alpha",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "outAlpha",inputAttributeValue = "acitvation_alpha",indexValue = 2),
                listAttributeValueLookup(outputAttributeValue = "gateBeta",inputAttributeValue = "activation_beta",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "cellBeta",inputAttributeValue = "activation_beta",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "outBeta",inputAttributeValue = "activation_beta",indexValue = 2),
                listAttributeValueLookup(outputAttributeValue = "gateAct",inputAttributeValue = "activations",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "cellAct",inputAttributeValue = "activations",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "outAct",inputAttributeValue = "activations",indexValue = 2))
)
//TODO: LabelEncoder
//TODO: DID NOT PICK UP ALPHA PROPERTY
val leakyRelu = OnnxMappingProcess(
        inputFrameworkOpName = "LeakyRelu",
        opName = "leakyrelu",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "x"))),
        attributeMappingRules = listOf(valueMappings(mapOf("alpha" to "alpha"))),
        opMappingRegistry = onnxOpRegistry
)
//TODO: LinearClassifier
//TODO: LinearRegressor
//TODO: Loop
//TODO: LpNormalization
//TODO: LpPool
val matMul = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "MatMul",
        opName = "mmul",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "A","y" to "B")))
)
//TODO: MatMulInteger
//TODO: Max
val maxPool = OnnxMappingProcess(
        inputFrameworkOpName = "MaxPool",
        opName = "maxpool2d",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = listOf(
                argDescriptorConstant(argDescriptorConstants = listOf(ArgDescriptor {
                    name = "isNCHW"
                    boolValue = true
                })),
                stringContainsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "auto_pad",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH",inputFrameworkAttributeName = "dilations",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW",inputFrameworkAttributeName = "dilations",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "pH",inputFrameworkAttributeName = "pads",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "pW",inputFrameworkAttributeName = "pads",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH",inputFrameworkAttributeName = "kernel_shape",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW",inputFrameworkAttributeName = "kernel_shape",targetValue = "NCHW",trueIndex = 3,falseIndex = 2)))


//TODO: MaxRoiPool
//TODO: MaxUnpool
//TODO: name: "MeanVarianceNormalization"
//todo: Momentum
//TODO: Multinomial
//TODO: NegativeLogLikelihoodLoss
val nonMaxSuppression = OnnxMappingProcess(
        inputFrameworkOpName = "NonMaxSuppression",
        opName = "non_max_suppression_v3",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "boxes" to "boxes",
                "scores" to "scores",
                "maxOutSize" to "max_output_boxes_per_class",
                "iouThreshold" to "iou_threshold",
        "scoreThreshold" to "score_threshold")))
)
//TODO: NonZero PRIORITIZE
//TODO: Normalizer
//TODO: OneHot
//TODO: OneHotEncoder
val pRelu = OnnxMappingProcess(
        inputFrameworkOpName = "PRelu",
        opName = "prelu",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X","slope" to "alpha"))),
        opMappingRegistry = onnxOpRegistry
)

val pad = OnnxMappingProcess(
        inputFrameworkOpName = "Pad",
        opMappingRegistry = onnxOpRegistry,
        opName = "pad",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("in" to "data","padding" to "pads"))),
        attributeMappingRules = listOf(stringToIndex(outputAttributeValue = "mode",inputAttributeValue = "mode",listOfValues = listOf("constant","reflect","edge")))
)

//TODO: QLinearConv
//TODO: QLinearMatMul
//TODO: QuantizeLinear
//TODO: RNN PRIORITIZE
val randomNormal = OnnxMappingProcess(
        inputFrameworkOpName = "RandomNormal",
        opName = "random_normal",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(valueMappings(mapOf("mean" to "mean","stdev" to "scale")),
                convertNumberListToInputNDArray(outputAttributeValue = "shape",inputAttributeValue = "shape"))
)


//TODO: RandomNormalLike
//TODO: Note that the attributes for random unifrom are wrong and needed to be discovered through other means.
//The combination of a lack of a java class + the c++ calling out to other functions which had the actual parameters
//names prevented resolution of the real parameter names. May have to look in to values that are passed inline in to functions and look up
//parameter names that way.

val randomUniform = OnnxMappingProcess(
        inputFrameworkOpName = "RandomUniform",
        opName = "random_uniform",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(valueMappings(mapOf("mean" to "mean","stdev" to "scale")),
                convertNumberListToInputNDArray(outputAttributeValue = "shape",inputAttributeValue = "shape"))
)

//TODO: RandomUniformLike
val range = OnnxMappingProcess(
        inputFrameworkOpName = "Range",
        opName = "range",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(valueMappings(mapOf("s" to "start","to" to "limit","d" to "delta")))
)

val norm1 = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceL1",
        opMappingRegistry = onnxOpRegistry,
        opName = "reduce_norm1",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "axesVector",inputAttributeValue = "axes"))

)

val norm2 = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceL2",
        opMappingRegistry = onnxOpRegistry,
        opName = "reduce_norm2",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "axesVector",inputAttributeValue = "axes"))
)

//TODO: ReduceLogSum
val reduceLogSumExp = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceLogSumExp",
        opName = "reduce_logsumexp",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "axesVector",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceMax = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMax",
        opName = "reduce_max",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "axesVector",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceMean = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMean",
        opName = "reduce_mean",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "axesVector",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceMin = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMin",
        opName = "reduce_min",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "axesVector",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceProd = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceProd",
        opName = "reduce_prod",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "axesVector",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)

val reduceSum = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceSum",
        opName = "reduce_sum",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "axesVector",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
//TODO: ReduceSumSquare
//TODO: Resize PRIORITIZE
//TODO: ReverseSequence
//TODO: RoiAlign
//TODO: SVMClassifier
//TODO: SVMRegressor
//TODO: Scaler
//TODO: Scan
val scatter = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "ScatterElements",
        opName = "scatter_upd",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("ref" to "data","indices" to "indices","updates" to "updates")))
)

val scatterNd = OnnxMappingProcess(
        opName = "scatter_nd_update",
        inputFrameworkOpName = "ScatterNd",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data","indices" to "indices","updates" to "updates"))),
        opMappingRegistry = onnxOpRegistry
)

//TODO: SequenceAt
//TODO: SequenceConstruct
//TODO: SequenceErase
//TODO: SequenceInsert
//TODO: SequenceLength
val shape = OnnxMappingProcess(
        opName = "shape_of",
        inputFrameworkOpName = "Shape",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "data"))))
)
//TODO: Shrink

val not = OnnxMappingProcess(
        opName = "not",
        inputFrameworkOpName = "not",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "X","y" to "Y"))))
)


val pow = OnnxMappingProcess(
        opName = "pow",
        inputFrameworkOpName = "Pow",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "X","y" to "Y"))))
)

val size = OnnxMappingProcess(
        opName = "size",
        inputFrameworkOpName = "Size",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "data"))))
)

//TODO: map axes
val slice = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Slice",
        opName = "strided_slice",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("begin" to "starts","end" to "ends","strides" to "steps")))
)


//TODO: SoftmaxCrossEntropyLoss
val spaceToDepth = OnnxMappingProcess(
        opName = "space_to_depth",
        inputFrameworkOpName = "SpaceToDepth",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMappings(mapOf("block_size" to "block_size")),
                argDescriptorConstant(listOf(ArgDescriptor {
                    name = "isNCHW"
                    boolValue = true
                }))),
        opMappingRegistry = onnxOpRegistry
)

val split = OnnxMappingProcess(
        opName = "split",
        inputFrameworkOpName = "Split",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("a" to "","b" to "split"))),
        attributeMappingRules = listOf(valueMappings(mapOf("splitDim" to "axis")))
)

val softplus = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Softplus",
        opName = "softplus",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X")))
)

//TODO: SplitToSequence
val squeeze = OnnxMappingProcess(
        opName = "squeeze",
        inputFrameworkOpName = "Squeeze",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(ndarrayToIntList(ndarrayNameToAttributeName = mutableMapOf("squeezeDims" to "axes")))
)

//TODO: StringNormalizer
//TODO: TfIdfVectorizer
//TODO: ThresholdedRelu
val tile = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Tile",
        opName = "tile",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","reps_vector" to "repeats")))
)

val topK = OnnxMappingProcess(
        opName = "top_k",
        inputFrameworkOpName = "TopK",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = listOf(valueMappings(mapOf("needSort" to "sorted")), convertNDArrayInputToScalarAttr(outputAttributeValue = "k",inputAttributeValue = "K")),
        opMappingRegistry = onnxOpRegistry
)

val transpose = OnnxMappingProcess(
        opName = "transpose",
        inputFrameworkOpName = "Transpose",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("permuteDims" to "perm"))),
        opMappingRegistry = onnxOpRegistry
)

//TODO: TreeEnsembleClassifier
//TODO: TreeEnsembleRegressor
//TODO: Unique PRIORITIZE
//TODO: Unsqueeze PRIORITIZE
//TODO: Upsample PRIORITIZE
//TODO: Where PRIORITIZE
//TODO: ZipMap
fun defOnnxSingleTransform(opName: String, inputFrameworkOpName: String, outputName: String, inputFrameworkInput: String = "input", attributeMappingRules: List<AttributeMappingRule<Onnx.NodeProto,Onnx.NodeProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto,Onnx.TensorProto.DataType>> = emptyList()): OnnxMappingProcess {
    return OnnxMappingProcess(
            opName = opName,
            tensorMappingRules = listOf(
                    NDArrayMappingRule(mappingNamesToPerform = mutableMapOf(outputName to inputFrameworkInput))),
            inputFrameworkOpName = inputFrameworkOpName,
            inputFramework = "onnx",
            attributeMappingRules =   attributeMappingRules,
            opMappingRegistry = onnxOpRegistry)
}

fun defineOnnxPairwiseTransforms(opName: String, inputFrameworkOpName: String,
                                 firstOutputName: String = "input",
                                 secondOutputName: String = "y",
                                 firstInput: String = "A", secondInput: String = "B") : OnnxMappingProcess {
    return OnnxMappingProcess(
            opName = opName,
            tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf(
                    firstOutputName to firstInput,
                    secondOutputName to secondInput))),
            inputFrameworkOpName = inputFrameworkOpName,
            inputFramework = "onnx",
            opMappingRegistry = onnxOpRegistry)
}

fun defineOnnxSingleTransform(inputOpName: String, inputFrameworkOpName: String): OnnxMappingProcess {
    return  OnnxMappingProcess(
            opName = inputOpName,
            inputFrameworkOpName = inputFrameworkOpName, tensorMappingRules =  listOf(NDArrayMappingRule(
            mappingNamesToPerform = mutableMapOf("input" to "input"))),
            opMappingRegistry = onnxOpRegistry)

}



val abs = OnnxMappingProcess(
        opName = "abs", tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("x" to "x"))),
        inputFrameworkOpName = "Abs",
        inputFramework = "onnx",
        opMappingRegistry = onnxOpRegistry)

val const = OnnxMappingProcess(
        opName = "identity",
        inputFrameworkOpName = "Constant",
        opMappingRegistry = onnxOpRegistry
)

val ceil = defOnnxSingleTransform(inputFrameworkOpName = "Ceil",opName = "ceil",inputFrameworkInput = "X",outputName = "x")



val conv2d = OnnxMappingProcess(
        inputFramework = "onnx",
        inputFrameworkOpName = "Conv",
        opName = "conv2d",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","filter" to "weights"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "auto_pad",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "pH",inputFrameworkAttributeName = "padding",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "pW",inputFrameworkAttributeName = "padding",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                sizeThreshold(outputAttribute = "dH",index = 0,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "dilations"),
                sizeThreshold(outputAttribute = "dW",index = 1,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "dilations"),
                sizeThreshold(outputAttribute = "sH",index = 0,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "strides"),
                sizeThreshold(outputAttribute = "sW",index = 0,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "strides"),
                sizeThreshold(outputAttribute = "kH",index = 0,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "kernel_shape"),
                sizeThreshold(outputAttribute = "kW",index = 1,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "kernel_shape")
        ),opMappingRegistry = onnxOpRegistry)

val elu = defOnnxSingleTransform(opName = "elu",inputFrameworkOpName = "Elu",outputName = "input",inputFrameworkInput = "X",
        attributeMappingRules = listOf(valueMappings(mutableMapOf("alpha" to "alpha"))))

val mean = defOnnxSingleTransform(opName = "reduce_mean",inputFrameworkOpName = "Mean",outputName = "input",inputFrameworkInput = "data_0")
val min = defOnnxSingleTransform(opName = "reduce_min",inputFrameworkOpName = "Min",outputName = "input",inputFrameworkInput = "data_0")
val selu = defOnnxSingleTransform(inputFrameworkOpName = "Selu",opName = "selu",inputFrameworkInput = "input",outputName = "x",
        attributeMappingRules = listOf(valueMappings(mutableMapOf("alpha" to "alpha","gamma" to "gamma"))))
val sum = defOnnxSingleTransform(opName = "reduce_sum",inputFrameworkOpName = "Sum",outputName = "input",inputFrameworkInput = "data_0")

object OnnxOpDeclarations {
    init {
        OpRegistryHolder.registerOpMappingRegistry("onnx", onnxOpRegistry)
        names.forEach {
            defineOnnxSingleTransform(inputFrameworkOpName = it.key,inputOpName = it.value)
        } ?: "Error initializing single defined transforms in onnx."

        pairWiseNames.forEach {
            defineOnnxPairwiseTransforms(opName = it.value,inputFrameworkOpName = it.key)
        } ?: "Error initializing pair wise transforms"

        onnxops.forEach {
            onnxOpRegistry.registerInputFrameworkOpDef(it.name,it)
        }

        nd4jOpDescriptors.opListList.forEach {
            onnxOpRegistry.registerNd4jOpDef(it.name,it)
        }
    }
}



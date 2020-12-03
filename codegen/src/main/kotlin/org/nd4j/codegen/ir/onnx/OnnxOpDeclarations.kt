package org.nd4j.codegen.ir.onnx

import onnx.Onnx
import org.nd4j.codegen.ir.ArgDescriptor
import org.nd4j.codegen.ir.AttributeMappingRule
import org.nd4j.codegen.ir.nd4jOpDescriptors
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.codegen.ir.tensorflow.convertNumberListToInputNDArray
import org.nd4j.codegen.ir.tensorflow.listAttributeValueLookupToIndex

val onnxOpRegistry = OpMappingRegistry<Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto>("onnx")
val names = mapOf(
        "Acos" to "acos",
        "Acosh" to "acosh",
        "Asin" to "asin",
        "Asinh" to "asinh",
        "Atan" to "atan",
        "Atanh" to "atanh",
        "Cos" to "cos",
        "Cosh" to "cosh",
        "Erf" to "erf",
        "Exp" to "exp",
        "Identity" to "identity",
        "Log" to "log",
        "Sign" to "sign",
        "Sin" to "sin",
        "Sinh" to "sinh",
        "Softsign" to "softsign",
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
        "Sub" to "subtract"
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
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = listOf(
                argDescriptorConstant(argDescriptorConstants = listOf(ArgDescriptor {
                    name = "isNCHW"
                    boolValue = true
                })),
                intConstant(inputName = "dH",constantValue = 0 as Integer)[0],
                intConstant(inputName = "dW",constantValue = 0 as Integer)[0],
                intConstant(inputName = "extraParam0",constantValue = 0 as Integer)[0],
                stringContainsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "auto_pad",valueToTest = "SAME"),
                listAttributeValueLookup(outputAttributeValue = "pH",inputAttributeValue = "pads",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "pW",inputAttributeValue = "pads",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "sH",inputAttributeValue = "strides",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "sW",inputAttributeValue = "strides",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "kW",inputAttributeValue = "kernel_shape",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "kH",inputAttributeValue = "kernel_shape",indexValue = 0)))

val batchNorm = OnnxMappingProcess(
        opName = "batchnorm",
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "BatchNormalization",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X","mean" to "mean","variance" to "var","gamma" to "scale",
                //TODO: VERIFY THIS! note this mapping is erroneous, onnx does not have a beta/offset
                "beta" to "X"))),
        attributeMappingRules = listOf(valueMappings(mapOf("epsilon" to "epsilon")),
                booleanConstant(inputName = "inPlace",constantValue = false)[0],
                booleanConstant(inputName = "applyGamma",constantValue = true)[0],
                booleanConstant(inputName = "applyBeta",constantValue = true)[0],
                intConstant(inputName = "applyScale",constantValue = 1 as Integer)[0],
                intConstant(inputName = "applyOffset",constantValue = 1 as Integer)[0],
                intConstant(inputName = "applyBeta",constantValue = 1 as Integer)[0]
        ))
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
        attributeMappingRules = listOf(valueMappings(mapOf("concatDimension" to "axis")),
                booleanConstant(inputName = "isDynamicAxis",constantValue = false)[0])

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
        attributeMappingRules = listOf(valueMappings(mapOf("exclusive" to "exclusive","reverse" to "reverse")), ndarrayToIntList(ndarrayNameToAttributeName = mutableMapOf("dimensions" to "axis")))
)

val depthToSpace = OnnxMappingProcess(
        opName = "depth_to_space",
        inputFrameworkOpName = "DepthToSpace",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        //note onnx is NCHW by default
        attributeMappingRules = listOf(valueMappings(mapOf("block_size" to "blocksize")), booleanConstant(inputName = "isNHWC",constantValue = false)[0]),
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
        attributeMappingRules = listOf(convertNDArrayInputToScalarAttr(outputAttributeValue = "p" ,inputAttributeValue = "ratio")),
        opMappingRegistry = onnxOpRegistry
)


val floor = OnnxMappingProcess(
        opName = "floor",
        inputFrameworkOpName = "Floor",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false),
        opMappingRegistry = onnxOpRegistry
)

val round = OnnxMappingProcess(
        opName = "round",
        inputFrameworkOpName = "Round",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false),
        opMappingRegistry = onnxOpRegistry
)

val mod = OnnxMappingProcess(
        opName = "mod",
        inputFrameworkOpName = "Mod",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "A","y" to "B"))),
        opMappingRegistry = onnxOpRegistry
)


val sigmoid = OnnxMappingProcess(
        opName = "sigmoid",
        inputFrameworkOpName = "Sigmoid",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        opMappingRegistry = onnxOpRegistry
)


val logSoftmax = OnnxMappingProcess(
        opName = "log_softmax",
        inputFrameworkOpName = "LogSoftmax",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMappings(mutableMapOf("dimension" to "axis")), booleanConstant(inputName = "inPlace",constantValue = false)[0]),
        opMappingRegistry = onnxOpRegistry
)
val softmax = OnnxMappingProcess(
        opName = "softmax",
        inputFrameworkOpName = "Softmax",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMappings(mutableMapOf("dimension" to "axis")), booleanConstant(inputName = "inPlace",constantValue = false)[0]),
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
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "X",
                "Wru" to "R",
                "Wc" to "W",
                "bc" to "B",
                "hLast" to "initial_h",
                //TODO: erroneous mappings
                "bru" to "B"))),
        attributeMappingRules = listOf()
)

val gather = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Gather",
        opName = "gather",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("indices" to "indices","input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("dimensions" to "axis")),
                booleanConstant(inputName = "inPlace",constantValue = false)[0])
)
//TODO: GatherElements
val gatherNd = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "GatherND",
        opName = "gather_nd",
        attributeMappingRules = booleanConstant(inputName = "checkIndices",constantValue = true),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("indices" to "indices","input" to "data")))
)


val gemm = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Gemm",
        opName = "mmul",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "A","y" to "B"))),
        attributeMappingRules = listOf(valueMappings(mapOf("alpha" to "alpha","beta" to "beta",
                "transposeX" to "transA","transposeY" to "transB")),
                booleanConstant(inputName = "transposeZ",constantValue = false)[0])
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
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X")))
)


//        "IsInf" to "isinf",

//TODO: map is-negative,is-positive
val isInf = OnnxMappingProcess(
        opName = "isinf",
        inputFrameworkOpName = "IsInf",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = booleanConstant(inputName = "inPlace", constantValue = false),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X")))
)

val or = OnnxMappingProcess(
        opName = "or",
        inputFrameworkOpName = "Or",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace", constantValue = false)[0],
                doubleConstant(inputName = "comparable", constantValue = 0.0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "A","y" to "B"))))
)

val xor = OnnxMappingProcess(
        opName = "xor",
        inputFrameworkOpName = "Xor",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace", constantValue = false)[0],
                doubleConstant(inputName = "comparable", constantValue = 0.0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "A","y" to "B"))))
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
        attributeMappingRules = listOf(valueMappings(mapOf("alpha" to "alpha","beta" to "beta","bias" to "bias","depth" to "size")),
                booleanConstant(inputName = "inPlace",constantValue = false)[0])

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
                "input" to "X",
                "Wx" to "W",
                "Wr" to "R",
                "Wp" to "P",
                "b" to "B",
                "seqLen" to "sequence_lens",
                "hI" to "initial_h",
                "cI" to "initial_c"))),
        attributeMappingRules =  listOf(valueMappings(mapOf("cellClip" to "clip")),
                stringToIndex(outputAttributeValue = "directionMode",
                        inputAttributeValue = "direction",
                        listOfValues = listOf("forward","reverse","bidirectional")),
                intConstant(inputName = "dataFormat",constantValue = 0 as Integer)[0],
                booleanConstant(inputName = "hasBiases",constantValue = true)[0],
                booleanConstant(inputName = "hasSeqLen",constantValue = true)[0],
                booleanConstant(inputName = "hasInitH",constantValue = true)[0],
                booleanConstant(inputName = "hasInitC",constantValue = true)[0],
                booleanConstant(inputName = "hasPH",constantValue = true)[0],
                booleanConstant(inputName = "retFullSeq",constantValue = true)[0],
                booleanConstant(inputName = "retLastH",constantValue = true)[0],
                booleanConstant(inputName = "retLastC",constantValue = true)[0],
                listAttributeValueLookup(outputAttributeValue = "gateAlpha",inputAttributeValue = "activation_alpha",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "cellAlpha",inputAttributeValue = "activation_alpha",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "outAlpha",inputAttributeValue = "activation_alpha",indexValue = 2),
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
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = listOf(valueMappings(mapOf("alpha" to "alpha")),
                booleanConstant(inputName = "inPlace",constantValue = false)[0]),
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
        attributeMappingRules = listOf(booleanConstant(inputName = "transposeX",constantValue = false)[0],
                booleanConstant(inputName = "transposeY",constantValue = false)[0],
                booleanConstant(inputName = "transposeZ",constantValue = false)[0],
                doubleConstant(inputName = "alpha",constantValue = 0.0)[0],
                doubleConstant(inputName = "beta",constantValue = 1.0)[0]),
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
                listAttributeValueLookup(outputAttributeValue = "dH",inputAttributeValue = "dilations",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "dW",inputAttributeValue = "dilations",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "pH",inputAttributeValue = "dilations",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "pW",inputAttributeValue = "dilations",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "sH",inputAttributeValue = "strides",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "sW",inputAttributeValue = "strides",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "kH",inputAttributeValue = "kernel_shape",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "kW",inputAttributeValue = "kernel_shape",indexValue = 1)))


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
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("maxOutputSize" to "max_output_boxes_per_class")),
                convertNDArrayInputToScalarAttr(outputAttributeValue = "overlayThreshold",inputAttributeValue = "iou_threshold")),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "boxes" to "boxes",
                "scales" to "scores",
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
        //TODO: verify default value
        attributeMappingRules = intConstant(inputName = "sharedAxes",constantValue = 0 as Integer),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X","alpha" to "slope"))),
        opMappingRegistry = onnxOpRegistry
)

val pad = OnnxMappingProcess(
        inputFrameworkOpName = "Pad",
        opMappingRegistry = onnxOpRegistry,
        opName = "pad",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data","paddings" to "pads"))),
        attributeMappingRules = listOf(stringToIndex(outputAttributeValue = "mode",inputAttributeValue = "mode",listOfValues = listOf("constant","reflect","edge")),
                doubleConstant(inputName = "padValue",constantValue = 0.0)[0])
)

//TODO: QLinearConv
//TODO: QLinearMatMul
//TODO: QuantizeLinear
//TODO: RNN PRIORITIZE
val randomNormal = OnnxMappingProcess(
        inputFrameworkOpName = "RandomNormal",
        opName = "random_normal",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(listNumberToNDarray(outputAttributeValue = "input",inputAttributeValue = "shape"))
)


//TODO: RandomNormalLike
//TODO: Note that the attributes for random unifrom are wrong and needed to be discovered through other means.
//The combination of a lack of a java class + the c++ calling out to other functions which had the actual parameters
//names prevented resolution of the real parameter names. May have to look in to values that are passed inline in to functions and look up
//parameter names that way.

val randomUniform = OnnxMappingProcess(
        inputFrameworkOpName = "RandomUniform",
        opName = "randomuniform",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(valueMappings(mapOf("min" to "low","max" to "high")),
                listNumberToNDarray(outputAttributeValue = "shape",inputAttributeValue = "shape"))
)

//TODO: RandomUniformLike
val range = OnnxMappingProcess(
        inputFrameworkOpName = "Range",
        opName = "range",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(
        convertNDArrayInputToScalarAttr(outputAttributeValue = "s",inputAttributeValue = "start"),
                convertNDArrayInputToScalarAttr(outputAttributeValue = "to",inputAttributeValue = "limit"),
                convertNDArrayInputToScalarAttr(outputAttributeValue = "d",inputAttributeValue = "delta"))
)

val neg = OnnxMappingProcess(
        opName = "neg",
        inputFrameworkOpName = "Neg",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X")))
)


val norm1 = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceL1",
        opMappingRegistry = onnxOpRegistry,
        opName = "reduce_norm1",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes"))

)

val norm2 = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceL2",
        opMappingRegistry = onnxOpRegistry,
        opName = "reduce_norm2",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes"))
)

//TODO: ReduceLogSum
val reduceLogSumExp = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceLogSumExp",
        opName = "reduce_logsumexp",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDim" to "keepdims","keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceMax = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMax",
        opName = "reduce_max",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceMean = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMean",
        opName = "reduce_mean",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceMin = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMin",
        opName = "reduce_min",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceProd = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceProd",
        opName = "reduce_prod",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)

val reduceSum = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceSum",
        opName = "reduce_sum",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes")),
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
        opName = "scatter_update",
        attributeMappingRules =   listOf(
                valueMappings(mutableMapOf("dimension" to "axis")),
                ndarrayToIntList(ndarrayNameToAttributeName = mutableMapOf("indices" to "indices"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("operand" to "data","updates" to "updates")))
)

/*
val scatterNd = OnnxMappingProcess(
        opName = "scatter_nd_update",
        inputFrameworkOpName = "ScatterNd",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data","indices" to "indices","updates" to "updates"))),
        opMappingRegistry = onnxOpRegistry
)
*/

//TODO: SequenceAt
//TODO: SequenceConstruct
//TODO: SequenceErase
//TODO: SequenceInsert
//TODO: SequenceLength
val shape = OnnxMappingProcess(
        opName = "shape_of",
        inputFrameworkOpName = "Shape",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "data"))))
)
//TODO: Shrink

val not = OnnxMappingProcess(
        opName = "not",
        inputFrameworkOpName = "Not",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = doubleConstant(inputName = "comparable",constantValue = 0.0),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "X"))))
)


val pow = OnnxMappingProcess(
        opName = "pow",
        inputFrameworkOpName = "Pow",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(convertNDArrayInputToScalarAttr(outputAttributeValue = "pow",inputAttributeValue = "Y"),
                booleanConstant(inputName = "inPlace",constantValue = false)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "X"))))
)

val size = OnnxMappingProcess(
        opName = "size",
        inputFrameworkOpName = "Size",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "data"))))
)

//TODO: map axes
//TODO: slice and strided slice work too differently,revisit one
/*val slice = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Slice",
        opName = "strided_slice",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("v_begin" to "starts","v_end" to "ends","v_stride" to "steps",
        //TODO: note these mappings are erroneous, we need better default values here for equivalent functionality in onnx
        "begin_mask" to "begin","end_mask" to "end")))
)*/


//TODO: SoftmaxCrossEntropyLoss
val spaceToDepth = OnnxMappingProcess(
        opName = "space_to_depth",
        inputFrameworkOpName = "SpaceToDepth",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMappings(mapOf("block_size" to "blocksize")),
                argDescriptorConstant(listOf(ArgDescriptor {
                    name = "isNHWC"
                    boolValue = false
                }))),
        opMappingRegistry = onnxOpRegistry
)

//TODO: don't know a good default value for num_splits, look at TF and implementation in libnd4j to figure out best value
val split = OnnxMappingProcess(
        opName = "split",
        inputFrameworkOpName = "Split",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("a" to "input"))),
        attributeMappingRules = listOf(valueMappings(mapOf("dimensions" to "axis")),
                intConstant(inputName = "num_splits",constantValue = 0 as Integer)[0],
                listNumberToNDarray(outputAttributeValue = "b" ,inputAttributeValue = "split"))
)

val sqrt = OnnxMappingProcess(
        opName = "sqrt",
        inputFrameworkOpName = "Sqrt",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "X"))))
)

val softplus = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Softplus",
        opName = "softplus",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X")))
)

//TODO: SplitToSequence
val squeeze = OnnxMappingProcess(
        opName = "squeeze",
        inputFrameworkOpName = "Squeeze",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(convertNumericalListToNDArray(outputAttributeValue = "a" ,inputAttributeValue =  "axes"),
                listNumberToListNumber(outputAttributeValue = "_a",inputAttributeValue = "axes"))
)

//TODO: StringNormalizer
//TODO: TfIdfVectorizer
//TODO: ThresholdedRelu
val tile = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Tile",
        opName = "tile",
        attributeMappingRules = listOf(booleanConstant(inputName = "is_static_reps",constantValue = true)[0],
                intConstant(inputName = "dimensions",constantValue = 0 as Integer)[0]),
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
            attributeMappingRules = listOf(argDescriptorConstant(listOf(ArgDescriptor {
                name = "inPlace"
                boolValue = false
            }))),
            opMappingRegistry = onnxOpRegistry)
}

fun defineOnnxSingleTransform(inputOpName: String, inputFrameworkOpName: String): OnnxMappingProcess {
    return  OnnxMappingProcess(
            opName = inputOpName,
            inputFrameworkOpName = inputFrameworkOpName, tensorMappingRules =  listOf(NDArrayMappingRule(
            mappingNamesToPerform = mutableMapOf("input" to "input"))),
            attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false),
            opMappingRegistry = onnxOpRegistry)

}


fun booleanConstant(inputName: String, constantValue: Boolean): List<OnnxArgDescriptorConstant> {
    return listOf(argDescriptorConstant(listOf(
            ArgDescriptor {
                name = inputName
                boolValue = constantValue
            }
    )))
}

fun doubleConstant(inputName: String, constantValue: Double): List<OnnxArgDescriptorConstant> {
    return listOf(argDescriptorConstant(listOf(
            ArgDescriptor {
                name = inputName
                doubleValue = constantValue
            }
    )))
}

fun intConstant(inputName: String, constantValue: Integer): List<OnnxArgDescriptorConstant> {
    return listOf(argDescriptorConstant(listOf(
            ArgDescriptor {
                name = inputName
                int64Value = constantValue.toLong()
            }
    )))
}


val abs = OnnxMappingProcess(
        opName = "abs", tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("input" to "X"))),
        inputFrameworkOpName = "Abs",
        inputFramework = "onnx",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false),
        opMappingRegistry = onnxOpRegistry)

val const = OnnxMappingProcess(
        opName = "identity",
        inputFrameworkOpName = "Constant",
        opMappingRegistry = onnxOpRegistry
)

val ceil = defOnnxSingleTransform(inputFrameworkOpName = "Ceil",opName = "ceil",inputFrameworkInput = "X",outputName = "input",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false))



val conv2d = OnnxMappingProcess(
        inputFramework = "onnx",
        inputFrameworkOpName = "Conv",
        opName = "conv2d",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "X","weights" to "W","bias" to "B"))),
        attributeMappingRules = listOf(
                //stringEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW"),
                intConstant(inputName = "isNCHW",constantValue = 1 as Integer)[0],
                intConstant(inputName = "wFormat",constantValue = 1 as Integer)[0],
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "auto_pad",valueToTest = "SAME"),
                listAttributeValueLookup(outputAttributeValue = "dH",inputAttributeValue = "dilations",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "dW",inputAttributeValue = "dilations",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "pH",inputAttributeValue = "pads",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "pW",inputAttributeValue = "pads",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "sH",inputAttributeValue = "strides",indexValue = 0),
                listAttributeValueLookup(outputAttributeValue = "sW",inputAttributeValue = "strides",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "kW",inputAttributeValue = "kernel_shape",indexValue = 1),
                listAttributeValueLookup(outputAttributeValue = "kH",inputAttributeValue = "kernel_shape",indexValue = 0)
        ),opMappingRegistry = onnxOpRegistry)

val elu = defOnnxSingleTransform(opName = "elu",inputFrameworkOpName = "Elu",outputName = "input",inputFrameworkInput = "X",
        attributeMappingRules = listOf(valueMappings(mutableMapOf("alpha" to "alpha"))))

val mean = defOnnxSingleTransform(opName = "reduce_mean",inputFrameworkOpName = "ReduceMean",outputName = "input",inputFrameworkInput = "data",
        attributeMappingRules = listOf(valueMappings(mutableMapOf("keepDims" to "keepdims","dimensions" to "axes")))
)
val min = defOnnxSingleTransform(opName = "reduce_min",inputFrameworkOpName = "ReduceMin",outputName = "input",inputFrameworkInput = "data",
        attributeMappingRules = listOf(valueMappings(mutableMapOf("keepDims" to "keepdims","dimensions" to "axes")))
)
val relu = defOnnxSingleTransform(inputFrameworkOpName = "Relu",opName = "relu",inputFrameworkInput = "X",outputName = "input",
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace",constantValue = false)[0], doubleConstant(inputName = "cutoff",constantValue = 0.0)[0]))

val isNan = defOnnxSingleTransform(inputFrameworkOpName = "IsNaN",opName = "isnan",inputFrameworkInput = "X",outputName = "input",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false))


val selu = defOnnxSingleTransform(inputFrameworkOpName = "Selu",opName = "selu",inputFrameworkInput = "X",outputName = "input",attributeMappingRules =
booleanConstant(inputName = "inPlace",constantValue = false))
val sum = defOnnxSingleTransform(opName = "reduce_sum",inputFrameworkOpName = "ReduceSum",outputName = "input",inputFrameworkInput = "data",
        attributeMappingRules = listOf(valueMappings(mutableMapOf("keepDims" to "keepdims","dimensions" to "axes")))
)

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



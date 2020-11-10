package org.nd4j.codegen.ir.onnx

import onnx.Onnx
import org.nd4j.codegen.ir.ArgDescriptor
import org.nd4j.codegen.ir.AttributeMappingRule
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder

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
        "IsInf" to "isinf",
        "IsNaN" to "isnan",
        "Log" to "log",
        "LogSoftmax" to "log_softmax",
        "Mod" to "mod",
        "Mul" to "mul",
        "Neg" to "neg",
        "Not" to "not",
        "Relu" to "relu",
        "Reciprocal" to "reciprocal",
        "Round" to "round",
        "Sigmoid" to "sigmoid",
        "Sign" to "sign",
        "Sin" to "sin",
        "Sinh" to "sinh",
        "Softmax" to "softmax",
        "Softplus" to "softplus",
        "Softsign" to "softsign",
        "Sqrt" to "sqrt",
        "Tan" to "tan",
        "Tanh" to "tanh"

)

val pairWiseNames = mapOf(
        "Add" to "add",
        "And" to "boolean_and",
        "Div" to "div",
        "Equal" to "equal",
        "Greater" to "greater",
        "GreaterOrEqual" to "greater_equal",
        "Less" to "less",
        "LessOrEqual" to "less_equal",
        "Or" to "or",
        "Pow" to "pow",
        "Sub" to "sub",
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

//TODO: AveragePool
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
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("inputs" to "inputs"))),
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
//TODO: GRU
val gather = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Gather",
        opName = "gather",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("indices" to "indices","input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("axis" to "axis")))
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
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "A","y" to "B"))),
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

//TODO: LSTM
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
        attributeMappingRules =  listOf(valueMappings(mapOf("cellClip" to "clip")))
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
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "A","y" to "B")))
)
//TODO: MatMulInteger
//TODO: Max
//TODO: MaxPool
//TODO: MaxRoiPool
//TODO: MaxUnpool
//TODO: name: "MeanVarianceNormalization"
//todo: Momentum
//TODO: Multinomial
//TODO: NegativeLogLikelihoodLoss
//TODO: NonMaxSuppression
//TODO: NonZero
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
        attributeMappingRules = listOf(ndarrayStringToIndex(outputAttributeValue = "mode",inputAttributeValue = "mode",listOfValues = listOf("constant","reflect","edge")))
)

//TODO: QLinearConv
//TODO: QLinearMatMul
//TODO: QuantizeLinear
//TODO: RNN
//TODO: RandomNormal
val randomNormal = OnnxMappingProcess(
        inputFrameworkOpName = "RandomNormal",
        opName = "random_normal",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(valueMappings(mapOf("mean" to "mean","stdev" to "scale")),
                convertNumberListToInputNDArray(outputAttributeValue = "shape",inputAttributeValue = "shape"))
)


//TODO: RandomNormalLike
//TODO: RandomUniform
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
        attributeMappingRules = listOf(valueMappings(mapOf("from" to "start","to" to "limit","delta" to "delta")))
)
//TODO: ReduceL1
//TODO: ReduceL2
//TODO: ReduceLogSum
//TODO: ReduceLogSumExp
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
//TODO: Resize
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
//TODO: Shape
//TODO: Shrink
//TODO: Size
//TODO: Slice
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
//TODO: Split
//TODO: SplitToSequence
//TODO: Squeeze
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
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("in" to "X"))),
        attributeMappingRules = listOf(valueMappings(mapOf("sorted" to "sorted")), convertNDArrayInputToScalarAttr(outputAttributeValue = "k",inputAttributeValue = "K")),
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
//TODO: Unique
//TODO: Unsqueeze
//TODO: Upsample
//TODO: Where
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

fun definePairwiseTransforms(opName: String, inputFrameworkOpName: String,
                             firstOutputName: String = "x",
                             secondOutputName: String = "y",
                             firstInput: String = "A",secondInput: String = "B") : OnnxMappingProcess {
    return OnnxMappingProcess(
            opName = opName,
            tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf(
                    firstOutputName to firstInput,
                    secondOutputName to secondInput))),
            inputFrameworkOpName = inputFrameworkOpName,
            inputFramework = "onnx",
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

val elu = defOnnxSingleTransform(opName = "elu",inputFrameworkOpName = "Elu",outputName = "x",inputFrameworkInput = "X",
        attributeMappingRules = listOf(valueMappings(mutableMapOf("alpha" to "alpha"))))

val mean = defOnnxSingleTransform(opName = "mean",inputFrameworkOpName = "Mean",outputName = "x",inputFrameworkInput = "data_0")
val min = defOnnxSingleTransform(opName = "min",inputFrameworkOpName = "Min",outputName = "x",inputFrameworkInput = "data_0")
val selu = defOnnxSingleTransform(inputFrameworkOpName = "Selu",opName = "selu",inputFrameworkInput = "X",outputName = "x",
        attributeMappingRules = listOf(valueMappings(mutableMapOf("alpha" to "alpha","gamma" to "gamma"))))
val sum = defOnnxSingleTransform(opName = "sum",inputFrameworkOpName = "Sum",outputName = "x",inputFrameworkInput = "data_0")

object OnnxOpDeclarations {
    init {
        OpRegistryHolder.registerOpMappingRegistry("onnx", onnxOpRegistry)
    }
}



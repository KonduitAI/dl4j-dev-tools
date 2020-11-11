package org.nd4j.codegen.ir.tensorflow

import org.nd4j.codegen.ir.AttributeMappingRule
import org.nd4j.codegen.ir.onnx.definePairwiseTransforms
import org.nd4j.codegen.ir.onnx.pairWiseNames
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.tensorflow.framework.*

val tensorflowOpRegistry = OpMappingRegistry<NodeDef,OpDef, TensorProto,DataType, OpDef.AttrDef,AttrValue>("tensorflow")
//val listOfRules =  listOf(NDArrayMappingRule("abs",mapOf("x" to "x", "y" to "y"))

val singleTransformArgs = mapOf(
        "Abs" to "abs",
        "Acos" to "acos",
        "Acosh" to "acosh",
        "Asin" to "asin",
        "Asinh" to "asinh",
        "Atan" to "atan",
        "Atan2" to "tf_atan2",
        "Atanh" to "atanh",
        "BatchMatMul" to "mmul",
        "BatchMatMulV2" to "mmul",
        "Ceil" to "ceil",
        "Copy" to "copy",
        "CopyHost" to "identity",
        "Cos" to "cos",
        "Cosh" to "cosh",
        "DeepCopy" to "identity",
        "Div" to "divide",
        "DivNoNan" to "divide_no_nan",
        "Elu" to "elu",
        "Erf" to "erf",
        "Erfc" to "erfc",
        "Exp" to "exp",
        "Expm1" to "expm1",
        "FloorMod" to "fmod",
        "FloorDiv" to "floordiv",
        "Floor" to "floor",
        "Greater" to "greater",
        "GreaterEqual" to "greater_equal",
        "HardSigmoid" to "hard_sigmoid",
        "HardTanh" to "hardtanh",
        "Less" to "less",
        "LessEqual" to "less_equal",
        "LGamma" to "lgamma",
        "Log" to "log",
        "LogicalAnd" to "boolean_and",
        "LogicalNot" to "boolean_not",
        "Log1p" to "log1p",
        "LogMatrixDeterminant" to "log_matrix_determinant",
        "LogSigmoid" to "logsigmoid",
        "Maximum" to "maximum",
        "MatMul" to "mmul",
        "Mod" to "mod",
        "MatrixDeterminant" to "matrix_determinant",
        "BatchMatrixDeterminant" to "matrix_determinant",
        "MatrixDiag" to "matrix_diag",
        "BatchMatrixDiag" to "matrix_diag",
        "MatrixDiagPart" to "matrix_diag_part",
        "BatchMatrixDiagPart" to "matrix_diag_part",
        "MatrixInverse" to "matrix_inverse",
        "BatchMatrixInverse" to "matrix_inverse",
        "Minimum" to "min_pairwise",
        "Min" to "reduce_min",
        "Mish" to "mish",
        "Mul" to "multiply",
        "Neg" to "neg",
        "NotEqual" to "not_equals",
        "RationalTanh" to "rational_tanh",
        "Reciprocal" to "Reciprocal",
        "Inv" to "Reciprocal",
        "Rank" to "rank",
        "RealDiv" to "realdiv",
        "Relu" to "relu",
        "Relu6" to "relu6",
        "Rint" to "rint",
        "RightShift" to "rshift_bits",
        "Round" to "round",
        "Rsqrt" to "rsqrt",
        "RGBToHSV" to "rgb_to_hsv",
        "Selu" to "selu",
        "Select" to "select",
        "Shape" to "shape_of",
        "ShapeN" to "shapes_of",
        "Sigmoid" to "sigmoid",
        "Sign" to "sign",
        "Sin" to "sin",
        "Sinh" to "sinh",
        "Size" to "size",
        "Softplus" to "softplus",
        "Softsign" to "softsign",
        "Softmax" to "softmax",
        "Swish" to "swish",
        "Square" to "square",
        "SquaredDifference" to "squaredsubtract",
        "Sqrt" to "sqrt",
        "Tan" to "tan",
        "Tanh" to "tanh",
        "TruncateDiv" to "truncatediv"
)

val elementWiseTransformOps = mapOf(
        "Add" to "add",
        "AddV2" to "add"
)


val reduceOps = mapOf(
        "AccumulateNV2" to "mergeadd",
        "Mean" to "reduce_mean",
        "Prod" to "reduce_prod",
        "Sum" to "reduce_sum"
)

fun mapSameName(names: List<String>): List<NDArrayMappingRule> {
    return listOf(mappingNDArrayInputs(names.map { name -> Pair(name,name) }.toMap().toMutableMap()))
}

fun mapTensorNamesWithOp(inputFrameworkOpName: String,
                         opName: String,
                         tensorNames: MutableMap<String,String>,
                         attributeMappingRules: List<AttributeMappingRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>> = emptyList()): TensorflowMappingProcess {
    return TensorflowMappingProcess(
            opName = opName,
            inputFrameworkOpName = inputFrameworkOpName,
            opMappingRegistry = tensorflowOpRegistry,
            tensorMappingRules = listOf(mappingNDArrayInputs(tensorNames)),
            attributeMappingRules = attributeMappingRules
    )

}

fun multipleNameMapping(inputFrameworkOpNames: List<String>,
                        opName: String,tensorNames: MutableMap<String, String>,
                        attributeMappingRules: List<AttributeMappingRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>> = emptyList()): List<TensorflowMappingProcess> {
    return inputFrameworkOpNames.map {
        mapTensorNamesWithOp(inputFrameworkOpName = it,opName = opName,tensorNames = tensorNames,attributeMappingRules = attributeMappingRules)
    }
}



/*"""
op {
  name: "AddN"
  input_arg {
    name: "inputs"
    type_attr: "T"
    number_attr: "N"
  }
  output_arg {
    name: "sum"
    type_attr: "T"
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_INT64
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_BFLOAT16
        type: DT_UINT16
        type: DT_COMPLEX128
        type: DT_HALF
        type: DT_UINT32
        type: DT_UINT64
        type: DT_VARIANT
      }
    }
  }
  is_aggregate: true
  is_commutative: true
}"""*/

val addN = TensorflowMappingProcess(
        inputFrameworkOpName = "AddN",
        opName = "mergesum",
        opMappingRegistry = tensorflowOpRegistry
)

val adjustContrastRule = TensorflowMappingProcess(
        inputFrameworkOpName = "AdjustConstrast",
        opName = "adjust_contrast_v2",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mapOf("factor" to "contrast_factor")))
)

val allRule = TensorflowMappingProcess(
        inputFrameworkOpName = "All",
        opName = "all",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("x" to "input", "y" to "reduction_indices"))),
        attributeMappingRules = listOf(valueMapping((mapOf("keepDims" to "keep_dims"))))
)

val anyRule = TensorflowMappingProcess(
        inputFrameworkOpName = "Any",
        opName = "any",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("x" to "input", "y" to "reduction_indices"))),
        attributeMappingRules = listOf(valueMapping((mapOf("keepDims" to "keep_dims"))))
)

val angleRule = TensorflowMappingProcess(
        inputFrameworkOpName = "Angle",
        opName = "zeros_like",
        opMappingRegistry = tensorflowOpRegistry
)

val approxEqualRule = TensorflowMappingProcess(
        inputFrameworkOpName = "ApproximateEqual",
        opName = "equals_with_eps",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "x","y" to "y"))),
        attributeMappingRules = listOf(valueMapping(mapOf("eps" to "tolerance")))
)

val argMaxRule = TensorflowMappingProcess(
        inputFrameworkOpName = "ArgMax",
        opName = "argmax",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = (listOf(valueMapping(mapOf("keep_dims" to "keepDims","dimensions" to "dimension")),
                ndarrayToIntList(mutableMapOf("dimensions" to "dimension"))))

)

val argMinRule = TensorflowMappingProcess(
        inputFrameworkOpName = "ArgMin",
        opName = "argmin",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = (listOf(valueMapping(mapOf("keep_dims" to "keepDims","dimensions" to "dimension")),
                ndarrayToIntList(mutableMapOf("dimensions" to "dimension"))))

)

/**
 * Note need to fix Assign parsing.
 * It struggles with int vararg arrays and also seems to  have the wrong values for ndarrays (completely missing)
 */
val assignOp = TensorflowMappingProcess(
        inputFrameworkOpName = "Assign",
        opName = "assign",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "ref","y" to "value")))
)

//BaseTransformBoolOp

val avgPool = TensorflowMappingProcess(
        inputFrameworkOpName = "AvgPool",
        opName = "avgpool2d",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "value"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH",inputFrameworkAttributeName = "ksize",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW",inputFrameworkAttributeName = "ksize",targetValue = "NCHW",trueIndex = 3,falseIndex = 2)
        )
)

val avgPool3d = TensorflowMappingProcess(
        inputFrameworkOpName = "AvgPool3D",
        opName = "avgpool3dnew",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "value"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCDHW",inputFrameworkAttributeName = "data_format",valueToTest = "NDHWC"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH",inputFrameworkAttributeName = "strides",targetValue = "NDHWC",trueIndex = 2,falseIndex = 4),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW",inputFrameworkAttributeName = "strides",targetValue = "NDHWC",trueIndex = 4,falseIndex = 5),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH",inputFrameworkAttributeName = "ksize",targetValue = "NDHWC",trueIndex = 2,falseIndex = 4),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW",inputFrameworkAttributeName = "ksize",targetValue = "NDHWC",trueIndex = 4,falseIndex = 5)

        )
)

val batchToSpace = TensorflowMappingProcess(
        opName = "batch_to_space",
        inputFrameworkOpName = "BatchToSpace",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mapOf("block_size" to "blockSize"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","crop" to "crops")))
)

val batchToSpaceND = TensorflowMappingProcess(
        opName = "batch_to_space_nd",
        inputFrameworkOpName = "BatchToSpaceND",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mapOf("block_shape" to "blockShape"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","crop" to "crops")))
)

val betaInc = TensorflowMappingProcess(
        opName = "betainc",
        inputFrameworkOpName = "BetaInc",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("a" to "a","b" to "b","x" to "x"))),
        attributeMappingRules = emptyList()
)

fun defineBiasAdd(names :List<String> =  listOf("BiasAdd","BiasAddV1")) {
    names.forEach {
        TensorflowMappingProcess(
                opName = "biasadd",
                inputFrameworkOpName = it,
                opMappingRegistry = tensorflowOpRegistry,
                tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "value","bias" to "bias"))),
                attributeMappingRules = emptyList()

        )
    }
}

val biasAddResult = defineBiasAdd()
//TODO: https://www.tensorflow.org/api_docs/python/tf/math/bincount
//TOD: Clean up parser with no names INPUT_VARIABLE(..)
val binCountTf = """
op {
  name: "Bincount"
  input_arg {
    name: "arr"
    type: DT_INT32
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  input_arg {
    name: "weights"
    type_attr: "T"
  }
  output_arg {
    name: "bins"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
"""
val binCount = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "bincount",
        inputFrameworkOpName = "Bincount",
        tensorMappingRules =  listOf(mappingNDArrayInputs(mutableMapOf("weights" to "weights","max" to "arr"))),
        attributeMappingRules = emptyList()
)

val bitCast = TensorflowMappingProcess(
        opName = "bitcast",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "Bitcast",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("newType" to "type")))
)

val bitwiseAnd = TensorflowMappingProcess(
        opName = "bitwise_and",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BitwiseAnd",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "x","y" to "y"))),
        attributeMappingRules = emptyList()
)

val bitwiseOr = TensorflowMappingProcess(
        opName = "bitwise_or",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BitwiseOr",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "x","y" to "y"))),
        attributeMappingRules = emptyList()
)



val bitwiseXOr = TensorflowMappingProcess(
        opName = "bitwise_xor",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BitwiseXor",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "x","y" to "y"))),
        attributeMappingRules = emptyList()
)

val broadcastDynamicShape = TensorflowMappingProcess(
        opName = "broadcast_dynamic_shape",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BroadcastArgs",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "s0","y" to "s1")))
)

val broadcastCatGradientArgs = TensorflowMappingProcess(
        opName = "broadcastgradientargs",
        inputFrameworkOpName = "BroadcastGradientArgs",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "s0","y" to "s1")))
)

val broadcastTo = TensorflowMappingProcess(
        opName = "broadcast_to",
        inputFrameworkOpName = "BroadcastTo",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","shape" to "shape")))
)

val cholesky = TensorflowMappingProcess(
        opName = "cholesky",
        inputFrameworkOpName = "Cholesky",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = mapSameName(listOf("input"))
)


val clipByValue = TensorflowMappingProcess(
        opName = "ClipByValue",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "ClipByValue",
        tensorMappingRules = mapSameName(listOf("input")),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("clipValueMin" to "clip_value_min","clipValueMax" to "clip_value_max")))
)


val compareAndBitPack = TensorflowMappingProcess(
        opName = "compare_and_bitpack",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "CompareAndBitpack",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "input","y" to "threshold")))
)


/**
 * TODO: Fix auto generated variable name for input.
 * Current value is empty\");
 */
val concat = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "concat",
        inputFrameworkOpName = "Concat",
        tensorMappingRules = emptyList(),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("concatDimension" to "concat_dimension")))
)

val concatv2 = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "concat",
        inputFrameworkOpName = "ConcatV2",
        tensorMappingRules = emptyList(),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("concatDimension" to "axis")))
)


val cropAndResize = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "crop_and_resize",
        inputFrameworkOpName = "CropAndResize",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "image" to "image",
                "boxes" to "boxes",
                "boxIndexes" to "box_ind",
                "newImageSize" to "crop_size"))),
        attributeMappingRules = listOf(
                ndarrayStringToIndex(outputAttributeValue = "method",inputAttributeValue = "method",listOfValues = listOf("bilinear","nearest")),
                valueMapping(mapOf("extrapolationValue" to "extrapolation_value")))
)

val cumProd = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "cumprod",
        inputFrameworkOpName = "CumProd",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x"))),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("exclusive" to "exclusive","reverse" to "reverse")), ndarrayToIntList(mutableMapOf("axis" to "a")))

)


val cumSum= TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "cumsum",
        inputFrameworkOpName = "CumSum",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x"))),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("exclusive" to "exclusive","reverse" to "reverse")), ndarrayToIntList(mutableMapOf("axis" to "a")))

)


val cross = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "cross",
        inputFrameworkOpName = "Cross",
        tensorMappingRules = mapSameName(listOf("a","b"))
)

val depthToSpace = TensorflowMappingProcess(
        opName = "depth_to_space",
        inputFrameworkOpName = "DepthToSpace",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMapping(mapOf("block_size" to "block_size")), stringEqualsRule("isNWHC",inputFrameworkAttributeName = "data_format",valueToTest = "NWHC")),
        opMappingRegistry = tensorflowOpRegistry
)

/**
 * depth_conv
 */
val depthWiseConv2d = TensorflowMappingProcess(
        opName = "depthwise_conv2d",
        inputFrameworkOpName = "DepthwiseConv2dNative",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","filter" to "weights"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH",inputFrameworkAttributeName = "dilations",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW",inputFrameworkAttributeName = "dilations",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                //NOTE: This is a dynamically resolved attribute at runtime.
                sizeAtRule(outputAttributeName = "kH",dimensionIndex = 0,inputFrameworkAttributeName = "filter"),
                sizeAtRule(outputAttributeName = "kW",dimensionIndex = 1,inputFrameworkAttributeName = "filter"))
)


val diag = TensorflowMappingProcess(
        inputFrameworkOpName = "Diag",
        opName = "diag",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("diag" to "diagonal"))),
        opMappingRegistry = tensorflowOpRegistry
)


val diagPart = TensorflowMappingProcess(
        inputFrameworkOpName = "DiagPart",
        opName = "diag_part",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        opMappingRegistry = tensorflowOpRegistry
)



val diGamma = TensorflowMappingProcess(
        inputFrameworkOpName = "Digamma",
        opName = "digamma",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "x"))),
        opMappingRegistry = tensorflowOpRegistry
)


val dilation2D = TensorflowMappingProcess(
        opName = "dilation2d",
        inputFrameworkOpName = "Dilation2D",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","filter" to "weights"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME"),
                listNumberToListNumber(outputAttributeValue = "rates",inputAttributeValue = "rates"),
                listNumberToListNumber(outputAttributeValue = "strides",inputAttributeValue = "strides"),
                //NOTE: This is a dynamically resolved attribute at runtime.
                sizeAtRule(outputAttributeName = "kH",dimensionIndex = 0,inputFrameworkAttributeName = "filter"),
                sizeAtRule(outputAttributeName = "kW",dimensionIndex = 1,inputFrameworkAttributeName = "filter"))
)


fun defineSingleTransform(inputOpName: String,inputFrameworkOpName: String): TensorflowMappingProcess {
    return  TensorflowMappingProcess(
            opName = inputOpName,
            inputFrameworkOpName = inputFrameworkOpName, tensorMappingRules =  listOf(NDArrayMappingRule(
            mappingNamesToPerform = mutableMapOf("x" to "x"))),
            opMappingRegistry = tensorflowOpRegistry)

}




class ConstMappingProcess: TensorflowMappingProcess(
        opName = "identity",
        inputFrameworkOpName = "Const",
        opMappingRegistry = tensorflowOpRegistry
)

val conv2d =  TensorflowMappingProcess(
        inputFramework = "tensorflow",
        inputFrameworkOpName = "Conv2D",
        opName = "conv2d",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","filter" to "weights"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH",inputFrameworkAttributeName = "dilations",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW",inputFrameworkAttributeName = "dilations",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                //NOTE: This is a dynamically resolved attribute at runtime.
                sizeAtRule(outputAttributeName = "kH",dimensionIndex = 0,inputFrameworkAttributeName = "filter"),
                sizeAtRule(outputAttributeName = "kW",dimensionIndex = 1,inputFrameworkAttributeName = "filter")
        ),opMappingRegistry = tensorflowOpRegistry)


val conv3d =  TensorflowMappingProcess(
        inputFrameworkOpName = "Conv3D",
        opName = "conv3dnew",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","filter" to "weights"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCDHW",inputFrameworkAttributeName = "data_format",valueToTest = "NDHWC"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH",inputFrameworkAttributeName = "strides",targetValue = "NDHWC",trueIndex = 2,falseIndex = 4),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW",inputFrameworkAttributeName = "strides",targetValue = "NDHWC",trueIndex = 4,falseIndex = 5),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sD",inputFrameworkAttributeName = "strides",targetValue = "NDHWC",trueIndex = 1,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH",inputFrameworkAttributeName = "ksize",targetValue = "NDHWC",trueIndex = 2,falseIndex = 4),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW",inputFrameworkAttributeName = "ksize",targetValue = "NDHWC",trueIndex = 4,falseIndex = 5),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kD",inputFrameworkAttributeName = "ksize",targetValue = "NDHWC",trueIndex = 1,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW",inputFrameworkAttributeName = "dilations",targetValue = "NDHWC",trueIndex = 2,falseIndex = 4),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH",inputFrameworkAttributeName = "dilations",targetValue = "NDHWC",trueIndex = 4,falseIndex = 5),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dD",inputFrameworkAttributeName = "dilations",targetValue = "NDHWC",trueIndex = 1,falseIndex = 2)


        ),opMappingRegistry = tensorflowOpRegistry)

val copy = TensorflowMappingProcess(
        opName = "copy",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "input"))),
        inputFrameworkOpName = "Copy",
        opMappingRegistry = tensorflowOpRegistry
)

fun defineBoundingBoxes(listOfNames: List<String> = listOf("DrawBoundingBoxes","DrawBoundingBoxesV2")) {
    listOfNames.forEach {
        val drawBoundingBoxes = TensorflowMappingProcess(
                inputFrameworkOpName = it,
                opName = "draw_bounding_boxes",
                tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("images" to "images","boxes" to "boxes","colors" to "colors"))),
                opMappingRegistry = tensorflowOpRegistry
        )
    }
}

val defineBoundingBoxesResult = defineBoundingBoxes()




val dynamicPartition = TensorflowMappingProcess(
        opName = "dynamic_partition",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data","indices" to "partitions"))),
        inputFrameworkOpName = "DynamicPartition",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mapOf("numPartition" to "num_partitions")))
)


/**
 * TODO: check if n attribute has value for tensorflow
 */
val dynamicStitch = TensorflowMappingProcess(
        opName = "dynamic_stitch",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data","indices" to "index"))),
        inputFrameworkOpName = "DynamicStitch",
        opMappingRegistry = tensorflowOpRegistry
)

val empty = TensorflowMappingProcess(
        opName = "create",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "shape"))),
        inputFrameworkOpName = "Empty",
        attributeMappingRules = listOf(valueMapping(mapOf("initialize" to "init"))),
        opMappingRegistry = tensorflowOpRegistry
)


val enter = TensorflowMappingProcess(
        opName = "enter",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "shape"))),
        inputFrameworkOpName = "Enter",
        attributeMappingRules = listOf(valueMapping(mapOf("isConstant" to "is_constant"))),
        opMappingRegistry = tensorflowOpRegistry
)

val equal = TensorflowMappingProcess(
        opName = "equal",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "x","y" to "y"))),
        inputFrameworkOpName = "Equal",
        opMappingRegistry = tensorflowOpRegistry
)

val exit = TensorflowMappingProcess(
        opName = "exit",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "data"))),
        inputFrameworkOpName = "Exit",
        opMappingRegistry = tensorflowOpRegistry
)

val expandDims = TensorflowMappingProcess(
        opName = "expand_dims",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "images"))),
        inputFrameworkOpName = "ExpandDims",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(ndarrayToIntList(ndarrayNameToAttributeName = mutableMapOf("axis" to "dim")))
)

val extractImagesPatches = TensorflowMappingProcess(
        opName = "extract_image_patches",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        inputFrameworkOpName = "ExtractImagePatches",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(ndarrayToIntList(ndarrayNameToAttributeName = mutableMapOf("kSizes" to "ksizes","strides" to "strides","rates" to "rates")),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "data_format",valueToTest = "SAME"))
)




val fusedBatchnorm = TensorflowMappingProcess(
        opName = "fused_batch_norm",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "x","scale" to "scale","offset" to "offset","mean" to "mean","variance" to "variance"))),
        inputFrameworkOpName = "FusedBatchNorm",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mutableMapOf("isTraining" to "is_training")), stringEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW"))
)

fun defineBatchNorm(names: List<String> = listOf("FusedBatchNorm","FusedBatchNormV2","FusedBatchNormV3")) {
    names.forEach {
        val fusedBatchnorm = TensorflowMappingProcess(
                opName = "fused_batch_norm",
                tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "x","scale" to "scale","offset" to "offset","mean" to "mean","variance" to "variance"))),
                inputFrameworkOpName = it,
                opMappingRegistry = tensorflowOpRegistry,
                attributeMappingRules = listOf(valueMapping(mutableMapOf("isTraining" to "is_training")), stringEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW"))
        )
    }
}

val defineBatchNormResult = defineBatchNorm()

val gather = TensorflowMappingProcess(
        opName = "gather",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "params","indices" to "indices"))),
        inputFrameworkOpName = "Gather",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("axis" to "axis")))
)


fun defineGather(names: List<String> = listOf("Gather","GatherV2")) {
    names.forEach {
        val gather = TensorflowMappingProcess(
                opName = "gather",
                tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "params","indices" to "indices"))),
                inputFrameworkOpName = it,
                opMappingRegistry = tensorflowOpRegistry,
                attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("axis" to "axis")))
        )
    }
}

val gatherResult = defineGather()

val histogramFixedWidth = TensorflowMappingProcess(
        opName = "histogram_fixed_width",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "values","range" to "value_range"))),
        inputFrameworkOpName = "HistogramFixedWidth",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("nbins" to "nbins")))
)

val identityN = TensorflowMappingProcess(
        opName = "identity_n",
        inputFrameworkOpName = "IdentityN",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules =  listOf(mappingNDArrayInputs(mutableMapOf("x" to "input")))
)

val ifOp = TensorflowMappingProcess(
        opName = "if",
        inputFrameworkOpName = "If",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","condition" to "cond")))
)

//TODO: java names mapped only, check c++ parsing. No ops seem to be found in descriptor.
val iGamma = mapTensorNamesWithOp(inputFrameworkOpName = "IGamma",opName = "igamma",tensorNames = mutableMapOf("n" to "a","x" to "x"))
val iGammaC = mapTensorNamesWithOp(inputFrameworkOpName = "IGammaC",opName = "igammac",tensorNames = mutableMapOf("n" to "a","x" to "x"))
val inTopKResults = multipleNameMapping(inputFrameworkOpNames = listOf("InTopK","InTopKV2"),opName = "in_top_k",
        tensorNames = mutableMapOf("targets" to "target","predictions" to "predictions"),attributeMappingRules = listOf(valueMapping(mutableMapOf("k" to "k"))))
//TODO: no inputs found for toggle_bits either
val invert = mapTensorNamesWithOp(inputFrameworkOpName = "Invert",opName = "toggle_bits",tensorNames = mutableMapOf("input" to "x"))
val invertPermutation = mapTensorNamesWithOp(inputFrameworkOpName = "InvertPermutation",opName = "invert_permutation",tensorNames = mutableMapOf("input" to "x"))
val isFinite = mapTensorNamesWithOp(inputFrameworkOpName = "IsFinite",opName = "isfinite",tensorNames = mutableMapOf("input" to "x"))
val isInf = mapTensorNamesWithOp(inputFrameworkOpName = "IsInf",opName = "isinf",tensorNames = mutableMapOf("input" to "x"))
val isNan = mapTensorNamesWithOp(inputFrameworkOpName = "IsNan",opName = "isnan",tensorNames = mutableMapOf("input" to "x"))
//TODO: weird parameter values with config.getBias( and other similar names
val lrn = mapTensorNamesWithOp(inputFrameworkOpName = "LRN",opName = "lrn",tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("depth" to "depth_radius","alpha" to "alpha","bias" to "bias","beta" to "beta"))))

//TODO: DID NOT PICK UP ALPHA PROPERTY
val leakyRelu = mapTensorNamesWithOp(inputFrameworkOpName = "LeakyRelu",opName = "leakyrelu",
        attributeMappingRules = listOf(valueMapping(mappings = mutableMapOf("alpha" to "alpha"))),
        tensorNames = mutableMapOf("input" to "x"))
//TODO: no input values found
val leftShift = mapTensorNamesWithOp(inputFrameworkOpName = "LeftShift",opName = "shift_bits",
        tensorNames = mutableMapOf("input" to "x"))

val linspace = mapTensorNamesWithOp(inputFrameworkOpName = "LinSpace",opName = "lin_space",tensorNames = mutableMapOf(),
        attributeMappingRules = listOf(convertNDArrayInputToScalarAttr(mutableMapOf("start" to "start","stop" to "stop","number" to "num"))))

val listDiff = mapTensorNamesWithOp(inputFrameworkOpName = "ListDiff",opName = "listdiff",tensorNames = mutableMapOf("values" to "x","keep" to "y"))
val lu = mapTensorNamesWithOp(inputFrameworkOpName = "Lu",opName = "lu",tensorNames = mutableMapOf("input" to "input"))

val matrixSetDiag = multipleNameMapping(inputFrameworkOpNames = listOf("MatrixSetDiag","BatchMatrixSetDiag"),opName = "matrix_set_diag",tensorNames = mutableMapOf("input" to "input","diagonal" to "diagonal"))
val matrixSolve = mapTensorNamesWithOp(inputFrameworkOpName = "MatrixSolve",opName = "solve",tensorNames = mutableMapOf("a" to "matrix","b" to "rhs"),
        attributeMappingRules = listOf(valueMapping(mapOf("adjoint" to "adjoint"))))
val matrixTriangularSolve = mapTensorNamesWithOp(inputFrameworkOpName = "MatrixTriangularSolve",opName = "triangular_solve",tensorNames = mutableMapOf("a" to "matrix","b" to "rhs"),
        attributeMappingRules = listOf(valueMapping(mapOf("adjoint" to "adjoint","lower" to "lower"))))
val max = mapTensorNamesWithOp(inputFrameworkOpName = "Max" ,opName = "reduce_max",tensorNames = mutableMapOf("input" to "input","axesVector" to "reduction_indices"))

val maxPool = multipleNameMapping(
        inputFrameworkOpNames = listOf("MaxPool","MaxPoolV2"),
        opName = "maxpool2d",
        tensorNames = mutableMapOf("input" to "value"),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH",inputFrameworkAttributeName = "ksize",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW",inputFrameworkAttributeName = "ksize",targetValue = "NCHW",trueIndex = 3,falseIndex = 2)
        )
)




val maxPool3d = TensorflowMappingProcess(
        inputFrameworkOpName = "MaxPool3D",
        opName = "maxpool3dnew",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "value"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCDHW",inputFrameworkAttributeName = "data_format",valueToTest = "NDHWC"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH",inputFrameworkAttributeName = "strides",targetValue = "NDHWC",trueIndex = 2,falseIndex = 4),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW",inputFrameworkAttributeName = "strides",targetValue = "NDHWC",trueIndex = 4,falseIndex = 5),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH",inputFrameworkAttributeName = "ksize",targetValue = "NDHWC",trueIndex = 2,falseIndex = 4),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW",inputFrameworkAttributeName = "ksize",targetValue = "NDHWC",trueIndex = 4,falseIndex = 5)

        )
)

val maxPoolWithArgMax = multipleNameMapping(
        inputFrameworkOpNames = listOf("MaxPoolWithArgmax"),
        opName = "max_pool_with_argmax",
        tensorNames = mutableMapOf("input" to "value"),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNHWC",inputFrameworkAttributeName = "data_format",valueToTest = "NWHC"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH",inputFrameworkAttributeName = "ksize",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW",inputFrameworkAttributeName = "ksize",targetValue = "NCHW",trueIndex = 3,falseIndex = 2)
        )
)

//TODO: Not likely correct. Need to figure out true mapping. Likely an implicit control flow op?
val merge = mapTensorNamesWithOp(inputFrameworkOpName = "Merge",opName = "merge",tensorNames = mutableMapOf("a" to "inputs"))

val mirrorPadding = mapTensorNamesWithOp(inputFrameworkOpName = "MirrorPad",opName = "mirror_pad",
        tensorNames = mutableMapOf("input" to "input","paddings" to "paddings"),attributeMappingRules = listOf(stringNotEqualsRule(outputAttribute = "isSymmetric",inputFrameworkAttributeName = "mode",valueToTest = "REFLECT")))

val nonMaxSuppression = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppression","NonMaxSuppressionV2"),
        opName = "non_max_suppression",
        tensorNames = mutableMapOf("boxes" to "boxes","scores" to "scores"),
        attributeMappingRules = listOf(
                valueMapping(mutableMapOf("iouThreshold" to "iou_threshold")),
                convertNDArrayInputToScalarAttr(mutableMapOf("maxOutputSize" to "max_output_size"))))


val nonMaxSuppressionV3 = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppressionV3","NonMaxSuppressionV4","NonMaxSuppressionV5"),
        opName = "non_max_suppression_v3",
        tensorNames = mutableMapOf("boxes" to "boxes","scores" to "scores","scoreThreshold" to "score_threshold"),
        attributeMappingRules = listOf(
                convertNDArrayInputToScalarAttr(mutableMapOf("maxOutputSize" to "max_output_size","iouThreshold" to "iou_threshold"))))

//TODO: optional argument resolution not working
/**
 *   if (block.getTArguments()->size() > 0)
overlapThreshold = T_ARG(0);
if (block.getTArguments()->size() > 1)
scoreThreshold = T_ARG(1);

was not captured in parser.
 */
val nonMaxSuppressionOverlaps = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppressionWithOverlaps"),
        opName = "non_max_suppression_overlaps",
        tensorNames = mutableMapOf("boxes" to "boxes","scores" to "scores","scoreThreshold" to "score_threshold"),
        attributeMappingRules = listOf(
                convertNDArrayInputToScalarAttr(mutableMapOf("maxOutputSize" to "max_output_size","overlapThreshold" to "overlap_threshold","scoreThreshold" to "score_threshold"))))

val nthElement = mapTensorNamesWithOp(inputFrameworkOpName = "NthElement",opName = "nth_element",tensorNames = mutableMapOf("n" to "n","input" to "input"),
        attributeMappingRules = listOf(booleanToInt(mapOf("reverse" to "reverse"))))

val oneHot = mapTensorNamesWithOp(inputFrameworkOpName = "OneHot",opName = "onehot",tensorNames = mutableMapOf("indices" to "indices"),
        attributeMappingRules = listOf(convertNDArrayInputToScalarAttr(mutableMapOf("depth" to "depth","on" to "on_value","off" to "off_value")),
                valueMapping(mutableMapOf("axis" to "axis"))))


val onesLike = mapTensorNamesWithOp(inputFrameworkOpName = "OnesLike",opName = "ones_as",tensorNames = mutableMapOf("input" to "x"))

val stack = multipleNameMapping(inputFrameworkOpNames = listOf("Pack","Stack"),opName = "stack",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dim" to "axis"))),
        tensorNames = mutableMapOf("values" to "values"))

//TODO: Check assignemnt c++ parsing generating INPUT_VARIABLE(2) as an attribute
val pad = multipleNameMapping(inputFrameworkOpNames = listOf("Pad","PadV2"),opName = "pad",tensorNames = mutableMapOf("input" to "input","paddings" to "paddings"))

val parallelConcat = mapTensorNamesWithOp(inputFrameworkOpName = "ParallelConcat",opName = "ParallelConcat",tensorNames = mutableMapOf("input" to "values"))
//TODO: map placeholder
val randomCrop = mapTensorNamesWithOp(inputFrameworkOpName = "RandomCrop",opName = "random_crop",tensorNames = mutableMapOf("input" to "image","shape" to "size"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed"))))

val randomGamma = mapTensorNamesWithOp(inputFrameworkOpName = "RandomGamma",opName = "random_gamma",tensorNames = mutableMapOf("shape" to "shape","alpha" to "alpha"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed"))))

val randomPoisson = multipleNameMapping(inputFrameworkOpNames = listOf("RandomPoisson","RandomPoissonV2"),opName = "random_poisson",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed"))),
        tensorNames = mutableMapOf("shape" to "shape","rate" to "lambda"))

val randomShuffle = mapTensorNamesWithOp(inputFrameworkOpName = "RandomShuffle",opName = "random_shuffle",tensorNames = mutableMapOf("input" to "value"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seeds" to "seed","seeds" to "seed2"))))

//TODO: Look at extra arguments generated like T_ARG(1));
val randomStandardNormal = multipleNameMapping(inputFrameworkOpNames = listOf("RandomStandardNormal"),opName = "random_normal",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed"))),
        tensorNames = mutableMapOf("x" to "shape"))

//TODO: Look in to numerical only named attributes like 0.0
val randomUniform = multipleNameMapping(inputFrameworkOpNames = listOf("RandomUniform","RandomUniformInt"),opName = "randomuniform",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed","min" to "minval","max" to "maxval"))),
        tensorNames = mutableMapOf("input" to "shape"))


val range = multipleNameMapping(inputFrameworkOpNames = listOf("Range"),opName = "range",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("from" to "start","to" to "limit","step" to "delta"))),
        tensorNames = mutableMapOf("shape" to "shape","rate" to "lambda"))

val reshape = multipleNameMapping(inputFrameworkOpNames = listOf("Reshape"),opName = "reshape",
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("shape" to "shape"))),
        tensorNames = mutableMapOf("x" to "tensor"))

val resizeArea = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeArea"),opName = "resize_area",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners"))),
        tensorNames = mutableMapOf("image" to "images","size" to "size"))

val resizeBiCubic = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeBicubic"),opName = "resize_bicubic",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners"))),
        tensorNames = mutableMapOf("image" to "images","size" to "size"))

val resizeBiLinear = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeBilinear"),opName = "resize_bilinear",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners","halfPixelCenters" to "half_pixel_centers"))),
        tensorNames = mutableMapOf("image" to "images","size" to "size"))

val resizeNearestNeighbor = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeNearestNeighbor"),opName = "resize_nearest_neighbor",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners"))),
        tensorNames = mutableMapOf("image" to "images"))

val reverse = multipleNameMapping(inputFrameworkOpNames = listOf("Reverse","ReverseV2"),opName = "reverse",
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("dims" to "dimensions"))),
        tensorNames = mutableMapOf("input" to "tensor"))

val reverseSequence = multipleNameMapping(inputFrameworkOpNames = listOf("ReverseSequence"),opName = "reverse_sequence",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seqLengths" to "seq_lengths","batchDim" to "batch_dim"))),
        tensorNames = mutableMapOf("input" to "input","seqLengths" to "seq_lengths"))

val roll = multipleNameMapping(inputFrameworkOpNames = listOf("Roll"),opName = "roll",
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("axis" to "axis","shift" to "shift"))),
        tensorNames = mutableMapOf("input" to "input","axesI" to "axis","shiftsI" to "shift"))

//TODO: verify usingLocking property, it's not showing up in descriptors
val scatterAdd = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterAdd"),opName = "scatter_add",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("useLocking" to "use_locking"))),
        tensorNames = mutableMapOf("input" to "input","indices" to "indices","updates" to "updates"))

val scatterDiv = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterDiv"),opName = "scatter_div",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("useLocking" to "use_locking"))),
        tensorNames = mutableMapOf("input" to "input","indices" to "indices","updates" to "updates"))

val scatterMax = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterMax"),opName = "scatter_max",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("useLocking" to "use_locking"))),
        tensorNames = mutableMapOf("input" to "input","indices" to "indices","updates" to "updates"))


val scatterMin = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterMin"),opName = "scatter_min",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("useLocking" to "use_locking"))),
        tensorNames = mutableMapOf("input" to "input","indices" to "indices","updates" to "updates"))

val scatterMul = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterMul"),opName = "scatter_mul",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("useLocking" to "use_locking"))),
        tensorNames = mutableMapOf("input" to "input","indices" to "indices","updates" to "updates"))

val scatterNd = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNd"),opName = "scatter_nd",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("useLocking" to "use_locking"))),
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","shape" to "shape"))

val scatterNdAdd = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNdAdd"),opName = "scatter_nd_add",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("useLocking" to "use_locking"))),
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","input" to "ref"))

val scatterNdSub = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNdSub"),opName = "scatter_nd_sub",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("useLocking" to "use_locking"))),
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","input" to "ref"))

val scatterNdUpdate = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNdUpdate"),opName = "scatter_nd_update",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("useLocking" to "use_locking"))),
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","input" to "ref"))


val scatterSub = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterSub"),opName = "scatter_sub",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("useLocking" to "use_locking"))),
        tensorNames = mutableMapOf("input" to "input","indices" to "indices","updates" to "updates"))


val scatterUpdate = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterUpdate"),opName = "scatter_update",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("useLocking" to "use_locking"))),
        tensorNames = mutableMapOf("input" to "input","indices" to "indices","updates" to "updates"))


val segmentMean = multipleNameMapping(inputFrameworkOpNames = listOf("SegmentMean"),opName = "segment_mean",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"))

val segmentMin = multipleNameMapping(inputFrameworkOpNames = listOf("SegmentMin"),opName = "segment_min",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"))


val segmentMax = multipleNameMapping(inputFrameworkOpNames = listOf("SegmentMax"),opName = "segment_max",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"))


val segmentProd = multipleNameMapping(inputFrameworkOpNames = listOf("SegmentProd"),opName = "segment_prod",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"))

val segmentSum = multipleNameMapping(inputFrameworkOpNames = listOf("SegmentSum"),opName = "segment_sum",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"))

val slice = mapTensorNamesWithOp(inputFrameworkOpName = "Slice",opName = "slice",tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("begin" to "begin","size" to "size"))))


val softmaxCrossEntryLossWithLogits = mapTensorNamesWithOp(inputFrameworkOpName = "SoftmaxCrossEntropyWithLogits",opName = "softmax_cross_entropy_loss_with_logits",
        tensorNames = mutableMapOf("logits" to "features","labels" to "labels"))

val spaceToBatch = TensorflowMappingProcess(
        opName = "space_to_batch",
        inputFrameworkOpName = "SpaceToBatch",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mapOf("block_size" to "blockSize"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","paddings" to "paddings")))
)

val spaceToBatchNd = TensorflowMappingProcess(
        opName = "space_to_batch_nd",
        inputFrameworkOpName = "SpaceToBatchND",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mapOf("block_size" to "blockSize"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","blockShape" to "block_shape","paddings" to "paddings")))
)

val spaceToDepth = TensorflowMappingProcess(
        opName = "space_to_depth",
        inputFrameworkOpName = "SpaceToDepth",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMapping(mapOf("block_size" to "block_size")), stringEqualsRule("isNWHC",inputFrameworkAttributeName = "data_format",valueToTest = "NWHC")),
        opMappingRegistry = tensorflowOpRegistry
)

val split = TensorflowMappingProcess(
        opName = "split",
        inputFrameworkOpName = "Split",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "value"))),
        attributeMappingRules = listOf(valueMapping(mapOf("num_splits" to "num_split","splitDim" to "split_dim"))),
        opMappingRegistry = tensorflowOpRegistry
)


val splitV = TensorflowMappingProcess(
        opName = "split_v",
        inputFrameworkOpName = "SplitV",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "value","sizes" to "size_splits"))),
        attributeMappingRules = listOf(valueMapping(mapOf("num_splits" to "num_split","splitDim" to "split_dim"))),
        opMappingRegistry = tensorflowOpRegistry
)

val squeeze = TensorflowMappingProcess(
        opName = "squeeze",
        inputFrameworkOpName = "Squeeze",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(listNumberToListNumber(outputAttributeValue = "squeezeDims",inputAttributeValue = "squeeze_dims")),
        opMappingRegistry = tensorflowOpRegistry
)

val stridedSlice = TensorflowMappingProcess(
        opName = "stridedslice",
        inputFrameworkOpName = "StridedSlice",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("x" to "input"))),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("beginMask" to "begin","endMask" to "end","strides" to "strides")),
                valueMapping(mutableMapOf("beginMask" to "begin_mask","endMask" to "end_mask","ellipsisMask" to "ellipsis_mask","newAxisMask" to "new_axis_mask","shrinkAxisMask" to "shrink_axis_mask")))
)

val svd = TensorflowMappingProcess(
        opName = "svd",
        inputFrameworkOpName = "Svd",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("computeUv" to "compute_uv","fullMatrices" to "full_matrices")))
)

val switch = TensorflowMappingProcess(
        opName = "switch",
        inputFrameworkOpName = "Switch",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data","predicate" to "predicate")))
)


//TODO: revisit this, not sure why the ops are off
val tensorArrayConcat = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayConcat", "TensorArrayConcatV2", "TensorArrayConcatV3"),
        opName = "tensorarrayconcatv3",
        tensorNames = mutableMapOf("args" to "args"))

//TODO: revisit this, not sure why the ops are off
val tensorArrayGather = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayGather", "TensorArrayGatherV2", "TensorArrayGatherV3"),
        opName = "tensorarraygatherv3",
        tensorNames = mutableMapOf("indices" to "indices"))
//TODO: revisit this, not sure why the ops are off
/*val tensorArrayPack = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayPack", "TensorArrayPackV2", "TensorArrayPackV3"),
        opName = "tensorarraypackv3",
        tensorNames = mutableMapOf("indices" to "indices"))*/
//TODO: revisit this, not sure why the ops are off

val tensorArrayRead = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayRead", "TensorArrayReadV2", "TensorArrayReadV3"),
        opName = "tensorarrayreadv3",
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("index" to "index"))),
        tensorNames = mutableMapOf("vec" to "flow_in"))
//TODO: revisit this, not sure why the ops are off

val tensorArrayScatter = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayScatter", "TensorArrayScatterV2", "TensorArrayScatterV3"),
        opName = "tensorarrayscatterv3",
        tensorNames = mutableMapOf("indices" to "indices","array" to "value"))

//TODO: revisit this, not sure why the ops are off

val tensorArraySize = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArraySize", "TensorArraySizeV2", "TensorArraySizeV3"),
        opName = "tensorarraysizev3",
        tensorNames = mutableMapOf("indices" to "indices","array" to "value"))

//TODO: revisit this, not sure why the ops are off

val tensorArraySplit = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArraySplit", "TensorArraySplitV2", "TensorArraySplitV3"),
        opName = "tensorarraysplitv3",
        tensorNames = mutableMapOf("sizes" to "lengths","array" to "value"))

val tile = mapTensorNamesWithOp(inputFrameworkOpName = "Tile",opName = "tile",attributeMappingRules = listOf(valueMapping(mutableMapOf("axis" to "axis"))),
        tensorNames = mutableMapOf("input" to "input","reps_vector" to "multiples"))

val topk = multipleNameMapping(inputFrameworkOpNames = listOf("TopK","TopKV2"),opName = "top_k",
        tensorNames = mutableMapOf("indices" to "input","values" to "values"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("k" to "k","sorted" to "sorted"))))

val transpose = mapTensorNamesWithOp(
        inputFrameworkOpName = "Tranpose",
        opName = "transpose",
        tensorNames = mutableMapOf("x" to "x"),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("permuteDims" to "perm")))
)

val unique = multipleNameMapping(
        inputFrameworkOpNames = listOf("Unique","UniqueV2"),
        opName = "unique",
        tensorNames = mutableMapOf("x" to "x")
)


val uniqueWithCounts = multipleNameMapping(
        inputFrameworkOpNames = listOf("UniqueWithCounts","UniqueWithCountsV2"),
        opName = "unique_with_counts",
        tensorNames = mutableMapOf("x" to "x")
)

val unpack = multipleNameMapping(inputFrameworkOpNames = listOf("Unpack","Unstack"),
        opName = "unstack",
        tensorNames = mutableMapOf("input" to "value"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dim" to "axis","num" to "num"))))


val unsortedSegmentMax = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentMax",
opName = "unsorted_segment_max",
tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids","numOfClasses" to "num_segments"))

val unsortedSegmentMin = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentMin",
        opName = "unsorted_segment_min",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids","numOfClasses" to "num_segments"))

val unsortedSegmentProd = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentProd",
opName = "unsorted_segment_prod",
tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids","numOfClasses" to "num_segments"))


val unsortedSegmentSum = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentSum",
        opName = "unsorted_segment_sum",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids","numOfClasses" to "num_segments"))

val where = mapTensorNamesWithOp(inputFrameworkOpName = "Where",opName = "Where",
        tensorNames = mutableMapOf("condition" to "input")
)


val whileOp = mapTensorNamesWithOp(inputFrameworkOpName = "While",opName = "while",
        tensorNames = mutableMapOf("condition" to "input")
)

val zerosLike = mapTensorNamesWithOp(inputFrameworkOpName = "ZerosLike",opName = "zeros_like",
tensorNames = mutableMapOf("x" to "input"))

val zeta = mapTensorNamesWithOp(inputFrameworkOpName = "Zeta",opName = "zeta",
        tensorNames = mutableMapOf("x" to "x","q" to "q"))


object TensorflowOpDeclarations {
    init {
        OpRegistryHolder.registerOpMappingRegistry("tensorflow", tensorflowOpRegistry)
        singleTransformArgs.forEach {
            defineSingleTransform(inputFrameworkOpName = it.key,inputOpName = it.value)
        }

        pairWiseNames.forEach {
            definePairwiseTransforms(opName = it.value,inputFrameworkOpName = it.key)
        }
    }
}



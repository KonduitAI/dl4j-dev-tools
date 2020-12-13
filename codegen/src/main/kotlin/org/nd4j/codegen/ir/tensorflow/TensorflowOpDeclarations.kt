package org.nd4j.codegen.ir.tensorflow

import org.nd4j.codegen.ir.ArgDescriptor
import org.nd4j.codegen.ir.AttributeMappingRule
import org.nd4j.codegen.ir.nameSpaceTensorFromNDarray
import org.nd4j.codegen.ir.nd4jOpDescriptors
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.factory.Nd4j
import org.tensorflow.framework.*

val tensorflowOpRegistry = OpMappingRegistry<GraphDef,NodeDef,OpDef, TensorProto,DataType, OpDef.AttrDef,AttrValue>("tensorflow")

val singleTransformArgs = mapOf(
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
        "Erfc" to "erfc",
        "Exp" to "exp",
        "Expm1" to "expm1",
        "Floor" to "floor",
        "Log" to "log",
        "Log1p" to "log1p",
        "Neg" to "neg",
        "Rint" to "rint",
        "Round" to "round",
        "Rsqrt" to "rsqrt",
        "Sigmoid" to "sigmoid",
        "Sign" to "sign",
        "Sin" to "sin",
        "Sinh" to "sinh",
        "Square" to "square",
        "Sqrt" to "sqrt",
        "Tan" to "tan",
        "Tanh" to "tanh"
)

val elementWiseTransformOps = mapOf(
        "Add" to "add",
        "AddV2" to "add",
        "Div" to "divide",
        "Greater" to "greater",
        "GreaterEqual" to "greater_equal",
        "Less" to "less",
        "LessEqual" to "less_equal",
        "Mul" to "multiply",
        "Maximum" to "maximum",
        "FloorDiv" to "floordiv",
        "Mod" to "mod",
        "FloorMod" to "fmod",
        "SquaredDifference" to "squaredsubtract",
        "NotEqual" to "not_equals",
        "RealDiv" to "realdiv",
        "RightShift" to "rshift_bits",
        "Atan2" to "tf_atan2",
        "TruncateDiv" to "truncatediv"
)


val reduceOps = mapOf(
        "AccumulateNV2" to "mergeadd",
        "Mean" to "reduce_mean",
        "Prod" to "reduce_prod",
        "Sum" to "reduce_sum",
        "Min" to "reduce_min"

)

//TODO: optional argument resolution not working
/**
 *   if (block.getTArguments()->size() > 0)
overlapThreshold = T_ARG(0);
if (block.getTArguments()->size() > 1)
scoreThreshold = T_ARG(1);

was not captured in parser.
 */

fun booleanConstant(inputName: String, constantValue: Boolean,argumentIndex: Int): List<TensorflowArgDescriptorConstant> {
        return listOf(argDescriptorConstant(listOf(
                ArgDescriptor {
                        name = inputName
                        boolValue = constantValue
                        argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                        argIndex = argumentIndex
                }
        )))
}

fun doubleConstant(inputName: String, constantValue: Double,argumentIndex: Int): List<TensorflowArgDescriptorConstant> {
        return listOf(argDescriptorConstant(listOf(
                ArgDescriptor {
                        argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                        name = inputName
                        doubleValue = constantValue
                        argIndex = argumentIndex
                }
        )))
}

fun intConstant(inputName: String, constantValue: Integer,argumentIndex: Int): List<TensorflowArgDescriptorConstant> {
        return listOf(argDescriptorConstant(listOf(
                ArgDescriptor {
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        name = inputName
                        int64Value = constantValue.toLong()
                        argIndex = argumentIndex
                }
        )))
}


fun mapSameName(names: List<String>): List<NDArrayMappingRule> {
        return listOf(mappingNDArrayInputs(names.map { name -> Pair(name,name) }.toMap().toMutableMap()))
}

fun mapTensorNamesWithOp(inputFrameworkOpName: String,
                         opName: String,
                         tensorNames: MutableMap<String,String>,
                         attributeMappingRules: List<AttributeMappingRule<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>> = emptyList()): TensorflowMappingProcess {
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
                        attributeMappingRules: List<AttributeMappingRule<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>> = emptyList()):
        List<TensorflowMappingProcess> {
        return inputFrameworkOpNames.map {
                mapTensorNamesWithOp(inputFrameworkOpName = it,opName = opName,tensorNames = tensorNames,attributeMappingRules = attributeMappingRules)
        }
}


val addN = TensorflowMappingProcess(
        inputFrameworkOpName = "AddN",
        opName = "mergesum",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "inputs"))),
        opMappingRegistry = tensorflowOpRegistry
)


val assert = mapTensorNamesWithOp(inputFrameworkOpName = "Assert",opName = "Assert",tensorNames = mutableMapOf("input" to "condition"))


val allRule = TensorflowMappingProcess(
        inputFrameworkOpName = "All",
        opName = "all",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMapping((mapOf("keepDims" to "keep_dims"))),
                ndarrayToIntList(mutableMapOf("dimensions" to "reduction_indices"))))


val anyRule = TensorflowMappingProcess(
        inputFrameworkOpName = "Any",
        opName = "any",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMapping((mapOf("keepDims" to "keep_dims"))),ndarrayToIntList(mutableMapOf("dimensions" to "reduction_indices")))
)

val angleRule = TensorflowMappingProcess(
        inputFrameworkOpName = "Angle",
        opName = "zeros_like",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        opMappingRegistry = tensorflowOpRegistry
)

val approxEqualRule = TensorflowMappingProcess(
        inputFrameworkOpName = "ApproximateEqual",
        opName = "equals_with_eps",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        attributeMappingRules = listOf(valueMapping(mapOf("eps" to "tolerance")),
                //TODO: note dimensions isn't on the TF op, need to investigate if there is a better default here
                intConstant(inputName = "dimensions",constantValue = 0 as Integer,argumentIndex = 0)[0],
                booleanConstant(inputName = "keepDims",constantValue = false,argumentIndex = 0)[0]))


val argMaxRule = TensorflowMappingProcess(
        inputFrameworkOpName = "ArgMax",
        opName = "argmax",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(
                ndarrayToIntList(mutableMapOf("dimensions" to "dimension")),
                booleanConstant(inputName = "keepDims",constantValue = false,argumentIndex = 0)[0])

)

val argMinRule = TensorflowMappingProcess(
        inputFrameworkOpName = "ArgMin",
        opName = "argmin",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("dimensions" to "dimension")),
                booleanConstant(inputName = "keepDims",constantValue = false,argumentIndex = 0)[0])

)

/**
 * Note need to fix Assign parsing.
 * It struggles with int vararg arrays and also seems to  have the wrong values for ndarrays (completely missing)
 */
val assignOp = TensorflowMappingProcess(
        inputFrameworkOpName = "Assign",
        opName = "assign",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "ref","y" to "value")))
)

//BaseTransformBoolOp

val avgPool = TensorflowMappingProcess(
        inputFrameworkOpName = "AvgPool",
        opName = "avgpool2d",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "value"))),
        attributeMappingRules = listOf(
                stringNotEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 10),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 8),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 0),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 1),
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                name = "pH"
                                int64Value = 0
                                argIndex = 4
                        },
                        ArgDescriptor {
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                name = "pW"
                                int64Value = 0
                                argIndex = 5
                        },
                        ArgDescriptor {
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                name = "dW"
                                int64Value = 0
                                argIndex = 6
                        },
                        ArgDescriptor {
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                name = "dH"
                                int64Value = 0
                                argIndex = 7
                        },
                        ArgDescriptor {
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                name = "extraParam0"
                                int64Value = 0
                                argIndex = 9
                        }
                ))
        )
)

val avgPool3d = TensorflowMappingProcess(
        inputFrameworkOpName = "AvgPool3D",
        opName = "avgpool3dnew",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(
                intConstant(inputName = "extraParam0",constantValue = 0 as Integer,argumentIndex = 13)[0],
                intConstant(inputName = "pD",constantValue = 0 as Integer,argumentIndex = 6)[0],
                intConstant(inputName = "pH",constantValue = 0 as Integer,argumentIndex = 7)[0],
                intConstant(inputName = "pW",constantValue = 0 as Integer,argumentIndex = 8)[0],
                intConstant(inputName = "dD",constantValue = 0 as Integer,argumentIndex = 9)[0],
                intConstant(inputName = "dH",constantValue = 0 as Integer,argumentIndex = 10)[0],
                intConstant(inputName = "dW",constantValue = 0 as Integer,argumentIndex = 11)[0],
                stringEqualsRule(outputAttribute = "isNCDHW",inputFrameworkAttributeName = "data_format",valueToTest = "NDHWC",argumentIndex = 14),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 12),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH", attributeNameOfListAttribute = "ksize", targetValue = "NDHWC", trueIndex = 2,
                        falseIndex = 4,inputFrameworkStringNameToTest = "data_format",argumentIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW", attributeNameOfListAttribute = "ksize", targetValue = "NDHWC", trueIndex = 4,
                        falseIndex = 5,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kD", attributeNameOfListAttribute = "ksize", targetValue = "NDHWC", trueIndex = 1,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 0),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NDHWC", trueIndex = 2,
                        falseIndex = 4,inputFrameworkStringNameToTest = "data_format",argumentIndex = 4),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NDHWC", trueIndex = 4,
                        falseIndex = 5,inputFrameworkStringNameToTest = "data_format",argumentIndex = 5),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sD", attributeNameOfListAttribute = "strides", targetValue = "NDHWC", trueIndex = 1,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3)


        )
)

val batchToSpace = TensorflowMappingProcess(
        opName = "batch_to_space",
        inputFrameworkOpName = "BatchToSpace",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mapOf("blockSize" to "block_size")),
                ndarrayToIntList(mutableMapOf("croppingTop" to "crops","croppingBottom" to "crops")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","crop" to "crops")))
)

val batchToSpaceND = TensorflowMappingProcess(
        opName = "batch_to_space_nd",
        inputFrameworkOpName = "BatchToSpaceND",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("blocks" to "block_shape")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","crop" to "crops","blockShape" to "block_shape")))
)

val betaInc = TensorflowMappingProcess(
        opName = "betainc",
        inputFrameworkOpName = "Betainc",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("a" to "a","b" to "b","input" to "x"))),
        attributeMappingRules = emptyList()
)

fun defineBiasAdd(names :List<String> =  listOf("BiasAdd","BiasAddV1")) {
        names.forEach {
                TensorflowMappingProcess(
                        opName = "biasadd",
                        inputFrameworkOpName = it,
                        opMappingRegistry = tensorflowOpRegistry,
                        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "value","bias" to "bias"))),
                        attributeMappingRules = booleanConstant(inputName = "nchw", constantValue = false,argumentIndex = 0)

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
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("minLength" to "size","maxLength" to "size")),
                intConstant(inputName = "outputType",constantValue = 0 as Integer,argumentIndex = 2)[0])
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
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
)

val bitwiseOr = TensorflowMappingProcess(
        opName = "bitwise_or",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BitwiseOr",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
)



val bitwiseXOr = TensorflowMappingProcess(
        opName = "bitwise_xor",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BitwiseXor",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
)

val broadcastDynamicShape = TensorflowMappingProcess(
        opName = "broadcast_dynamic_shape",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BroadcastArgs",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "s0","y" to "s1")))
)

val broadcastCatGradientArgs = TensorflowMappingProcess(
        opName = "broadcastgradientargs",
        inputFrameworkOpName = "BroadcastGradientArgs",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                intConstant(inputName = "dimension",constantValue = 0 as Integer,argumentIndex = 0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "s0","y" to "s1")))
)

val broadcastTo = TensorflowMappingProcess(
        opName = "broadcast_to",
        inputFrameworkOpName = "BroadcastTo",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","shape" to "shape")))
)


val copy2 = multipleNameMapping(
        inputFrameworkOpNames = listOf("Copy"),
        opName = "copy",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorNames = mutableMapOf("input" to "input")
)

val const = mapTensorNamesWithOp(inputFrameworkOpName = "Const",
        opName = "identity",
        tensorNames = mutableMapOf(),
        attributeMappingRules = listOf(ndArrayAttributeToNDarrayInput(mutableMapOf("input" to "value")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]))


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
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "t"))),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("clipValueMin" to "clip_value_min","clipValueMax" to "clip_value_max")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0])
)


val compareAndBitPack = TensorflowMappingProcess(
        opName = "compare_and_bitpack",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "CompareAndBitpack",
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("threshold" to "threshold"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","y" to "threshold")))
)


val concat = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "concat",
        inputFrameworkOpName = "Concat",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "values"))),
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("concatDimension" to "concat_dim")),
                booleanConstant(inputName = "isDynamicAxis",constantValue = false,argumentIndex = 0)[0])
)

val concatv2 = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "concat",
        inputFrameworkOpName = "ConcatV2",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "values"))),
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("concatDimension" to "axis")),
                booleanConstant(inputName = "isDynamicAxis",constantValue = false,argumentIndex = 0)[0]))


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
                ndarrayStringToIndex(outputAttributeValue = "method",inputAttributeValue = "method",listOfValues = listOf("bilinear","nearest"),argumentIndex = 0),
                valueMapping(mapOf("extrapolationVal" to "extrapolation_value")))
)

val cumProd = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "cumprod",
        inputFrameworkOpName = "Cumprod",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x"))),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("exclusive" to "exclusive","reverse" to "reverse")),
                ndarrayToIntList(mutableMapOf("dimensions" to "axis")))

)


val cumSum= TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "cumsum",
        inputFrameworkOpName = "Cumsum",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x"))),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("exclusive" to "exclusive","reverse" to "reverse")),
                ndarrayToIntList(mutableMapOf("dimensions" to "axis")))

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
        attributeMappingRules = listOf(valueMapping(mapOf("block_size" to "block_size")),
                stringEqualsRule("isNHWC"
                        ,inputFrameworkAttributeName = "data_format",valueToTest = "NWHC",argumentIndex = 1)),
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
                "input" to "input","weights" to "filter"))),
        attributeMappingRules = listOf(
                stringNotEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 9),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 8),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH", attributeNameOfListAttribute = "dilations", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 6),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW", attributeNameOfListAttribute = "dilations", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 7),
                //NOTE: This is a dynamically resolved attribute at runtime.
                sizeAtRule(outputAttributeName = "kH",dimensionIndex = 0,inputFrameworkAttributeName = "filter",argumentIndex = 0),
                sizeAtRule(outputAttributeName = "kW",dimensionIndex = 1,inputFrameworkAttributeName = "filter",argumentIndex = 1),
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                name = "pH"
                                int64Value = 0
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                argIndex = 4
                        },
                        ArgDescriptor {
                                name = "pW"
                                int64Value = 0
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                argIndex = 5
                        },
                        ArgDescriptor {
                                name = "wFormat"
                                int64Value = 0
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                argIndex = 10
                        }
                )))
)


val diag = TensorflowMappingProcess(
        inputFrameworkOpName = "Diag",
        opName = "diag",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "diagonal"))),
        opMappingRegistry = tensorflowOpRegistry
)


val diagPart = TensorflowMappingProcess(
        inputFrameworkOpName = "DiagPart",
        opName = "diag_part",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        opMappingRegistry = tensorflowOpRegistry
)



val diGamma = TensorflowMappingProcess(
        inputFrameworkOpName = "Digamma",
        opName = "digamma",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x"))),
        opMappingRegistry = tensorflowOpRegistry
)


val dilation2D = TensorflowMappingProcess(
        opName = "dilation2d",
        inputFrameworkOpName = "Dilation2D",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","weights" to "filter"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isSameShape",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 0),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 0),
                listNumberToListNumber(outputAttributeValue = "rates",inputAttributeValue = "rates"),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 1)[0],
                listNumberToListNumber(outputAttributeValue = "strides",
                        inputAttributeValue = "strides"))
)


fun defineTensorflowSingleTransform(inputOpName: String, inputFrameworkOpName: String): TensorflowMappingProcess {
        return  TensorflowMappingProcess(
                opName = inputOpName,
                inputFrameworkOpName = inputFrameworkOpName, tensorMappingRules =  listOf(NDArrayMappingRule(
                        mappingNamesToPerform = mutableMapOf("input" to "x"))),
                attributeMappingRules = listOf(argDescriptorConstant(
                        listOf(
                                ArgDescriptor {
                                        name = "inPlace"
                                        boolValue = false
                                        argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                                        argIndex = 0
                                }
                        )
                )),
                opMappingRegistry = tensorflowOpRegistry)

}

fun defineTensorflowPairwiseTransforms(opName: String, inputFrameworkOpName: String,
                                       firstOutputName: String = "input",
                                       secondOutputName: String = "y",
                                       firstInput: String = "x", secondInput: String = "y") : TensorflowMappingProcess {
        return TensorflowMappingProcess(
                opName = opName,
                tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf(
                        firstOutputName to firstInput,
                        secondOutputName to secondInput))),
                inputFrameworkOpName = inputFrameworkOpName,
                inputFramework = "tensorflow",
                attributeMappingRules =  booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
                opMappingRegistry = tensorflowOpRegistry)
}



val conv2d =  TensorflowMappingProcess(
        inputFramework = "tensorflow",
        inputFrameworkOpName = "Conv2D",
        opName = "conv2d",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","weights" to "filter"))),
        attributeMappingRules = listOf(
                intConstant(inputName = "pH",constantValue = 0 as Integer,argumentIndex = 4)[0],
                intConstant(inputName = "pW",constantValue = 0 as Integer,argumentIndex = 5)[0],
                intConstant(inputName = "wFormat",constantValue = 0 as Integer,argumentIndex = 10)[0],
                stringNotEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 9),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 8),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH", attributeNameOfListAttribute = "dilations", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 6),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW", attributeNameOfListAttribute = "dilations", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 7),
                //NOTE: This is a dynamically resolved attribute at runtime.
                sizeAtRule(outputAttributeName = "kH",dimensionIndex = 0,inputFrameworkAttributeName = "filter",argumentIndex = 0),
                sizeAtRule(outputAttributeName = "kW",dimensionIndex = 1,inputFrameworkAttributeName = "filter",argumentIndex = 1)
        ),opMappingRegistry = tensorflowOpRegistry)

/**
 * TODO: verify the amounts
 */
val conv3d =  TensorflowMappingProcess(
        inputFrameworkOpName = "Conv3D",
        opName = "conv3dnew",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","weights" to "filter"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCDHW",inputFrameworkAttributeName = "data_format",valueToTest = "NDHWC",argumentIndex = 13),
                stringEqualsRule(outputAttribute = "paddingMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 12),
                intConstant(inputName = "pD",constantValue = 0 as Integer,argumentIndex = 6)[0],
                intConstant(inputName = "pH",constantValue = 0 as Integer,argumentIndex = 7)[0],
                intConstant(inputName = "pW",constantValue = 0 as Integer,argumentIndex = 8)[0],
                intConstant(inputName = "wFormat",constantValue = 0 as Integer,argumentIndex = 14)[0],
                intConstant(inputName = "kH",constantValue = 0 as Integer,argumentIndex = 1)[0],
                intConstant(inputName = "kW",constantValue = 0 as Integer,argumentIndex = 2)[0],
                intConstant(inputName = "kD",constantValue = 0 as Integer,argumentIndex = 0)[0],
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NDHWC", trueIndex = 2,
                        falseIndex = 4,inputFrameworkStringNameToTest = "data_format",argumentIndex = 4),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NDHWC", trueIndex = 4,
                        falseIndex = 5,inputFrameworkStringNameToTest = "data_format",argumentIndex = 5),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sD", attributeNameOfListAttribute = "strides", targetValue = "NDHWC", trueIndex = 1,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW", attributeNameOfListAttribute = "dilations", targetValue = "NDHWC", trueIndex = 2,
                        falseIndex = 4,inputFrameworkStringNameToTest = "data_format",argumentIndex = 11),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH", attributeNameOfListAttribute = "dilations", targetValue = "NDHWC", trueIndex = 4,
                        falseIndex = 5,inputFrameworkStringNameToTest = "data_format",argumentIndex = 10),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dD", attributeNameOfListAttribute = "dilations", targetValue = "NDHWC", trueIndex = 1,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 9)


        ),opMappingRegistry = tensorflowOpRegistry)



fun defineBoundingBoxes(listOfNames: List<String> = listOf("DrawBoundingBoxes")) {
        listOfNames.forEach {
                val drawBoundingBoxes = TensorflowMappingProcess(
                        inputFrameworkOpName = it,
                        opName = "draw_bounding_boxes",
                        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("images" to "images","boxes" to "boxes"))),
                        opMappingRegistry = tensorflowOpRegistry
                )
        }
}

val defineBoundingBoxesResult = defineBoundingBoxes()


val divideNoNan = TensorflowMappingProcess(
        opName = "divide_no_nan",
        inputFrameworkOpName = "DivNoNan",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        opMappingRegistry = tensorflowOpRegistry
)

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
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("index" to "data","input" to "indices"))),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("numPartitions" to "N"))),
        inputFrameworkOpName = "DynamicStitch",
        opMappingRegistry = tensorflowOpRegistry
)

val empty = TensorflowMappingProcess(
        opName = "create",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("shape" to "shape"))),
        inputFrameworkOpName = "Empty",
        attributeMappingRules = listOf(valueMapping(mapOf("init" to "init")),
                intConstant(inputName = "dataType",constantValue = 0 as Integer,argumentIndex = 0)[0]),
        opMappingRegistry = tensorflowOpRegistry
)


val elu = mapTensorNamesWithOp(inputFrameworkOpName = "Elu",opName = "elu",tensorNames = mutableMapOf("input" to "features"),
        attributeMappingRules = listOf(argDescriptorConstant(
                listOf(
                        ArgDescriptor {
                                name = "alpha"
                                doubleValue = 1.0
                                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                        }
                )
        )))

val enter = TensorflowMappingProcess(
        opName = "enter",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        inputFrameworkOpName = "Enter",
        attributeMappingRules = listOf(valueMapping(mapOf("isConstant" to "is_constant"))),
        opMappingRegistry = tensorflowOpRegistry
)

val equal = TensorflowMappingProcess(
        opName = "equals",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        inputFrameworkOpName = "Equal",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = tensorflowOpRegistry
)

val exit = TensorflowMappingProcess(
        opName = "exit",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        inputFrameworkOpName = "Exit",
        opMappingRegistry = tensorflowOpRegistry
)

val expandDims = TensorflowMappingProcess(
        opName = "expand_dims",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        inputFrameworkOpName = "ExpandDims",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(ndarrayToIntList(ndarrayNameToAttributeName = mutableMapOf("dimensions" to "dim")), argDescriptorConstant(
                listOf(
                        ArgDescriptor {
                                name = "inPlace"
                                boolValue = false
                                argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                                argIndex = 0
                        }
                )
        ))
)

val extractImagesPatches = TensorflowMappingProcess(
        opName = "extract_image_patches",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "images"))),
        inputFrameworkOpName = "ExtractImagePatches",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(
                listAttributeValueLookupToIndex(outputAttributeValue = "ksizeRows",inputAttributeValue = "ksizes",idx =  0,argumentIndex = 0),
                listAttributeValueLookupToIndex(outputAttributeValue = "ksizeCols",inputAttributeValue = "ksizes",idx =  1,argumentIndex = 1),
                listAttributeValueLookupToIndex(outputAttributeValue = "kstrideRows",inputAttributeValue = "strides",idx =  0,argumentIndex = 2),
                listAttributeValueLookupToIndex(outputAttributeValue = "kstrideCols",inputAttributeValue = "strides",idx =  1,argumentIndex = 3),
                listAttributeValueLookupToIndex(outputAttributeValue = "krateRows",inputAttributeValue = "rates",idx =  1,argumentIndex = 4),
                listAttributeValueLookupToIndex(outputAttributeValue = "krateCols",inputAttributeValue = "rates",idx =  1,argumentIndex = 5),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 0))
)




val fusedBatchnorm = TensorflowMappingProcess(
        opName = "fused_batch_norm",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","scale" to "scale",
                "offset" to "offset","mean" to "mean","variance" to "variance"))),
        inputFrameworkOpName = "FusedBatchNorm",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mutableMapOf("epsilon" to "epsilon")),
                booleanToNumber(mutableMapOf("isTraining" to "is_training")),
                stringEqualsRule(outputAttribute = "dataFormat",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 0))
)


val gather = TensorflowMappingProcess(
        opName = "gather",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "params","indices" to "indices"))),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf()),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                intConstant(inputName = "dimensions",constantValue = 0 as Integer,argumentIndex = 0)[0]),
        inputFrameworkOpName = "Gather",
        opMappingRegistry = tensorflowOpRegistry
)


val histogramFixedWidth = TensorflowMappingProcess(
        opName = "histogram_fixed_width",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "values","range" to "value_range","numBins" to "nbins"))),
        inputFrameworkOpName = "HistogramFixedWidth",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("nbins" to "nbins")))
)

val identity = multipleNameMapping(
        opName = "identity",
        inputFrameworkOpNames = listOf("DeepCopy"),
        tensorNames =  mutableMapOf("input" to "x"))


val identityCopyToHost = multipleNameMapping(
        opName = "identity",
        inputFrameworkOpNames = listOf("CopyHost"),
        tensorNames =  mutableMapOf("input" to "input"))

val identityN = TensorflowMappingProcess(
        opName = "identity_n",
        inputFrameworkOpName = "IdentityN",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules =  listOf(mappingNDArrayInputs(mutableMapOf("input" to "input")))
)

val ifOp = TensorflowMappingProcess(
        opName = "Switch",
        inputFrameworkOpName = "If",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","condition" to "cond")))
)



val reciprocal = TensorflowMappingProcess(
        opName = "Reciprocal",
        inputFrameworkOpName = "Inv",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x")))
)

val inTopKResults = multipleNameMapping(inputFrameworkOpNames = listOf("InTopK"),
        opName = "in_top_k",
        tensorNames = mutableMapOf("target" to "targets","predictions" to "predictions"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("k" to "k")),
                booleanConstant(inputName = "sorted",constantValue = true,argumentIndex = 0)[0]))


val inTopKResults2 = multipleNameMapping(inputFrameworkOpNames = listOf("InTopKV2"),
        opName = "in_top_k",
        tensorNames = mutableMapOf("target" to "targets","predictions" to "predictions"),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("k" to "k")),
                booleanConstant(inputName = "sorted",constantValue = true,argumentIndex = 0)[0]))
//TODO: no inputs found for toggle_bits either
val invert = mapTensorNamesWithOp(inputFrameworkOpName = "Invert",opName = "toggle_bits",tensorNames = mutableMapOf("input" to "x"))
val invertPermutation = mapTensorNamesWithOp(inputFrameworkOpName = "InvertPermutation",
        opName = "invert_permutation",tensorNames = mutableMapOf("input" to "x"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))
val isFinite = mapTensorNamesWithOp(inputFrameworkOpName = "IsFinite",opName = "isfinite",tensorNames = mutableMapOf("input" to "x"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))
val isInf = mapTensorNamesWithOp(inputFrameworkOpName = "IsInf",opName = "isinf",
        tensorNames = mutableMapOf("input" to "x"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))
val isNan = mapTensorNamesWithOp(inputFrameworkOpName = "IsNan",opName = "isnan",
        tensorNames = mutableMapOf("input" to "x"),attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))
//TODO: weird parameter values with config.getBias( and other similar names
val lrn = mapTensorNamesWithOp(inputFrameworkOpName = "LRN",opName = "lrn",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("depth" to "depth_radius","alpha" to "alpha",
                "bias" to "bias","beta" to "beta")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]))

val leakyRelu = mapTensorNamesWithOp(inputFrameworkOpName = "LeakyRelu",opName = "leakyrelu",
        attributeMappingRules = listOf(valueMapping(mappings = mutableMapOf("alpha" to "alpha")),
                argDescriptorConstant(listOf(ArgDescriptor {
                        name = "inPlace"
                        boolValue = false
                        argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                }))),
        tensorNames = mutableMapOf("input" to "features"))
//TODO: no input values found
val leftShift = mapTensorNamesWithOp(inputFrameworkOpName = "LeftShift",opName = "shift_bits",
        tensorNames = mutableMapOf("input" to "x","y" to "y"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))

val linspace = mapTensorNamesWithOp(inputFrameworkOpName = "LinSpace",opName = "lin_space",
        tensorNames = mutableMapOf("from" to "start","to" to "stop","length" to "num"),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf(
                        "start" to "start",
                        "stop" to "stop",
                        "number" to "num")),
                intConstant(inputName = "dataType",constantValue = 0 as Integer,argumentIndex = 0)[0]))

val listDiff = mapTensorNamesWithOp(inputFrameworkOpName = "ListDiff",opName = "listdiff",tensorNames = mutableMapOf("values" to "x","keep" to "y"))
val logMatrixDeterinmant = mapTensorNamesWithOp(
        inputFrameworkOpName = "LogMatrixDeterminant",
        opName = "log_matrix_determinant",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))

val logicalAnd = mapTensorNamesWithOp(inputFrameworkOpName = "LogicalAnd",opName = "boolean_and",tensorNames = mutableMapOf("input" to "x","y" to "y"))
val logicalNot = mapTensorNamesWithOp(inputFrameworkOpName = "LogicalNot",opName = "boolean_not",tensorNames = mutableMapOf("input" to "x"))

val lu = mapTensorNamesWithOp(inputFrameworkOpName = "Lu",opName = "lu",tensorNames = mutableMapOf("input" to "input"))
val gemm = multipleNameMapping(inputFrameworkOpNames = listOf("MatMul"),opName = "mmul",
        tensorNames = mutableMapOf("input" to "a","y" to "b"),
        attributeMappingRules =
        listOf(doubleConstant(inputName = "alpha",constantValue = 1.0,argumentIndex = 0)[0],
                doubleConstant(inputName = "beta",constantValue = 0.0,argumentIndex = 1)[0],
                booleanConstant(inputName = "transposeX",constantValue = false,argumentIndex = 0)[0],
                booleanConstant(inputName = "transposeY",constantValue = false,argumentIndex = 1)[0],
                booleanConstant(inputName = "transposeZ",constantValue = false,argumentIndex = 2)[0]))


val matrixSetDiag = multipleNameMapping(inputFrameworkOpNames = listOf("MatrixSetDiag","BatchMatrixSetDiag"),
        opName = "matrix_set_diag",
        tensorNames = mutableMapOf("input" to "input","diagonal" to "diagonal"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))
val matrixSetDiagPart = multipleNameMapping(inputFrameworkOpNames = listOf("MatrixDiagPart"),
        opName = "matrix_diag_part",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorNames = mutableMapOf("input" to "input"))

val matrixSolve = mapTensorNamesWithOp(inputFrameworkOpName = "MatrixSolve",opName = "solve",tensorNames = mutableMapOf("a" to "matrix","b" to "rhs"),
        attributeMappingRules = listOf(valueMapping(mapOf("useAdjoint" to "adjoint"))))
val matrixTriangularSolve = mapTensorNamesWithOp(inputFrameworkOpName = "MatrixTriangularSolve",opName = "triangular_solve",tensorNames =
mutableMapOf("a" to "matrix","b" to "rhs"),
        attributeMappingRules = listOf(valueMapping(mapOf("useAdjoint" to "adjoint","isLower" to "lower"))))


val matrixDeterminant = multipleNameMapping(inputFrameworkOpNames = listOf("BatchMatrixDeterminant","MatrixDeterminant"),opName = "matrix_determinant",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))

val minPairWise = mapTensorNamesWithOp(inputFrameworkOpName = "Minimum",
        opName = "min_pairwise",
        tensorNames = mutableMapOf("input" to "x","y" to "y"))

val max = mapTensorNamesWithOp(inputFrameworkOpName = "Max" ,
        opName = "reduce_max",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("keepDims" to "keep_dims")),
                ndarrayToIntList(mutableMapOf("dimensions" to "reduction_indices"))),
        tensorNames = mutableMapOf("input" to "input","axesVector" to "reduction_indices"))

val maxPool = multipleNameMapping(
        inputFrameworkOpNames = listOf("MaxPool"),
        opName = "maxpool2d",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(
                intConstant(inputName = "pH",constantValue = 0 as Integer,argumentIndex = 4)[0],
                intConstant(inputName = "pW",constantValue = 0 as Integer,argumentIndex = 5)[0],
                intConstant(inputName = "dW",constantValue = 0 as Integer,argumentIndex = 6)[0],
                intConstant(inputName = "dH",constantValue = 0 as Integer,argumentIndex = 7)[0],
                intConstant(inputName = "extraParam0",constantValue = 0 as Integer,argumentIndex = 9)[0],
                stringNotEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 10),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 8),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 0),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 1)
        )
)

val maxPoolV2 = multipleNameMapping(
        inputFrameworkOpNames = listOf("MaxPoolV2"),
        opName = "maxpool2d",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(
                intConstant(inputName = "extraParam0",constantValue = 0 as Integer,argumentIndex = 9)[0],
                intConstant(inputName = "pH",constantValue = 0 as Integer,argumentIndex = 4)[0],
                intConstant(inputName = "pW",constantValue = 0 as Integer,argumentIndex = 5)[0],
                intConstant(inputName = "dW",constantValue = 0 as Integer,argumentIndex = 6)[0],
                intConstant(inputName = "dH",constantValue = 0 as Integer,argumentIndex = 7)[0],
                stringNotEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 10),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 8),
                conditionalFieldValueIntIndexNDArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexNDArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexNDArrayRule(outputAttribute = "kH", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 0),
                conditionalFieldValueIntIndexNDArrayRule(outputAttribute = "kW", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 1)
        )
)


val maxPool3d = TensorflowMappingProcess(
        inputFrameworkOpName = "MaxPool3D",
        opName = "maxpool3dnew",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCDHW",inputFrameworkAttributeName = "data_format",valueToTest = "NDHWC",argumentIndex = 14),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 12),
                intConstant(inputName = "pD",constantValue = 0 as Integer,argumentIndex = 6)[0],
                intConstant(inputName = "pH",constantValue = 0 as Integer,argumentIndex = 7)[0],
                intConstant(inputName = "pW",constantValue = 0 as Integer,argumentIndex = 8)[0],
                intConstant(inputName = "dD",constantValue = 0 as Integer,argumentIndex = 9)[0],
                intConstant(inputName = "dH",constantValue = 0 as Integer,argumentIndex = 10)[0],
                intConstant(inputName = "dW",constantValue = 0 as Integer,argumentIndex = 11)[0],
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NDHWC", trueIndex = 2,
                        falseIndex = 4,inputFrameworkStringNameToTest = "data_format",argumentIndex = 4),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NDHWC", trueIndex = 4,
                        falseIndex = 5,inputFrameworkStringNameToTest = "data_format",argumentIndex = 5),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sD", attributeNameOfListAttribute = "ksize", targetValue = "NDHWC", trueIndex = 1,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH", attributeNameOfListAttribute = "ksize", targetValue = "NDHWC", trueIndex = 2,
                        falseIndex = 4,inputFrameworkStringNameToTest = "data_format",argumentIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW", attributeNameOfListAttribute = "ksize", targetValue = "NDHWC", trueIndex = 4,
                        falseIndex = 5,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kD", attributeNameOfListAttribute = "ksize", targetValue = "NDHWC", trueIndex = 1,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 0)
        )
)
//TODO: potentially need more features to be compatible?
/*
val maxPoolWithArgMax = multipleNameMapping(
        inputFrameworkOpNames = listOf("MaxPoolWithArgmax"),
        opName = "max_pool_with_argmax",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNHWC",inputFrameworkAttributeName = "data_format",valueToTest = "NWHC"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2, falseIndex = 1,inputFrameworkStringNameToTest = "data_format"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3, falseIndex = 2,inputFrameworkStringNameToTest = "data_format"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 2, falseIndex = 1,inputFrameworkStringNameToTest = "data_format"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 3, falseIndex = 2,inputFrameworkStringNameToTest = "data_format")
        )
)*/

//TODO: Not likely correct. Need to figure out true mapping. Likely an implicit control flow op?
val loopCond = mapTensorNamesWithOp(inputFrameworkOpName = "LoopCond",opName = "loop_cond",tensorNames = mutableMapOf())
val merge = mapTensorNamesWithOp(inputFrameworkOpName = "Merge",opName = "merge",tensorNames = mutableMapOf("input" to "inputs","b" to "inputs"))

val mirrorPadding = mapTensorNamesWithOp(inputFrameworkOpName = "MirrorPad",opName = "mirror_pad",
        tensorNames = mutableMapOf("input" to "input","paddings" to "paddings"),
        attributeMappingRules = listOf(stringNotEqualsRule(outputAttribute = "mode",
                inputFrameworkAttributeName = "mode",valueToTest = "REFLECT",argumentIndex = 0),
                booleanConstant(inputName = "isSymmetric",constantValue = true,argumentIndex = 0)[0]))

/**
 * TODO: Need to add a constant mapping or something for NonMaxSuppression
 * v1 and 2 which do not have a scoreThreshold to map. V3 does.
 */

val nonMaxSuppressionV1 = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppression"),
        opName = "non_max_suppression",
        tensorNames = mutableMapOf("boxes" to "boxes","scales" to "scores",
                "maxOutSize" to "max_output_size"),
        attributeMappingRules = listOf(
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                doubleValue = 0.5
                                name = "scoreThreshold"
                                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                argIndex = 1
                        }
                )),
                valueMapping(mutableMapOf("overlayThreshold" to "iou_threshold")),
                attributeScalarToNDArrayInput(outputAttribute = "iouThreshold",inputFrameworkAttributeName = "iou_threshold"),
                convertNDArrayInputToNumericalAttr(mutableMapOf("maxOutputSize" to "max_output_size"))))



val nonMaxSuppressionV2 = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppressionV2"),
        opName = "non_max_suppression",
        tensorNames = mutableMapOf("boxes" to "boxes","scales" to "scores","iouThreshold" to "iou_threshold"),
        attributeMappingRules = listOf(
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                doubleValue = 0.5
                                name = "scoreThreshold"
                                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                argIndex = 1
                        }
                )),
                convertNDArrayInputToNumericalAttr(mutableMapOf("maxOutputSize" to "max_output_size",
                        "overlayThreshold" to "iou_threshold"))))


val nonMaxSuppressionV3 = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppressionV3","NonMaxSuppressionV4"),
        opName = "non_max_suppression_v3",
        tensorNames = mutableMapOf("boxes" to "boxes","scales" to "scores",
                "maxOutSize" to "max_output_size",
                "iouThreshold" to "iou_threshold"),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf(
                        "overlayThreshold" to "iou_threshold",
                        "maxOutputSize" to "max_output_size",
                        "scoreThreshold" to "score_threshold"))))


val matrixInverse = multipleNameMapping(inputFrameworkOpNames = listOf("MatrixInverse","BatchMatrixInverse"),opName = "matrix_inverse",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = true,argumentIndex = 0),
        tensorNames = mutableMapOf("input" to "input"))

val nonMaxSuppressionOverlaps = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppressionWithOverlaps"),
        opName = "non_max_suppression_overlaps",
        tensorNames = mutableMapOf("scales" to "scores"),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("maxOutputSize" to "max_output_size","overlapThreshold" to "overlap_threshold",
                        "scoreThreshold" to "score_threshold"))))

val nthElement = mapTensorNamesWithOp(inputFrameworkOpName = "NthElement",opName = "nth_element",
        tensorNames = mutableMapOf("n" to "n","input" to "input"),
        attributeMappingRules = listOf(valueMapping(mapOf("reverse" to "reverse"))))

val oneHot = mapTensorNamesWithOp(inputFrameworkOpName = "OneHot",opName = "onehot",tensorNames = mutableMapOf("input" to "indices","depth" to "depth"),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("on" to "on_value","off" to "off_value")),

                valueMapping(mutableMapOf("dimensions" to "axis")),
                intConstant(inputName = "dataType",constantValue = 0 as Integer,argumentIndex = 2)[0]))


val or = mapTensorNamesWithOp(inputFrameworkOpName = "LogicalOr",opName = "or",
        tensorNames = mutableMapOf("input" to "x","y" to "y"),
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                doubleConstant(inputName = "comparable",constantValue = 0.0,argumentIndex = 0)[0]))

val onesLike = mapTensorNamesWithOp(inputFrameworkOpName = "OnesLike",
        opName = "ones_as",
        attributeMappingRules = intConstant(inputName = "dataType",constantValue = 0 as Integer,argumentIndex = 0),
        tensorNames = mutableMapOf("input" to "x"))



val pow = mapTensorNamesWithOp(inputFrameworkOpName = "Pow",opName = "pow",
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("pow" to "y")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]),
        tensorNames = mutableMapOf("input" to "x")
)

val rank = mapTensorNamesWithOp(inputFrameworkOpName = "Rank", opName = "rank",tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(argDescriptorConstant(listOf(ArgDescriptor {
                name = "inPlace"
                boolValue = false
                argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                argIndex = 0

        }))))

val relu6 = multipleNameMapping(inputFrameworkOpNames = listOf("Relu6"),opName = "relu6",
        attributeMappingRules = listOf(argDescriptorConstant(
                listOf(ArgDescriptor {
                        name = "inPlace"
                        boolValue = false
                        argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                        argIndex = 0
                },
                        ArgDescriptor {
                                name = "cutoff"
                                doubleValue = 0.0
                                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                argIndex = 0
                        })
        )),
        tensorNames = mutableMapOf("input" to "features"))

val stack = multipleNameMapping(inputFrameworkOpNames = listOf("Pack"),opName = "stack",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dimensions" to "axis"))),
        tensorNames = mutableMapOf("input" to "values"))

//TODO: Check assignemnt c++ parsing generating INPUT_VARIABLE(2) as an attribute
/**
 * // in case of REFLECT and SYMMETRIC modes paddings must obey additional shape requirements
if (INT_ARG(0) == 0) { // CONSTANT mode
if(block.width() > 2) {
REQUIRE_TRUE(input->dataType() == INPUT_VARIABLE(2)->dataType(), 0, "PAD op: data types of input and padValue arrays should be the same but got %i and %i correspondingly !", input->dataType(), INPUT_VARIABLE(2)->dataType());
padValue.assign(INPUT_VARIABLE(2)->e(0));
}
else if (!block.getTArguments()->empty())
padValue = T_ARG(0);
}
else if(INT_ARG(0) == 1) {		// REFLECT mode
for(int dim=0; dim < rank; ++dim)
REQUIRE_TRUE(paddings->e<Nd4jLong>(dim,0) <= (input->shapeOf()[dim]-1) && paddings->e<Nd4jLong>(dim,1) <= (input->shapeOf()[dim]-1), 0, "PAD op: wrong content of paddings array for REFLECT mode !");
}
if(INT_ARG(0) == 2) {		// SYMMETRIC mode
for(int dim=0; dim < rank; ++dim)
REQUIRE_TRUE(paddings->e<Nd4jLong>(dim,0) <= input->shapeOf()[dim] && paddings->e<Nd4jLong>(dim,1)  <= input->shapeOf()[dim], 0, "PAD op: wrong content of paddings array for SYMMETRIC mode !");
}
 */
val pad = multipleNameMapping(inputFrameworkOpNames = listOf("Pad","PadV2"),
        opName = "pad",tensorNames = mutableMapOf("input" to "input","paddings" to "paddings"),attributeMappingRules =
        listOf(argDescriptorConstant(listOf(
                ArgDescriptor {
                        //note: tensorflow only supports constant mode
                        name = "mode"
                        int64Value = 0
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        argIndex = 0
                },
                ArgDescriptor {
                        name = "padValue"
                        doubleValue = 0.0
                        argIndex = 0
                        argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                        argIndex = 0
                }
        ))))

//val parallelConcat = mapTensorNamesWithOp(inputFrameworkOpName = "ParallelConcat",opName = "ParallelConcat",tensorNames = mutableMapOf("input" to "values"))
//TODO: map placeholder
val randomCrop = mapTensorNamesWithOp(inputFrameworkOpName = "RandomCrop",opName = "random_crop",tensorNames = mutableMapOf("input" to "image","shape" to "size"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed"))))

val randomGamma = mapTensorNamesWithOp(inputFrameworkOpName = "RandomGamma",opName = "random_gamma",tensorNames = mutableMapOf("shape" to "shape","alpha" to "alpha"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed"))))


val rgvToHsv = mapTensorNamesWithOp(inputFrameworkOpName = "RGBToHSV",opName = "rgb_to_hsv",tensorNames = mutableMapOf("input" to "images"),
        attributeMappingRules = intConstant(inputName = "dimC",constantValue = 0 as Integer,argumentIndex = 0))

val randomPoisson = multipleNameMapping(inputFrameworkOpNames = listOf("RandomPoisson","RandomPoissonV2"),opName = "random_poisson",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed"))),
        tensorNames = mutableMapOf("shape" to "shape","lambda" to "rate"))

val randomShuffle = mapTensorNamesWithOp(inputFrameworkOpName = "RandomShuffle",opName = "random_shuffle",
        tensorNames = mutableMapOf("input" to "value"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seeds" to "seed"))))

//TODO: Look at extra arguments generated like T_ARG(1));
val randomStandardNormal = multipleNameMapping(inputFrameworkOpNames = listOf("RandomStandardNormal"),opName = "random_normal",
        tensorNames = mutableMapOf("input" to "shape"))

//note: tensorflow hard codes the value at 0 to 1 while we allow customization here
val randomUniformHardCoded = multipleNameMapping(
        inputFrameworkOpNames = listOf("RandomUniform","StatelessRandomUniform"),
        opName = "randomuniform",
        tensorNames = mutableMapOf("shape" to "shape"),
        attributeMappingRules =  listOf(argDescriptorConstant(listOf(
                ArgDescriptor {
                        name = "min"
                        doubleValue = 0.0
                        argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                        argIndex = 0
                },
                ArgDescriptor {
                        name = "max"
                        doubleValue = 1.0
                        argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                        argIndex = 1
                },
                ArgDescriptor {
                        name = "min"
                        argIndex = 1
                        inputValue = nameSpaceTensorFromNDarray(Nd4j.scalar(1.0))
                        argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                },
                ArgDescriptor {
                        name = "max"
                        argIndex = 2
                        inputValue = nameSpaceTensorFromNDarray(Nd4j.scalar(1.0))
                        argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                }
        )))
)

val randomUniformInt = TensorflowMappingProcess(
        inputFrameworkOpName = "RandomUniformInt",
        opName = "randomuniform",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("shape" to "shape","min" to "minval","max" to "maxval"))),
        attributeMappingRules =  listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("min" to "minval","max" to "maxval"))),
        opMappingRegistry = tensorflowOpRegistry
)


val range = multipleNameMapping(inputFrameworkOpNames = listOf("Range"),opName = "range",
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("s" to "start",
                "l" to "limit","d" to "delta"))),
        tensorNames = mutableMapOf("s" to "start","l" to "limit","d" to "delta"))

val relu = mapTensorNamesWithOp(inputFrameworkOpName = "Relu",opName = "relu",tensorNames = mutableMapOf("input" to "features"),
        attributeMappingRules = listOf(doubleConstant(inputName = "cutoff",constantValue = 0.0,argumentIndex = 0)[0],
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]))


val reshape = multipleNameMapping(inputFrameworkOpNames = listOf("Reshape"),opName = "reshape",
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("shape" to "shape"))),
        tensorNames = mutableMapOf("input" to "tensor"))

val resizeArea = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeArea"),opName = "resize_area",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners")),
                ndarrayToIntList(mutableMapOf("height" to "size","width" to "size"))),
        tensorNames = mutableMapOf("image" to "images","size" to "size"))

val resizeBiCubic = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeBicubic"),opName = "resize_bicubic",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners")),
                booleanConstant(inputName = "alignPixelCenters",constantValue = false,argumentIndex = 1)[0]),
        tensorNames = mutableMapOf("image" to "images","size" to "size"))

val resizeBiLinear = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeBilinear"),opName = "resize_bilinear",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners")),
                ndarrayToIntList(mutableMapOf("height" to "size","width" to "size")),
                booleanConstant(inputName = "halfPixelCenter",constantValue = false,argumentIndex = 1)[0]),
        tensorNames = mutableMapOf("image" to "images","newImageSize" to "size"))

val resizeNearestNeighbor = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeNearestNeighbor"),opName = "resize_nearest_neighbor",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners")),
                booleanConstant(inputName = "halfPixelCenter",constantValue = false,argumentIndex = 1)[0],
                ndarrayToIntList(mutableMapOf("height" to "size","width" to "size"))),
        tensorNames = mutableMapOf("image" to "images","newImageSize" to "size"))

val reverse = multipleNameMapping(inputFrameworkOpNames = listOf("ReverseV2"),opName = "reverse",
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("dimensions" to "axis"))),
        tensorNames = mutableMapOf("input" to "tensor"))

val reverseSequence = multipleNameMapping(inputFrameworkOpNames = listOf("ReverseSequence"),opName = "reverse_sequence",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("batchDim" to "batch_dim","seqDim" to "seq_dim"))),
        tensorNames = mutableMapOf("input" to "input","seqLengths" to "seq_lengths"))

val roll = multipleNameMapping(inputFrameworkOpNames = listOf("Roll"),opName = "roll",
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("shift" to "shift"))),
        tensorNames = mutableMapOf("input" to "input","dimensions" to "axis","shiftsI" to "shift"))

//TODO: verify usingLocking property, it's not showing up in descriptors
val scatterAdd = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterAdd"),opName = "scatter_add",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("lock" to "use_locking")),
                booleanConstant(inputName = "checkIndices",constantValue = true,argumentIndex = 1)[0]),
        tensorNames = mutableMapOf("input" to "ref","indices" to "indices","updates" to "updates"))

val scatterDiv = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterDiv"),opName = "scatter_div",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("lock" to "use_locking")),
                booleanConstant(inputName = "checkIndices",constantValue = true,argumentIndex = 1)[0]),
        tensorNames = mutableMapOf("input" to "ref","indices" to "indices","updates" to "updates"))

val scatterMax = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterMax"),opName = "scatter_max",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("lock" to "use_locking")),
                booleanConstant(inputName = "checkIndices",constantValue = true,argumentIndex = 1)[0]),
        tensorNames = mutableMapOf("input" to "ref","indices" to "indices","updates" to "updates"))


val scatterMin = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterMin"),opName = "scatter_min",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("lock" to "use_locking")), argDescriptorConstant(
                listOf(ArgDescriptor {
                        name = "checkIndices"
                        boolValue = true
                        argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                        argIndex = 0
                })
        )),
        tensorNames = mutableMapOf("input" to "ref","indices" to "indices","updates" to "updates"))

val scatterMul = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterMul"),opName = "scatter_mul",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("lock" to "use_locking")),
                booleanConstant(inputName = "checkIndices",constantValue = true,argumentIndex = 1)[0]),
        tensorNames = mutableMapOf("input" to "ref","indices" to "indices","updates" to "updates"))

val scatterNd = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNd"),opName = "scatter_nd",
        attributeMappingRules = listOf(
                booleanConstant(inputName = "checkIndices",constantValue = true,argumentIndex = 1)[0],
                booleanConstant(inputName = "lock",constantValue = true,argumentIndex = 0)[0]),
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","shape" to "shape"))

val scatterNdAdd = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNdAdd"),opName = "scatter_nd_add",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("lock" to "use_locking")),
                booleanConstant(inputName = "checkIndices",constantValue = true,argumentIndex = 1)[0]),
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","input" to "ref"))

val scatterNdSub = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNdSub"),opName = "scatter_nd_sub",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("lock" to "use_locking")),
                argDescriptorConstant(
                        listOf(
                                ArgDescriptor {
                                        name = "checkIndices"
                                        boolValue = true
                                        argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                                }
                        )
                )),
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","input" to "ref"))

val scatterNdUpdate = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNdUpdate"),opName = "scatter_nd_update",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("lock" to "use_locking")),
                argDescriptorConstant(
                        listOf(
                                ArgDescriptor {
                                        name = "checkIndices"
                                        boolValue = true
                                        argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                                        argIndex = 0
                                }
                        )
                )),
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","input" to "ref"))


val scatterSub = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterSub"),opName = "scatter_sub",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("lock" to "use_locking")),
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                name = "checkIndices"
                                boolValue = true
                                argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                                argIndex =0
                        }
                ))),
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","input" to "ref"))

//TODO: note: TF expects indices, we don't support them?
val scatterUpdate = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterUpdate"),opName = "scatter_update",
        attributeMappingRules = listOf(intConstant(inputName = "dimension",constantValue = 0 as Integer,argumentIndex = 1)[0],
                ndarrayToIntList(mutableMapOf("indices" to "indices"))),
        tensorNames = mutableMapOf("operand" to "ref","updates" to "updates"))

val select = mapTensorNamesWithOp(inputFrameworkOpName = "Select",opName = "select",tensorNames = mutableMapOf("cond" to "condition","input" to "t","y" to "e"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))

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

val size = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "Size",
        opName = "size",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input")))
)

val slice = mapTensorNamesWithOp(inputFrameworkOpName = "Slice",opName = "slice",
        tensorNames = mutableMapOf("input" to "input","b" to "begin","e" to "size"),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("size" to "size"))))

val selu = mapTensorNamesWithOp(inputFrameworkOpName = "Selu",opName = "selu",tensorNames = mutableMapOf("input" to "features"),
        attributeMappingRules =
        booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))

val shapeOf = mapTensorNamesWithOp(inputFrameworkOpName = "Shape",opName = "shape_of",tensorNames = mutableMapOf("input" to "input"),attributeMappingRules =
booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))

val softPlus = mapTensorNamesWithOp(inputFrameworkOpName = "Softplus",opName = "softplus",tensorNames = mutableMapOf("input" to "features"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))
val softSign = mapTensorNamesWithOp(inputFrameworkOpName = "Softsign",opName = "softsign",tensorNames = mutableMapOf("input" to "features"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))

val shapeN = mapTensorNamesWithOp(inputFrameworkOpName = "ShapeN",opName = "shapes_of",tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex =  0))

val softMax = mapTensorNamesWithOp(inputFrameworkOpName = "Softmax",opName = "softmax",tensorNames = mutableMapOf("input" to "logits"),attributeMappingRules =
listOf(argDescriptorConstant(
        listOf(
                ArgDescriptor {
                        name = "dimension"
                        int64Value = 0
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        argIndex = 0
                },
                ArgDescriptor {
                        name = "inPlace"
                        boolValue = false
                        argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                        argIndex = 0
                }
        )
)))

val softmaxCrossEntryLossWithLogits = mapTensorNamesWithOp(inputFrameworkOpName = "SoftmaxCrossEntropyWithLogits",opName = "softmax_cross_entropy_loss_with_logits",
        tensorNames = mutableMapOf("logits" to "features","labels" to "labels"),
        attributeMappingRules = intConstant(inputName = "classesDim",constantValue = 0 as Integer,argumentIndex = 0))



val spaceToBatch = TensorflowMappingProcess(
        opName = "space_to_batch",
        inputFrameworkOpName = "SpaceToBatch",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(
                booleanConstant(inputName = "inPlace", constantValue = false,argumentIndex = 0)[0],
                valueMapping(mapOf("blockSize" to "block_size")),
                ndarrayToIntList(mutableMapOf(
                        "paddingTop" to "paddings",
                        "paddingBottom" to "paddings"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","padding" to "paddings")))
)

val spaceToBatchNd = TensorflowMappingProcess(
        opName = "space_to_batch_nd",
        inputFrameworkOpName = "SpaceToBatchND",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(
                ndarrayToIntList(mutableMapOf("blocks" to "block_shape")),
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                name = "inPlace"
                                boolValue = false
                                argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                                argIndex = 0

                        }
                ))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","blockShape" to "block_shape","padding" to "paddings")))
)

val spaceToDepth = TensorflowMappingProcess(
        opName = "space_to_depth",
        inputFrameworkOpName = "SpaceToDepth",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMapping(mapOf("block_size" to "block_size")),
                stringEqualsRule("isNHWC",inputFrameworkAttributeName = "data_format",valueToTest = "NWHC",argumentIndex = 1)),
        opMappingRegistry = tensorflowOpRegistry
)

val split = TensorflowMappingProcess(
        opName = "split",
        inputFrameworkOpName = "Split",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("a" to "split_dim","b" to "value"))),
        attributeMappingRules = listOf(valueMapping(mapOf("num_splits" to "num_split"))
                , ndarrayToIntList(mutableMapOf("dimensions" to "split_dim"))),
        opMappingRegistry = tensorflowOpRegistry
)


val splitV = TensorflowMappingProcess(
        opName = "split_v",
        inputFrameworkOpName = "SplitV",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "value",
                "sizes" to "size_splits",
                "_a"  to "split_dim"))),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("splitDim" to "split_dim")),
                ndarrayToIntList(mutableMapOf("dimensions" to "split_dim"))),
        opMappingRegistry = tensorflowOpRegistry
)

val squeeze = TensorflowMappingProcess(
        opName = "squeeze",
        inputFrameworkOpName = "Squeeze",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(
                listNumberToNDarray(mutableMapOf("a" to "squeeze_dims")),
                listNumberToListNumber(outputAttributeValue = "_a",inputAttributeValue = "squeeze_dims")),
        opMappingRegistry = tensorflowOpRegistry
)

val stridedSlice = TensorflowMappingProcess(
        opName = "strided_slice",
        inputFrameworkOpName = "StridedSlice",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input",
                "v_begin" to "begin",
                "v_end" to "end",
                "v_stride" to "strides"))),
        attributeMappingRules = listOf(
                valueMapping(mutableMapOf("begin_mask" to "begin_mask","end_mask" to "end_mask",
                        "ellipsis_mask" to "ellipsis_mask","new_axis_mask" to "new_axis_mask",
                        "shrink_axis_mask" to "shrink_axis_mask")))
)

/**
 * opList {
name: "svd"
argDescriptor {
name: "u"
argType: INPUT_TENSOR
argIndex: 6
}
argDescriptor {
name: "switchNum);"
argType: OUTPUT_TENSOR
}
argDescriptor {
name: "calcUV"
argType: INT64
argIndex: 1
}
argDescriptor {
name: "input"
argType: INPUT_TENSOR
}
argDescriptor {
name: "input"
argType: INPUT_TENSOR
argIndex: 5
}
argDescriptor {
name: "v"
argType: INPUT_TENSOR
argIndex: 2
}
argDescriptor {
name: "full_matrices"
argType: INT64
argIndex: 5
}
argDescriptor {
name: "switchNum"
argType: INT64
argIndex: 2
}
argDescriptor {
name: "s"
argType: INPUT_TENSOR
argIndex: 4
}
argDescriptor {
name: "fullUV"
argType: INT64
}
}

Fix multiple inputs among other things
 */
/*
val svd = TensorflowMappingProcess(
        opName = "svd",
        inputFrameworkOpName = "Svd",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("computeUv" to "compute_uv","fullMatrices" to "full_matrices")))
)
*/

val switch = TensorflowMappingProcess(
        opName = "Switch",
        inputFrameworkOpName = "Switch",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data","condition" to "pred")))
)


//TODO: revisit this, not sure why the ops are off
val tensorArrayConcat = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayConcat", "TensorArrayConcatV2", "TensorArrayConcatV3"),
        opName = "stack_list",
        tensorNames = mutableMapOf("list" to "flow_in"))

//TODO: revisit this, not sure why the ops are off
val tensorArrayGather = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayGather", "TensorArrayGatherV2", "TensorArrayGatherV3"),
        opName = "gather_list",
        tensorNames = mutableMapOf("indices" to "indices","list" to "flow_in"))
//TODO: revisit this, not sure why the ops are off
/*val tensorArrayPack = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayPack", "TensorArrayPackV2", "TensorArrayPackV3"),
        opName = "tensorarraypackv3",
        tensorNames = mutableMapOf("indices" to "indices"))*/
//TODO: revisit this, not sure why the ops are off

val tensorArrayRead = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayRead", "TensorArrayReadV2", "TensorArrayReadV3"),
        opName = "read_list",
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("index" to "index"))),
        tensorNames = mutableMapOf("list" to "flow_in"))
//TODO: revisit this, not sure why the ops are off

val tensorArrayScatter = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayScatter", "TensorArrayScatterV2", "TensorArrayScatterV3"),
        opName = "scatter_list",
        tensorNames = mutableMapOf("array" to "value","indices" to "indices","list" to "flow_in"))

//TODO: revisit this, not sure why the ops are off

val tensorArraySize = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArraySize", "TensorArraySizeV2", "TensorArraySizeV3"),
        opName = "size_list",
        tensorNames = mutableMapOf("list" to "handle","list" to "flow_in"))

//TODO: revisit this, not sure why the ops are off

val tensorArraySplit = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArraySplit", "TensorArraySplitV2", "TensorArraySplitV3"),
        opName = "split_list",
        tensorNames = mutableMapOf("sizes" to "lengths","array" to "value"))

val tile = mapTensorNamesWithOp(inputFrameworkOpName = "Tile",opName = "tile",
        attributeMappingRules = listOf(intConstant(inputName = "dimensions",constantValue = 0 as Integer,argumentIndex = 0)[0],
                booleanConstant(inputName = "is_static_reps",constantValue = true,argumentIndex = 0)[0]),
        tensorNames = mutableMapOf("input" to "input","reps_vector" to "multiples"))

val topk = multipleNameMapping(inputFrameworkOpNames = listOf("TopK"),opName = "top_k",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("needSort" to "sorted","k" to "k"))))

val topkV2 = multipleNameMapping(inputFrameworkOpNames = listOf("TopKV2"),opName = "top_k",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("needSort" to "sorted")),
                convertNDArrayInputToNumericalAttr(mutableMapOf("k" to "k"))))

val transpose = mapTensorNamesWithOp(
        inputFrameworkOpName = "Transpose",
        opName = "transpose",
        tensorNames = mutableMapOf("input" to "x","permutationVector" to "perm"),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("permuteDims" to "perm")))
)

//note we don't allow unique with an axis argument
val unique = multipleNameMapping(
        inputFrameworkOpNames = listOf("Unique","UniqueV2"),
        opName = "unique",
        tensorNames = mutableMapOf("input" to "x")
)





val uniqueWithCounts = multipleNameMapping(
        inputFrameworkOpNames = listOf("UniqueWithCounts","UniqueWithCountsV2"),
        opName = "unique_with_counts",
        tensorNames = mutableMapOf("input" to "x")
)

val unpack = multipleNameMapping(inputFrameworkOpNames = listOf("Unpack"),
        opName = "unstack",
        tensorNames = mutableMapOf("input" to "value"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dimensions" to "axis","num" to "num"))))


val unsortedSegmentMax = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentMax",
        opName = "unsorted_segment_max",
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("numSegments" to "num_segments","numOfClasses" to "num_segments"))),
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"))

val unsortedSegmentMin = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentMin",
        opName = "unsorted_segment_min",
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("numSegments" to "num_segments","numOfClasses" to "num_segments"))),
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"))

val unsortedSegmentProd = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentProd",
        opName = "unsorted_segment_prod",
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("numSegments" to "num_segments","numOfClasses" to "num_segments"))),
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"))


val unsortedSegmentSum = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentSum",
        opName = "unsorted_segment_sum",
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("numSegments" to "num_segments","numOfClasses" to "num_segments"))),
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"))


val nextIteration = mapTensorNamesWithOp(inputFrameworkOpName = "NextIteration",opName = "next_iteration",
        tensorNames = mutableMapOf("input" to "data"))

val noOp = mapTensorNamesWithOp(inputFrameworkOpName = "NoOp",opName = "noop",tensorNames = mutableMapOf())

val where = mapTensorNamesWithOp(inputFrameworkOpName = "Where",opName = "Where",
        attributeMappingRules =  listOf(argDescriptorConstant(
                listOf(ArgDescriptor {
                        name = "inPlace"
                        boolValue = false
                        argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                        argIndex = 0

                })
        )),
        tensorNames = mutableMapOf("condition" to "input")
)

/*val whileOp = mapTensorNamesWithOp(inputFrameworkOpName = "While",opName = "While",
        tensorNames = mutableMapOf("condition" to "input")
)*/

val zerosLike = mapTensorNamesWithOp(inputFrameworkOpName = "ZerosLike",opName = "zeros_like",
        tensorNames = mutableMapOf("input" to "x"))

val zeta = mapTensorNamesWithOp(inputFrameworkOpName = "Zeta",opName = "zeta",
        tensorNames = mutableMapOf("input" to "x","q" to "q"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0))


object TensorflowOpDeclarations {
        init {
                OpRegistryHolder.registerOpMappingRegistry("tensorflow", tensorflowOpRegistry)
                singleTransformArgs.forEach {
                        defineTensorflowSingleTransform(inputFrameworkOpName = it.key,inputOpName = it.value)
                }

                elementWiseTransformOps.forEach {
                        defineTensorflowPairwiseTransforms(opName = it.value,inputFrameworkOpName = it.key)
                }

                tensorflowOps.opList.forEach {
                        tensorflowOpRegistry.registerInputFrameworkOpDef(it.name,it)
                }

                nd4jOpDescriptors.opListList.forEach {
                        tensorflowOpRegistry.registerNd4jOpDef(it.name,it)
                }

        }
}



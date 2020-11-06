package org.nd4j.codegen.ir.tensorflow

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
        "Atanh" to "atanh",
        "Ceil" to "ceil",
        "Copy" to "copy",
        "Cos" to "cos",
        "Cosh" to "cosh",
        "Erf" to "erf",
        "Erfc" to "erfc",
        "Exp" to "exp",
        "Expm1" to "expm1",
        "FloorMod" to "fmod",
        "Floor" to "floor",
        "HardSigmoid" to "hard_sigmoid",
        "HardTanh" to "hardtanh",
        "Log" to "log",
        "Log1p" to "log1p",
        "LogSigmoid" to "logsigmoid",
        "Maximum" to "max_pairwise",
        "Minimum" to "min_pairwise",
        "Mish" to "mish",
        "Neg" to "neg",
        "RationalTanh" to "rational_tanh",
        "Reciprocal" to "Reciprocal",
        "Inv" to "Reciprocal",
        "Rint" to "rint",
        "Round" to "round",
        "Rsqrt" to "rsqrt",
        "Selu" to "selu",
        "Sign" to "sign",
        "Sin" to "sin",
        "Sinh" to "sinh",
        "Softplus" to "softplus",
        "Softsign" to "softsign",
        "Swish" to "swish",
        "Square" to "square",
        "Sqrt" to "sqrt",
        "Tan" to "tan",
        "Tanh" to "tanh"
)

val elementWiseTransformOps = mapOf(
        "Add" to "add",
        "AddV2" to "add"
)


val reduceOps = mapOf(
        "AccumulateNV2" to "mergeadd",
        "Mean" to "mean"
)

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
        attributeMappingRules = listOf(valueMapping(mapOf()))
)

//BaseTransformBoolOp



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

class Conv2DMappingProcess: TensorflowMappingProcess(
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

object TensorflowOpDeclarations {
    init {
        OpRegistryHolder.registerOpMappingRegistry("tensorflow", tensorflowOpRegistry)
        singleTransformArgs.forEach {
            defineSingleTransform(inputFrameworkOpName = it.key,inputOpName = it.value)

        }
        Conv2DMappingProcess()
        ConstMappingProcess()
    }
}



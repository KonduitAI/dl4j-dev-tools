package org.nd4j.codegen.ir.tensorflow

import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.tensorflow.framework.*

val tensorflowOpRegistry = OpMappingRegistry<NodeDef,OpDef, TensorProto,DataType, OpDef.AttrDef,AttrValue>("tensorflow")
//val listOfRules =  listOf(NDArrayMappingRule("abs",mapOf("x" to "x", "y" to "y"))
class AbsMappingProcess: TensorflowMappingProcess(
        opName = "abs",
        inputFrameworkOpName = "Abs", tensorMappingRules =  listOf(NDArrayMappingRule(
        mappingNamesToPerform = mapOf("x" to "x", "y" to "y"))),
        opMappingRegistry = tensorflowOpRegistry)


class Conv2DMappingProcess: TensorflowMappingProcess(
        inputFramework = "tensorflow",
        inputFrameworkOpName = "Conv2D",
        opName = "conv2d",
        tensorMappingRules = listOf(mappingNDArrayInputs(mapOf(
                "input" to "input","filter" to "weights"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW",inputFrameworkAttributeName = "strides",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH",inputFrameworkAttributeName = "dilations",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW",inputFrameworkAttributeName = "dilations",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                sizeAtRule(outputAttributeName = "kH",dimensionIndex = 0,inputFrameworkAttributeName = "value"),
                sizeAtRule(outputAttributeName = "kW",dimensionIndex = 1,inputFrameworkAttributeName = "value")
        ),opMappingRegistry = tensorflowOpRegistry)




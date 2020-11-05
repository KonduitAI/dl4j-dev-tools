package org.nd4j.codegen.ir.tensorflow

import org.bytedeco.tensorflow.Conv2D
import org.nd4j.codegen.ir.*
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.ir.OpNamespace
import org.tensorflow.framework.*

val tensorflowOpRegistry = OpMappingRegistry<NodeDef,OpDef, TensorProto,DataType, OpDef.AttrDef,AttrValue>("tensorflow")
//val listOfRules =  listOf(NDArrayMappingRule("abs",mapOf("x" to "x", "y" to "y"))
class AbsMappingProcess: TensorflowMappingProcess(
        opName = "abs",
        inputFrameworkOpName = "Abs", tensorMappingRules =  listOf(NDArrayMappingRule(
        mappingNamesToPerform = mutableMapOf("x" to "x"))),
        opMappingRegistry = tensorflowOpRegistry)


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
        AbsMappingProcess()
        Conv2DMappingProcess()
        ConstMappingProcess()
    }
}


class TensorflowConditionalFieldValueIntIndexArrayRule
(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
        ConditionalFieldValueIntIndexArrayRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        (mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun conditionalFieldValueIntIndexArrayRule(outputAttribute: String,
                                           inputFrameworkAttributeName: String,
                                           targetValue: String,
                                           trueIndex: Int,
                                           falseIndex: Int): TensorflowConditionalFieldValueIntIndexArrayRule {
    return TensorflowConditionalFieldValueIntIndexArrayRule(
            mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttribute to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                name = "targetValue"
                stringValue = targetValue
            }.build(),
                    OpNamespace.ArgDescriptor.newBuilder().apply {
                        name = "trueIndex"
                        int32Value = trueIndex
                    }.build(),
                    OpNamespace.ArgDescriptor.newBuilder().apply {
                        name = "falseIndex"
                        int32Value = falseIndex
                    }.build()))
    )
}

class TensorflowNDArraySizeAt(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>): NDArraySizeAtRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun sizeAtRule(dimensionIndex: Int, outputAttributeName: String, inputFrameworkAttributeName: String): TensorflowNDArraySizeAt {
    return TensorflowNDArraySizeAt(
            mappingNamesToPerform = mapOf(outputAttributeName to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttributeName to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                name = inputFrameworkAttributeName
                int32Value = dimensionIndex
            }.build()))
    )
}

class TensorflowStringEqualsAdapterRule(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                        transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
        StringEqualsAdapterRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun stringEqualsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String): TensorflowStringEqualsAdapterRule {
    return TensorflowStringEqualsAdapterRule(
            mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttribute to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                name = inputFrameworkAttributeName
                stringValue = valueToTest
            }.build())))
}

class TensorflowSizeThresholdIntArrayIntIndexRule(mappingNamesToPerform: Map<String, String>,
                                                  transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : SizeThresholdIntArrayIntIndexRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {
    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}
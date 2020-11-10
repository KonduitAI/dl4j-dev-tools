package org.nd4j.codegen.ir.tensorflow

import org.nd4j.codegen.ir.*
import org.nd4j.ir.OpNamespace
import org.tensorflow.framework.*

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


class TensorflowStringNotEqualsAdapterRule(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                        transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
        StringNotEqualsAdapterRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun stringNotEqualsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String): TensorflowStringNotEqualsAdapterRule {
    return TensorflowStringNotEqualsAdapterRule(
            mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttribute to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                name = inputFrameworkAttributeName
                stringValue = valueToTest
            }.build())))
}


class TensorflowStringContainsAdapterRule(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                           transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
        StringNotEqualsAdapterRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun stringContainsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String): TensorflowStringContainsAdapterRule {
    return TensorflowStringContainsAdapterRule(
            mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttribute to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                name = inputFrameworkAttributeName
                stringValue = valueToTest
            }.build())))
}


class TensorflowAttributeScalarNDArrayAttribute(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
        AttributeScalarNDArrayAttribute<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun attributeScalarToNDArrayInput(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String): TensorflowAttributeScalarNDArrayAttribute {
    return TensorflowAttributeScalarNDArrayAttribute(
            mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttribute to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                name = inputFrameworkAttributeName
                stringValue = valueToTest
            }.build())))
}




class TensorflowValueMappingRule(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
        ValueMapping<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun valueMapping(mappings: Map<String,String>): TensorflowValueMappingRule {
    return TensorflowValueMappingRule(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}

class TensorflowBooleanToInt(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
        BooleanToInt<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun booleanToInt(mappings: Map<String,String>): TensorflowBooleanToInt {
    return TensorflowBooleanToInt(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}


class TensorflowNDArrayToIntAttributeValue(mappingNamesToPerform: Map<String, String>) : NDArrayToIntAttributeValue<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform = mappingNamesToPerform,transformerArgs = emptyMap()) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef,attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun ndarrayToIntList(ndarrayNameToAttributeName: MutableMap<String,String>): TensorflowNDArrayToIntAttributeValue {
    return TensorflowNDArrayToIntAttributeValue(mappingNamesToPerform = ndarrayNameToAttributeName)
}

class TensorflowNdArrayToStringIndex(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : StringToIndex<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun ndarrayStringToIndex(outputAttributeValue: String,inputAttributeValue: String, listOfValues: List<String>): TensorflowNdArrayToStringIndex {
    return TensorflowNdArrayToStringIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = mapOf(outputAttributeValue to listOfValues.map {
        valueName -> ArgDescriptor {
        name = valueName
        stringValue = valueName
    }
    }))
}



class TensorflowListNumberToListNumber(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : ListNumberToListNumber<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun listNumberToListNumber(outputAttributeValue: String, inputAttributeValue: String): TensorflowListNumberToListNumber {
    return TensorflowListNumberToListNumber(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}


class TensorflowAttributeNumberListNDArray(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : AttributeNumberListNDArray<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun convertNumberListToInputNDArray(outputAttributeValue: String, inputAttributeValue: String): TensorflowAttributeNumberListNDArray {
    return TensorflowAttributeNumberListNDArray(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}



class TensorflowNDArrayInputToScalarAttribute(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : NDArrayInputToScalarAttribute<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun convertNDArrayInputToScalarAttr(mutableMap: MutableMap<String,String>): TensorflowNDArrayInputToScalarAttribute {
    return TensorflowNDArrayInputToScalarAttribute(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}


class TensorflowArgDescriptorConstant(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : ArgDescriptorConstant<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

}

fun argDescriptorConstant(mutableMap: MutableMap<String,String>): TensorflowArgDescriptorConstant {
    return TensorflowArgDescriptorConstant(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}
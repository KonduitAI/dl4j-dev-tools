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

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }


}

fun conditionalFieldValueIntIndexArrayRule(outputAttribute: String,
                                           inputFrameworkStringNameToTest: String,
                                           targetValue: String,
                                           trueIndex: Int,
                                           falseIndex: Int,
                                           attributeNameOfListAttribute: String): TensorflowConditionalFieldValueIntIndexArrayRule {
    return TensorflowConditionalFieldValueIntIndexArrayRule(
            mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkStringNameToTest),
            transformerArgs = mapOf(outputAttribute to listOf(
                    ArgDescriptor {
                        name = "targetValue"
                        stringValue = targetValue
                    },
                    ArgDescriptor {
                        name = "trueIndex"
                        int32Value = trueIndex
                    },
                    ArgDescriptor {
                        name = "falseIndex"
                        int32Value = falseIndex
                    },
                    ArgDescriptor {
                        name = "attributeNameOfListAttribute"
                        stringValue = attributeNameOfListAttribute
                    }))
    )
}

class TensorflowNDArraySizeAt(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        NDArraySizeAtRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun sizeAtRule(dimensionIndex: Int,
               outputAttributeName: String,
               inputFrameworkAttributeName: String): TensorflowNDArraySizeAt {
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
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun stringEqualsRule(outputAttribute: String,
                     inputFrameworkAttributeName: String,
                     valueToTest: String): TensorflowStringEqualsAdapterRule {
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
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
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
        StringContainsAdapterRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
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
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun attributeScalarToNDArrayInput(outputAttribute: String, inputFrameworkAttributeName: String): TensorflowAttributeScalarNDArrayAttribute {
    return TensorflowAttributeScalarNDArrayAttribute(
            mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName))
}




class TensorflowValueMappingRule(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
        ValueMapping<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
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
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
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
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
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
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
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
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
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
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun convertNumberListToInputNDArray(outputAttributeValue: String, inputAttributeValue: String): TensorflowAttributeNumberListNDArray {
    return TensorflowAttributeNumberListNDArray(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}


class TensorflowListAttributeValueLookupToIndex(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : ListAttributeValueLookupToIndex<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun listAttributeValueLookupToIndex(outputAttributeValue: String, inputAttributeValue: String, idx: Int): TensorflowListAttributeValueLookupToIndex {
    return TensorflowListAttributeValueLookupToIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),
            transformerArgs = mapOf(outputAttributeValue to listOf(ArgDescriptor {
                argType = OpNamespace.ArgDescriptor.ArgType.INT32
                int32Value = idx
                name = "index"
            })))
}




class TensorflowNDArrayInputToNumericalAttribute(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
        NDArrayInputToNumericalAttribute<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun convertNDArrayInputToNumericalAttr(mutableMap: MutableMap<String,String>): TensorflowNDArrayInputToNumericalAttribute {
    return TensorflowNDArrayInputToNumericalAttribute(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}

//ListNumberToNDArray
class TensorflowListNumberToNDArray(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
        ListNumberToNDArray<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun listNumberToNDarray(mutableMap: MutableMap<String,String>): TensorflowListNumberToNDArray {
    return TensorflowListNumberToNDArray(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}

class TensorflowArgDescriptorConstant(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>)
    : ArgDescriptorConstant<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun argDescriptorConstant(argDescriptorConstants: List<OpNamespace.ArgDescriptor>): TensorflowArgDescriptorConstant {
    return TensorflowArgDescriptorConstant(mappingNamesToPerform = emptyMap(),transformerArgs = mapOf("value" to argDescriptorConstants))
}
package org.nd4j.codegen.ir.onnx

import onnx.Onnx
import org.nd4j.codegen.ir.*
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace

class NDArrayMappingRule(mappingNamesToPerform: MutableMap<String,String>,
                         transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()):
        BaseNDArrayMappingRule<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto,
                Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {



    override fun createTensorProto(input: Onnx.TensorProto): TensorNamespace.TensorProto {
        return OnnxIRTensor(input).toArgTensor()
    }

    override fun isInputTensorName(inputName: String): Boolean {
        val onnxOp = onnxops.first { opDef -> opDef.name == mappingProcess!!.inputFrameworkOpName() }
        return onnxOp.inputList.contains(inputName)
    }

    override fun isOutputTensorName(outputName: String): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess!!.opName())
        return nd4jOpDescriptor.argDescriptorList.filter { inputDescriptor -> inputDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
                .map {inputDescriptor -> inputDescriptor.name }.contains(outputName)
    }
}

fun mappingNDArrayInputs(inputs: MutableMap<String,String>) : NDArrayMappingRule {
    return NDArrayMappingRule(
            mappingNamesToPerform = inputs)
}

class OnnxConditionalFieldValueIntIndexArrayRule
(mappingNamesToPerform: MutableMap<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
        ConditionalFieldValueIntIndexArrayRule<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>
        (mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }

}

fun conditionalFieldValueIntIndexArrayRule(outputAttribute: String,
                                           inputFrameworkAttributeName: String,
                                           targetValue: String,
                                           trueIndex: Int,
                                           falseIndex: Int): OnnxConditionalFieldValueIntIndexArrayRule {
    return OnnxConditionalFieldValueIntIndexArrayRule(
            mappingNamesToPerform = mutableMapOf(outputAttribute to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttribute to listOf(ArgDescriptor {
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
                    }))
    )
}

class OnnxValueMapping(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : ValueMapping<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {
    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef,attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }
}

fun valueMappings(mappings: Map<String,String>): OnnxValueMapping {
    return OnnxValueMapping(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}


class OnnxBooleanToInt(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : BooleanToInt<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {
    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef,attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }


    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }
}

fun booleanToInt(mappings: Map<String,String>): OnnxBooleanToInt {
    return OnnxBooleanToInt(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}



class OnnxNDArraySizeAt(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>): NDArraySizeAtRule<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto):
            IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }



    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }
}

fun sizeAtRule(dimensionIndex: Int, outputAttributeName: String, inputFrameworkAttributeName: String): OnnxNDArraySizeAt {
    return OnnxNDArraySizeAt(
            mappingNamesToPerform = mapOf(outputAttributeName to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttributeName to listOf(ArgDescriptor {
                name = inputFrameworkAttributeName
                int32Value = dimensionIndex
            }))
    )
}

class OnnxStringEqualsAdapterRule(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                  transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
        StringEqualsAdapterRule<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>):
            List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }
}

fun stringEqualsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String): OnnxStringEqualsAdapterRule {
    return OnnxStringEqualsAdapterRule(
            mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttribute to listOf(ArgDescriptor {
                name = inputFrameworkAttributeName
                stringValue = valueToTest
            })))
}


class OnnxStringContainsAdapterRule(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                    transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
        StringContainsAdapterRule<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }

}

fun stringContainsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String): OnnxStringContainsAdapterRule {
    return OnnxStringContainsAdapterRule(
            mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttribute to listOf(ArgDescriptor {
                name = inputFrameworkAttributeName
                stringValue = valueToTest
            })))
}



class OnnxStringNotEqualsAdapterRule(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                     transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
        StringNotEqualsAdapterRule<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>):
            List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }


    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }
}

fun stringNotEqualsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String): OnnxStringNotEqualsAdapterRule {
    return OnnxStringNotEqualsAdapterRule(
            mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttribute to listOf(ArgDescriptor {
                name = inputFrameworkAttributeName
                stringValue = valueToTest
            })))
}



class OnnxNDArrayToIntAttributeValue(mappingNamesToPerform: Map<String, String>) : NDArrayToIntAttributeValue<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform = mappingNamesToPerform,transformerArgs = emptyMap()) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef,attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }


    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }
}

fun ndarrayToIntList(ndarrayNameToAttributeName: MutableMap<String,String>): OnnxNDArrayToIntAttributeValue {
    return OnnxNDArrayToIntAttributeValue(mappingNamesToPerform = ndarrayNameToAttributeName)
}

class OnnxSizeThresholdIntArrayIntIndexRule(mappingNamesToPerform: Map<String, String>,
                                            transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : SizeThresholdIntArrayIntIndexRule<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }


    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }

}

fun sizeThreshold(outputAttribute: String, inputFrameworkAttributeName: String, sizeThreshold: Long, index: Long, fallbackIndex: Long): OnnxSizeThresholdIntArrayIntIndexRule {
    return OnnxSizeThresholdIntArrayIntIndexRule(mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttribute to listOf(
                    ArgDescriptor {
                        name = "index"
                        int64Value = index
                    },
                    ArgDescriptor {
                        name = "sizeThreshold"
                        int64Value = sizeThreshold
                    },
                    ArgDescriptor {
                        name = "fallbackIndex"
                        int64Value = fallbackIndex
                    })))
}


class OnnxStringToIndex(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : StringToIndex<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }

}

fun stringToIndex(outputAttributeValue: String, inputAttributeValue: String, listOfValues: List<String>): OnnxStringToIndex {
    return OnnxStringToIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = mapOf(outputAttributeValue to listOfValues.map {
        valueName -> ArgDescriptor {
        name = valueName
        stringValue = valueName
    }
    }))
}
//ListAttributeValueLookupToIndex

class OnnxListAttributeValueLookupToIndex(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : ListAttributeValueLookupToIndex<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }

}

fun listAttributeValueLookup(outputAttributeValue: String, inputAttributeValue: String, indexValue: Int): OnnxListAttributeValueLookupToIndex {
    return OnnxListAttributeValueLookupToIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),
            transformerArgs = mapOf(outputAttributeValue to listOf(ArgDescriptor {
                name = inputAttributeValue
                int32Value = indexValue
            })
            ))
}

class OnnxListNumberToListNumber(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : ListNumberToListNumber<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }


    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }

}

fun listNumberToListNumber(outputAttributeValue: String, inputAttributeValue: String): OnnxListNumberToListNumber {
    return OnnxListNumberToListNumber(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}



class OnnxAttributeNumberListNDArray(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
        AttributeNumberListNDArray<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }



    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }
}

fun convertNumericalListToNDArray(outputAttributeValue: String, inputAttributeValue: String): OnnxAttributeNumberListNDArray {
    return OnnxAttributeNumberListNDArray(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}


//ListNumberToNDArray


class OnnxListNumberToNDArray(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : ListNumberToNDArray<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }
}

fun listNumberToNDarray(outputAttributeValue: String, inputAttributeValue: String): OnnxListNumberToNDArray {
    return OnnxListNumberToNDArray(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}





class OnnxNDArrayInputToNumericalAttribute(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : NDArrayInputToNumericalAttribute<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }
}

fun convertNDArrayInputToScalarAttr(outputAttributeValue: String, inputAttributeValue: String): OnnxNDArrayInputToNumericalAttribute {
    return OnnxNDArrayInputToNumericalAttribute(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}



class OnnxAttributeScalarNDArrayAttribute(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : AttributeScalarNDArrayAttribute<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }


    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }

}

fun attributeScalarToNDArrayInput(outputAttributeValue: String, inputAttributeValue: String): OnnxAttributeScalarNDArrayAttribute {
    return OnnxAttributeScalarNDArrayAttribute(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}




class OnnxArgDescriptorConstant(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : NDArrayInputToNumericalAttribute<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
    }




    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxTensorName(name,onnxOp)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return isOnnxAttributeName(name,onnxOp)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>): AttributeValueType {
        val onnxOp = onnxops.find { op -> op.name == mappingProcess.inputFrameworkOpName() }!!
        return onnxAttributeTypeFor(name,onnxOp)
    }
}

fun argDescriptorConstant(argDescriptorConstants: List<OpNamespace.ArgDescriptor>): OnnxArgDescriptorConstant {
    return OnnxArgDescriptorConstant(mappingNamesToPerform = emptyMap(),transformerArgs = mapOf("value" to argDescriptorConstants))
}




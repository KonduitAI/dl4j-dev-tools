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

class OnnxNDArraySizeAt(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>): NDArraySizeAtRule<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto):
            IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
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

}

fun stringEqualsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String): OnnxStringEqualsAdapterRule {
    return OnnxStringEqualsAdapterRule(
            mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
            transformerArgs = mapOf(outputAttribute to listOf(ArgDescriptor {
                name = inputFrameworkAttributeName
                stringValue = valueToTest
            })))
}

class OnnxSizeThresholdIntArrayIntIndexRule(mappingNamesToPerform: Map<String, String>,
                                            transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : SizeThresholdIntArrayIntIndexRule<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: Onnx.AttributeProto, attributeValueType: Onnx.AttributeProto): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        TODO("Not yet implemented")
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
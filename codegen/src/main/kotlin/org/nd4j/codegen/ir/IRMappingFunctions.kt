package org.nd4j.codegen.ir

import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

abstract class BaseAttributeExtractionRule<ATTR_DEF : GeneratedMessageV3, ATTR_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(opDescriptor: OpNamespace.OpDescriptor,
                                                                                                                                                             mappingNamesToPerform: Map<String, String>, inputAttributeDef: ATTR_DEF,
                                                                                                                                                             inputAttributeValue: ATTR_VALUE_TYPE,
                                                                                                                                                             transformerArgs: Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>>):
        AttributeMappingRule<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {
    protected val opDescriptor = opDescriptor
    protected val mappingNamesToPerform = mappingNamesToPerform
    protected val inputAttributeDef = inputAttributeDef
    protected val inputAttributeValue = inputAttributeValue
    protected val transformerArgs = transformerArgs


    override fun mappingTransformerArgs(): Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>> {
        return transformerArgs
    }

    fun inputAttribute(name: String): IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE> {
        return createIRAttribute(name, inputAttributeDef, inputAttributeValue)
    }

    abstract fun createIRAttribute(name: String, attrDef: ATTR_DEF, attributeValueType: ATTR_VALUE_TYPE): IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>

    abstract fun typeForAttribute(name: String): AttributeValueType

    override fun opDescriptor(): OpNamespace.OpDescriptor {
        return opDescriptor
    }

    override fun mappingNamesToPerform(): Map<String, String> {
        return mappingNamesToPerform
    }

    override fun mappingType(): MappingRuleType {
        return MappingRuleType.ATTRIBUTE
    }

    override fun serialize(): MapperNamespace.MappingRule {
        val builder = MapperNamespace.MappingRule.newBuilder()
        builder.ruleName = name()
        builder.functionName = name()
        val descriptorList = opDescriptor.argDescriptorList
        for((k,v) in transformerArgs) {
            val filteredList = descriptorList.filter { input -> input.name == k }
            require(filteredList.isNotEmpty()) {"Output attribute " + k + " was not found in op descriptor " + name() + " list of attribtues was " + descriptorList.map { input -> input.name }}

            val descriptor = filteredList[0]
            when(descriptor.argType) {
                OpNamespace.ArgDescriptor.ArgType.BOOL -> builder.addOutputBooleanName(k)
                OpNamespace.ArgDescriptor.ArgType.INT64 -> builder.addOutputIntName(k)
                OpNamespace.ArgDescriptor.ArgType.FLOAT -> builder.addOutputFloatName(k)
                OpNamespace.ArgDescriptor.ArgType.DOUBLE -> builder.addOutputDoubleName(k)
                OpNamespace.ArgDescriptor.ArgType.INT64 -> builder.addOutputIntName(k)
                OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR -> builder.addInputTensorName(k)
                OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR -> builder.addOutputTensorName(k)
            }

            for(associatedInput in v) {
                when(associatedInput.attributeValueType()) {
                    AttributeValueType.STRING -> builder.addInputStringAttrName(associatedInput.name())
                    AttributeValueType.BOOL -> builder.addInputBooleanName(associatedInput.name())
                    AttributeValueType.FLOAT -> builder.addInputFloatName(associatedInput.name())
                    AttributeValueType.INT -> builder.addInputIntName(associatedInput.name())
                    AttributeValueType.TENSOR -> builder.addInputTensorName(associatedInput.name())
                }
            }



        }


        return builder.build()
    }
}

abstract class StringEqualsAdapterRule<ATTR_DEF : GeneratedMessageV3, ATTR_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(opDescriptor: OpNamespace.OpDescriptor,
                                                                                                                                                         mappingNamesToPerform: Map<String, String>, inputAttributeDef: ATTR_DEF,
                                                                                                                                                         inputAttributeValue: ATTR_VALUE_TYPE,
                                                                                                                                                         transformerArgs: Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>>):
        BaseAttributeExtractionRule<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (opDescriptor, mappingNamesToPerform, inputAttributeDef, inputAttributeValue, transformerArgs)
        where DATA_TYPE: ProtocolMessageEnum {

    override fun name(): String {
        return "sizethresholdarrayint"
    }

    override fun convertAttributes(): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = transformerArgs[k]
            val compString = descriptorForName!![0].stringValue()
            val testValue = descriptorForName!![1].stringValue()
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = v
            descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.BOOL
            descriptorBuilder.boolValue = testValue == compString
            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}

abstract class SizeThresholdIntArrayIntIndexRule<ATTR_DEF : GeneratedMessageV3, ATTR_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(opDescriptor: OpNamespace.OpDescriptor,
                                                                                                                                                                   mappingNamesToPerform: Map<String, String>, inputAttributeDef: ATTR_DEF,
                                                                                                                                                                   inputAttributeValue: ATTR_VALUE_TYPE,
                                                                                                                                                                   transformerArgs: Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>>):
        BaseAttributeExtractionRule<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (opDescriptor, mappingNamesToPerform, inputAttributeDef, inputAttributeValue, transformerArgs) where DATA_TYPE: ProtocolMessageEnum {

    override fun name(): String {
        return "sizethresholdarrayint"
    }

    override fun convertAttributes(): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = transformerArgs[k]
            val inputArr = descriptorForName!![0].listIntValue()
            val index = descriptorForName!![1].intValue()
            val sizeThreshold = descriptorForName!![2].intValue()
            val fallbackIndex = descriptorForName!![3].stringValue()
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = v
            descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.INT64
            if(inputArr.size < sizeThreshold) {
                descriptorBuilder.int64Value = inputArr[fallbackIndex.toInt()]
            } else {
                descriptorBuilder.int64Value = inputArr[index.toInt()]
            }

            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}

abstract class ConditionalFieldValueIntIndexArrayRule<ATTR_DEF : GeneratedMessageV3, ATTR_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(opDescriptor: OpNamespace.OpDescriptor,
                                                                                                                                                                        mappingNamesToPerform: Map<String, String>, inputAttributeDef: ATTR_DEF,
                                                                                                                                                                        inputAttributeValue: ATTR_VALUE_TYPE,
                                                                                                                                                                        transformerArgs: Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>>):
        BaseAttributeExtractionRule<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (opDescriptor, mappingNamesToPerform, inputAttributeDef, inputAttributeValue, transformerArgs)
        where DATA_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum {

    override fun name(): String {
        return "conditionalfieldvalueintindex"
    }

    override fun convertAttributes(): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = transformerArgs[k]
            val inputArr = descriptorForName!![1].listIntValue()
            val trueIndex = descriptorForName!![2].intValue()
            val falseIndex = descriptorForName!![3].intValue()
            val targetValueToTest = descriptorForName!![0].stringValue()
            val testValue = descriptorForName!![4].stringValue()
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = v
            descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.INT64
            if(testValue == targetValueToTest) {
                descriptorBuilder.int64Value = inputArr[trueIndex.toInt()]
            } else {
                descriptorBuilder.int64Value = inputArr[falseIndex.toInt()]
            }

            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}

abstract class ExtractIntMappingRule<ATTR_DEF : GeneratedMessageV3, ATTR_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(opDescriptor: OpNamespace.OpDescriptor,
                                                                                                                                                       mappingNamesToPerform: Map<String, String>, inputAttributeDef: ATTR_DEF,
                                                                                                                                                       inputAttributeValue: ATTR_VALUE_TYPE,
                                                                                                                                                       transformerArgs: Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>>): BaseAttributeExtractionRule<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
(opDescriptor, mappingNamesToPerform, inputAttributeDef, inputAttributeValue, transformerArgs)
        where DATA_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum {

    override fun name(): String {
        return "extractint"
    }

    override fun convertAttributes(): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = getArgByName(k, opDescriptor)
            val inputAttributeDef = inputAttribute(k)
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = v
            descriptorBuilder.argType = descriptorForName.argType
            descriptorBuilder.dataTypeValue = descriptorForName.dataTypeValue
            if(inputAttributeDef.attributeValueType() != AttributeValueType.LIST_INT) {

            }
            else {
                val listIntValue = inputAttributeDef.listIntValue()
                val index = mappingTransformerArgs()[k]?.get(0)?.intValue()
                val valueAtIndex = listIntValue[index?.toInt()!!]
                descriptorBuilder.int64Value = valueAtIndex

            }

            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}

abstract class BaseNDArrayMappingRule<ATTR_DEF : GeneratedMessageV3, ATTR_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(nd4jOpName: String,
                                                                                                                                                        mappingNamesToPerform: Map<String, String> = emptyMap(),
                                                                                                                                                        transformerArgs: Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>> = emptyMap()):
        TensorMappingRule<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    protected val opDescriptor: OpNamespace.OpDescriptor
    protected  val mappingNamesToPerform = mappingNamesToPerform
    protected val transformerArgs = transformerArgs

    init {
        opDescriptor = nd4jOpDescriptors.opListList.filter { input -> input.name ==  nd4jOpName }[0]
    }

    override fun name(): String {
        return "ndarraymapping"
    }


    override fun mappingNamesToPerform(): Map<String, String> {
        return mappingNamesToPerform
    }

    override fun mappingTransformerArgs():  Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>> {
        return transformerArgs
    }

    override fun convertInput(): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()

        val mappingsToPerform = inputArgumentMappings()
        for(i in 0.. opDescriptor.argDescriptorCount - 1) {
            if(mappingsToPerform.containsKey(opDescriptor.getArgDescriptor(i).name)) {
                val outputName = mappingsToPerform[mappingsToPerform[opDescriptor.getArgDescriptor(i).name]]
                val builder = OpNamespace.ArgDescriptor.newBuilder()
                builder.argType = opDescriptor.argDescriptorList[i].argType
                builder.name = outputName
                require(opDescriptor.argDescriptorList[i].argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR) {"Input type must be INPUT_TENSOR"}
                builder.argIndex = opDescriptor.argDescriptorList[i].argIndex
                ret.add(builder.build())
            }

        }

        return ret
    }

    abstract fun createTensorProto(input: TENSOR_TYPE): TensorNamespace.TensorProto


    override fun convertInputsReverse(toReverse: List<OpNamespace.ArgDescriptor>): List<TENSOR_TYPE> {
        for(argument in toReverse) {
            require(argument.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR) {"Type to reverse must be an input tensor."}
        }
        TODO("Not yet implemented")
    }

    override fun inputArgumentMappings(): Map<String, String> {
        return mappingNamesToPerform
    }

    override fun opDescriptor(): OpNamespace.OpDescriptor {
        return opDescriptor
    }



    override fun mappingType(): MappingRuleType {
        return MappingRuleType.INPUT_TENSOR
    }

    override fun serialize(): MapperNamespace.MappingRule {
        val builder = MapperNamespace.MappingRule.newBuilder()
        builder.ruleName = name()
        builder.functionName = name()
        for((k,v) in transformerArgs) {
            val descriptor = opDescriptor.argDescriptorList.filter { input -> input.name == k }[0]
            when(descriptor.argType) {
                OpNamespace.ArgDescriptor.ArgType.BOOL -> builder.addOutputBooleanName(k)
                OpNamespace.ArgDescriptor.ArgType.INT64 -> builder.addOutputIntName(k)
                OpNamespace.ArgDescriptor.ArgType.FLOAT -> builder.addOutputFloatName(k)
                OpNamespace.ArgDescriptor.ArgType.DOUBLE -> builder.addOutputDoubleName(k)
                OpNamespace.ArgDescriptor.ArgType.INT64 -> builder.addOutputIntName(k)
                OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR -> builder.addInputTensorName(k)
                OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR -> builder.addOutputTensorName(k)
            }

            for(associatedInput in v) {
                when(associatedInput.attributeValueType()) {
                    AttributeValueType.STRING -> builder.addInputStringAttrName(associatedInput.name())
                    AttributeValueType.BOOL -> builder.addInputBooleanName(associatedInput.name())
                    AttributeValueType.FLOAT -> builder.addInputFloatName(associatedInput.name())
                    AttributeValueType.INT -> builder.addInputIntName(associatedInput.name())
                    AttributeValueType.TENSOR -> builder.addInputTensorName(associatedInput.name())
                }
            }



        }


        return builder.build()
    }

}
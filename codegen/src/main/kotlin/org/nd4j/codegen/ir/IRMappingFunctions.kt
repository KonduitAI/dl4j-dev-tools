package org.nd4j.codegen.ir

import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

abstract class BaseAttributeExtractionRule<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(
        name: String,
        mappingNamesToPerform: Map<String, String>,
        transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        AttributeMappingRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    protected var opDescriptor: OpNamespace.OpDescriptor? = null
    protected val mappingNamesToPerform = mappingNamesToPerform
    protected val transformerArgs = transformerArgs
    protected val name = name
    protected var opDef: OP_DEF_TYPE? = null


    override fun initWithMappingProcess(mappingProcess: MappingProcess<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>) {
        this.opDef = mappingProcess.inputOpDef()

    }

    override fun mappingNamesToPerform(): Map<String, String> {
        return mappingNamesToPerform
    }

    override fun name(): String {
        return name
    }

    override fun mappingTransformerArgs(): Map<String, List<OpNamespace.ArgDescriptor>> {
        return transformerArgs
    }



    abstract fun createIRAttribute(name: String, attrDef: ATTR_DEF, attributeValueType: ATTR_VALUE_TYPE): IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>



    override fun serialize(): MapperNamespace.MappingRule {
        val builder = MapperNamespace.MappingRule.newBuilder()
        builder.ruleName = name()
        builder.functionName = name()
        val descriptorList = opDescriptor!!.argDescriptorList
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
                when(associatedInput.argType) {
                    AttributeValueType.STRING -> builder.addInputStringAttrName(associatedInput.name)
                    AttributeValueType.BOOL -> builder.addInputBooleanName(associatedInput.name)
                    AttributeValueType.FLOAT -> builder.addInputFloatName(associatedInput.name)
                    AttributeValueType.INT -> builder.addInputIntName(associatedInput.name)
                    AttributeValueType.TENSOR -> builder.addInputTensorName(associatedInput.name)
                }
            }



        }


        return builder.build()
    }
}

abstract class StringEqualsAdapterRule<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(
        mappingNamesToPerform: Map<String, String> = emptyMap(),
        transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "sizethresholdarrayint",
                mappingNamesToPerform =  mappingNamesToPerform,
                transformerArgs = transformerArgs)
        where DATA_TYPE: ProtocolMessageEnum {

    override fun convertAttributes(inputNode: NODE_TYPE): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = transformerArgs[k]
            val compString = descriptorForName!![0].stringValue
            val testValue = getIRAttribute(v,inputNode).stringValue()
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = v
            descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.BOOL
            descriptorBuilder.boolValue = testValue == compString
            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}

abstract class SizeThresholdIntArrayIntIndexRule<OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(mappingNamesToPerform: Map<String, String>,
                                                     transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "sizethresholdarrayint", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) where DATA_TYPE: ProtocolMessageEnum {


    override fun convertAttributes(inputNode: NODE_TYPE): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = transformerArgs[k]
            val inputArr = getIRAttribute(k, inputNode).listIntValue()
            val index = descriptorForName!![0].int32Value
            val sizeThreshold = descriptorForName!![1].int64Value
            val fallbackIndex = descriptorForName!![2].stringValue
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

abstract class ConditionalFieldValueIntIndexArrayRule<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(mappingNamesToPerform: Map<String, String>,
                                                     transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "conditionalfieldvalueintindex", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs)
        where  DATA_TYPE: ProtocolMessageEnum {


    override fun convertAttributes(inputNode: NODE_TYPE): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = transformerArgs[k]
            val inputArr = getIRAttribute(v, inputNode).listIntValue()
            val trueIndex = descriptorForName!![1].int32Value
            val falseIndex = descriptorForName!![2].int32Value
            val targetValueToTest = descriptorForName!![0].stringValue
            val testValue = getIRAttribute(v,inputNode).stringValue()
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = v
            descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.INT64
            if(testValue == targetValueToTest) {
                descriptorBuilder.int64Value = inputArr[trueIndex]
            } else {
                descriptorBuilder.int64Value = inputArr[falseIndex]
            }

            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}

abstract class ExtractIntMappingRule<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(mappingNamesToPerform: Map<String, String>,
                                                     transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "extractint", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs)
        where DATA_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum {


    override fun convertAttributes(inputNode: NODE_TYPE): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = getArgByName(k, opDescriptor!!)
            val inputAttributeDef = getAttributeDefFromName(k)
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = v
            descriptorBuilder.argType = descriptorForName.argType
            descriptorBuilder.dataTypeValue = descriptorForName.dataTypeValue
//            if(inputAttributeDef.attributeValueType() != AttributeValueType.LIST_INT) {
//
//            }
//            else {
//                val listIntValue = inputAttributeDef.listIntValue()
//                val index = mappingTransformerArgs()[k]?.get(0)?.intValue()
//                val valueAtIndex = listIntValue[index?.toInt()!!]
//                descriptorBuilder.int64Value = valueAtIndex
//
//            }

            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}

abstract class BaseNDArrayMappingRule<OP_DEF_TYPE: GeneratedMessageV3
        ,NODE_DEF_TYPE: GeneratedMessageV3,ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3,
        DATA_TYPE>(mappingNamesToPerform: Map<String, String> = emptyMap(),
                   transformerArgs: Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>> = emptyMap()):
        TensorMappingRule<OP_DEF_TYPE,NODE_DEF_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    protected var opDescriptor: OpNamespace.OpDescriptor? = null
    protected  val mappingNamesToPerform = mappingNamesToPerform
    protected val transformerArgs = transformerArgs
    protected var mappingProcess: MappingProcess<OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTR_DEF,ATTR_VALUE_TYPE,DATA_TYPE>? = null
    protected  var opDef: OP_DEF_TYPE? = null


    override fun initWithMappingProcess(mappingProcess: MappingProcess<OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>) {
        opDescriptor = nd4jOpDescriptors.opListList.filter { input -> input.name ==  mappingProcess.opName() }[0]
        this.mappingProcess = mappingProcess
        this.opDef = mappingProcess.inputOpDef()
    }

    override fun inputOpDescriptor(): OP_DEF_TYPE {
        return opDef!!
    }


    override fun name(): String {
        return "ndarraymapping"
    }


    override fun mappingNamesToPerform(): Map<String, String> {
        return mappingNamesToPerform
    }


    override fun convertInput(): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()

        val mappingsToPerform = inputArgumentMappings()
        for(i in 0 until opDescriptor!!.argDescriptorCount) {
            if(mappingsToPerform.containsKey(opDescriptor!!.getArgDescriptor(i).name)) {
                val outputName = mappingsToPerform[mappingsToPerform[opDescriptor!!.getArgDescriptor(i).name]]
                val builder = OpNamespace.ArgDescriptor.newBuilder()
                builder.argType = opDescriptor!!.argDescriptorList[i].argType
                builder.name = outputName
                require(opDescriptor!!.argDescriptorList[i].argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR) {"Input type must be INPUT_TENSOR"}
                builder.argIndex = opDescriptor!!.argDescriptorList[i].argIndex
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

    override fun serialize(): MapperNamespace.MappingRule {
        val builder = MapperNamespace.MappingRule.newBuilder()
        builder.ruleName = name()
        builder.functionName = name()
        for((k,v) in transformerArgs) {
            val descriptor = opDescriptor!!.argDescriptorList.filter { input -> input.name == k }[0]
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
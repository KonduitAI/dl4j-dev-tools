package org.nd4j.codegen.ir

import org.nd4j.gen.OpDeclarationDescriptor
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.CustomOp
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum


interface IRTensor<TENSOR_TYPE: GeneratedMessageV3, DATA_TYPE>
        where  DATA_TYPE: ProtocolMessageEnum {
    fun shape(): List<Long>
    fun stride(): List<Long>
    fun dataType(): IRDataType<DATA_TYPE>
    fun toArgTensor(): TensorNamespace.TensorProto
}


enum class AttributeValueType {
    FLOAT,
    LIST_FLOAT,
    BYTE,
    LIST_BYTE,
    INT,
    LIST_INT,
    BOOL,
    LIST_BOOL,
    STRING,
    LIST_STRING,
    TENSOR,
    LIST_TENSOR,
    INVALID
}

interface IRAttribute<ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    fun name(): String

    fun floatValue(): Float

    fun listFloatValue(): List<Float>

    fun tensorValue(): IRTensor<TENSOR_TYPE, DATA_TYPE>

    fun listTensorValue(): List<IRTensor<TENSOR_TYPE, DATA_TYPE>>

    fun intValue(): Long

    fun listIntValue(): List<Long>

    fun boolValue(): Boolean

    fun listBoolValue(): List<Boolean>

    fun stringValue(): String

    fun listStringValue(): List<String>

    fun attributeValueType(): AttributeValueType

    fun internalAttributeDef(): ATTRIBUTE_TYPE


    fun internalAttributeValue(): ATTRIBUTE_VALUE_TYPE
}



interface MappingProcess<NODE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE >
        where DATA_TYPE: ProtocolMessageEnum {
    fun opName(): String

    fun frameworkVersion(): String

    fun inputFramework(): String

    fun rules(): List<MappingRule<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>


    fun applyProcess(inputNode: IRNode<NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): OpDeclarationDescriptor

    fun applyProcessReverse(input: OpDeclarationDescriptor): IRNode<NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>

    fun createDescriptor(argDescriptors: List<OpNamespace.ArgDescriptor>): OpDeclarationDescriptor
}

enum class MappingRuleType {
    ATTRIBUTE,
    INPUT_TENSOR,
    OUTPUT_TENSOR
}

interface MappingRule<ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {
    fun name(): String

    /**
     * Map of string to list of arguments needed for a transform
     */
    fun mappingTransformerArgs(): Map<String, List<IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>>

    /**
     * Convert 1 or more attributes in to a list of {@link ArgDescriptor}
     */
    fun convertInput(): List<OpNamespace.ArgDescriptor>

    fun convertAttributes(): List<OpNamespace.ArgDescriptor>

    fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>

    fun convertInputsReverse(toReverse: List<OpNamespace.ArgDescriptor>): List<TENSOR_TYPE>

    fun inputArgumentMapping(): Map<String, String>


    fun opDescriptor(): OpNamespace.OpDescriptor

    fun inputTensorsToConvert(): Map<String, TENSOR_TYPE>

    fun getInputTensor(key: String): TENSOR_TYPE

    fun inputAttributeDefsToConvert(): Map<String, ATTRIBUTE_TYPE>

    fun getInputAttribute(input: String): ATTRIBUTE_TYPE

    fun inputAttributeValuesToConvert(): Map<String, ATTRIBUTE_VALUE_TYPE>

    fun getInputAttributeValue(input: String): ATTRIBUTE_VALUE_TYPE

    fun mappingType(): MappingRuleType
}



interface IRNode<NODE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE>
        where  DATA_TYPE: ProtocolMessageEnum {
    /**
     * List of inputs in to the node
     * @return the list of input names for this node
     */
    fun  createInputsFrom(inputData: List<TENSOR_TYPE>): List<IRTensor<TENSOR_TYPE, DATA_TYPE>>

    /**
     * List of outputs
     * @return the list of output names for this node
     */
    fun createOutputsFrom(inputValues: List<TENSOR_TYPE>): List<IRTensor<TENSOR_TYPE, DATA_TYPE>>

    /**
     * Op name
     */
    fun opName(): String

    /**
     * The name of the node
     * @return the name of the node
     */
    fun nodeName(): String

    /**
     * The input at a particular index
     * @return the name at the particular index
     */
    fun inputAt(index: Int): String
    fun outputAt(index: Int): String
    fun attributeMap(): Map<String, IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>
    fun getAttribute(inputName: String): IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
    fun hasAttribute(inputName: String): Boolean

    fun internalValue(): NODE_TYPE
}

interface IRArgDef<T : GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {
    fun name(): String

    fun description(): String

    fun dataType(): IRDataType<DATA_TYPE>

    fun internalValue(): T

    fun indexOf(): Integer
}

interface IROpDef<T : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, ARG_DEF_TYPE : GeneratedMessageV3, DATA_TYPE, ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3>
        where DATA_TYPE: ProtocolMessageEnum {
    fun opName(): String

    fun internalValue(): T

    fun inputArgs(): List<IRArgDef<ARG_DEF_TYPE, DATA_TYPE>>

    fun outputArgs(): List<IRArgDef<ARG_DEF_TYPE, DATA_TYPE>>

    fun attributes(): List<IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>

}

enum class IRDataTypeValue {
    DT_FLOAT,
    DT_DOUBLE,
    DT_INT32,
    DT_UINT8,
    DT_INT16,
    DT_INT8,
    DT_STRING,
    DT_COMPLEX64,  // Single-precision complex
    DT_INT64,
    DT_BOOL,
    DT_QINT8,     // Quantized int8
    DT_QUINT8,    // Quantized uint8
    DT_QINT32,    // Quantized int32
    DT_BFLOAT16,  // Float32 truncated to 16 bits.  Only for cast ops.
    DT_QINT16,    // Quantized int16
    DT_QUINT16,   // Quantized uint16
    DT_UINT16,
    DT_COMPLEX128,  // Double-precision complex
    DT_HALF,
    DT_RESOURCE,
    DT_VARIANT,  // Arbitrary C++ data types
    DT_UINT32,
    DT_UINT64,
    DT_INVALID

}

interface IRDataType<DATATYPE_TYPE> {
    fun convertToDataType(input: DATATYPE_TYPE): IRDataTypeValue

    fun dataType(): IRDataTypeValue

    fun internalValue(): DATATYPE_TYPE

}

interface Mapper<NODE_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, OP_DEF_TYPE, DATATYPE_TYPE> {


    /**
     * Creates a {@link CustomOp}
     * from the given input
     */
    fun createOpFrom(input: OP_DEF_TYPE) : CustomOp

    /**
     * Returns the name of the op in tensorflow
     * mapping to the one in nd4j.
     */
    fun mapNameOp(input: String): String

    fun map(input: NODE_TYPE) : INDArray

    fun map(input: OP_DEF_TYPE) : OpNamespace.OpDescriptor

    fun opDefList(): List<OP_DEF_TYPE>

    fun mapAttr(input: ATTR_DEF_TYPE, inputValue: ATTR_VALUE_TYPE): OpNamespace.ArgDescriptor

    fun typeFor(input: ATTR_DEF_TYPE): OpNamespace.ArgDescriptor.ArgType

    fun nd4jOpDefList(): OpNamespace.OpDescriptorList

    fun dataTypesForArgument(input: OP_DEF_TYPE, argName: String): List<DataType>

    fun typeFor(tensorflowType: DATATYPE_TYPE): DataType
}


fun getArgByName(name: String, descriptor: OpNamespace.OpDescriptor): OpNamespace.ArgDescriptor {
    for(i in 0 .. descriptor.argDescriptorCount) {
        if (descriptor.argDescriptorList[i].name == name)
            return descriptor.argDescriptorList[i]
    }

    return OpNamespace.ArgDescriptor.getDefaultInstance()
}


abstract class BaseAttributeExtractionRule<ATTR_DEF : GeneratedMessageV3, ATTR_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(opDescriptor: OpNamespace.OpDescriptor,
                                                                                                                                                             mappingNamesToPerform: Map<String, String>, inputAttributeDef: ATTR_DEF,
                                                                                                                                                             inputAttributeValue: ATTR_VALUE_TYPE,
                                                                                                                                                             transformerArgs: Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>>): MappingRule<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
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


    override fun opDescriptor(): OpNamespace.OpDescriptor {
        return opDescriptor
    }

    override fun inputArgumentMapping(): Map<String, String> {
        return mappingNamesToPerform
    }

    override fun mappingType(): MappingRuleType {
        return MappingRuleType.ATTRIBUTE
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
        for((k, v) in inputArgumentMapping()) {
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
        (opDescriptor, mappingNamesToPerform, inputAttributeDef, inputAttributeValue, transformerArgs) where DATA_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum {

    override fun name(): String {
        return "sizethresholdarrayint"
    }

    override fun convertAttributes(): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in inputArgumentMapping()) {
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
        for((k, v) in inputArgumentMapping()) {
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
        for((k, v) in inputArgumentMapping()) {
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


abstract class BaseNDArrayMappingRule<ATTR_DEF : GeneratedMessageV3, ATTR_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>
(opDescriptor: OpNamespace.OpDescriptor, inputTensors: Map<String, TENSOR_TYPE>,
 mappingNamesToPerform: Map<String, String>,
 transformerArgs: Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>>):
        MappingRule<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    protected val opDescriptor = opDescriptor
    protected val inputTensors = inputTensors
    protected  val mappingNamesToPerform = mappingNamesToPerform
    protected val transformerArgs = transformerArgs

    override fun name(): String {
        return "ndarraymapping"
    }

    override fun getInputTensor(key: String): TENSOR_TYPE {
        return inputTensors.getValue(key)
    }

    override fun getInputAttribute(input: String): ATTR_DEF {
       return  inputAttributeDefsToConvert().getValue(input)
    }

    override fun getInputAttributeValue(input: String): ATTR_VALUE_TYPE {
     return inputAttributeValuesToConvert().getValue(input)
    }

    override fun mappingTransformerArgs():  Map<String, List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>> {
        return transformerArgs
    }

    override fun convertInput(): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        val mappingsToPerform = inputArgumentMapping()
        for(i in 0.. opDescriptor.argDescriptorCount) {
            if(mappingsToPerform.containsKey(opDescriptor.getArgDescriptor(i).name)) {
                val outputName = mappingsToPerform[mappingsToPerform[opDescriptor.getArgDescriptor(i).name]]
                val builder = OpNamespace.ArgDescriptor.newBuilder()
                builder.argType = opDescriptor.argDescriptorList[i].argType
                builder.name = outputName
                require(opDescriptor.argDescriptorList[i].argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR) {"Input type must be INPUT_TENSOR"}
                builder.argIndex = opDescriptor.argDescriptorList[i].argIndex
                val tensorToConvert = getInputTensor(opDescriptor.getArgDescriptor(i).name)
                builder.inputValue = createTensorProto(tensorToConvert)
                ret.add(builder.build())
            }

        }

        return ret
    }

    abstract fun createTensorProto(input: TENSOR_TYPE): TensorNamespace.TensorProto

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>> {
        return emptyList()
    }

    override fun convertInputsReverse(toReverse: List<OpNamespace.ArgDescriptor>): List<TENSOR_TYPE> {
        for(argument in toReverse) {
            require(argument.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR) {"Type to reverse must be an input tensor."}
        }
        TODO("Not yet implemented")
    }

    override fun inputArgumentMapping(): Map<String, String> {
        return mappingNamesToPerform
    }

    override fun opDescriptor(): OpNamespace.OpDescriptor {
        return opDescriptor
    }

    override fun inputTensorsToConvert(): Map<String, TENSOR_TYPE> {
        return inputTensors
    }

    override fun convertAttributes(): List<OpNamespace.ArgDescriptor> {
        TODO("Not yet implemented")
    }

    override fun inputAttributeDefsToConvert(): Map<String, ATTR_DEF> {
        return emptyMap()
    }

    override fun inputAttributeValuesToConvert(): Map<String, ATTR_VALUE_TYPE> {
        return emptyMap()
    }

    override fun mappingType(): MappingRuleType {
        return MappingRuleType.INPUT_TENSOR
    }

}



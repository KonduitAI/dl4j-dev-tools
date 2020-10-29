package org.nd4j.codegen.ir

import org.apache.commons.io.IOUtils
import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.VariableType
import org.nd4j.common.io.ClassPathResource
import org.nd4j.gen.OpDeclarationDescriptor
import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.CustomOp
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.nd4j.shade.protobuf.TextFormat
import java.nio.charset.Charset



fun loadNd4jOpDescriptors(): OpNamespace.OpDescriptorList {
    val nd4jOpDescriptorResourceStream = ClassPathResource("op-ir.proto").inputStream
    val resourceString = IOUtils.toString(nd4jOpDescriptorResourceStream, Charset.defaultCharset())
    val descriptorListBuilder = OpNamespace.OpDescriptorList.newBuilder()
    TextFormat.merge(resourceString,descriptorListBuilder)
    val ret = descriptorListBuilder.build()
    val mutableList = ArrayList(ret.opListList)
    mutableList.sortBy { it.name }

    val newResultBuilder = OpNamespace.OpDescriptorList.newBuilder()
    newResultBuilder.addAllOpList(mutableList)
    return newResultBuilder.build()
}


val nd4jOpDescriptors = loadNd4jOpDescriptors()



interface IR<NODE_TYPE : GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3, DATA_TYPE,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,OP_LIST_TYPE: GeneratedMessageV3>
        where  DATA_TYPE: ProtocolMessageEnum {

    fun nd4jListOps(): OpNamespace.OpDescriptorList

    fun inputFrameworkOpDefs(): OP_LIST_TYPE

    fun createNode(name: String, inputs: List<String>, opName: String, nodeAttributes: Map<String, ATTRIBUTE_VALUE_TYPE>): NODE_TYPE

    fun createIRNode(name: String,inputNode: NODE_TYPE): IRNode<NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>

    fun <T> createTensor(name: String, dataType: DATA_TYPE, shape: List<Long>,inputData: T): TENSOR_TYPE

    fun createIRTensor(inputTensor: TENSOR_TYPE): IRTensor<TENSOR_TYPE, DATA_TYPE>

    fun createAttributeValue(name: String, attributeValue: AttributeValueType): ATTRIBUTE_VALUE_TYPE


    fun createAttributeDef(name: String, description: String, type: String): ATTRIBUTE_TYPE

    fun createIRAttribute(attributeDef: ATTRIBUTE_TYPE,attributeValue: ATTRIBUTE_VALUE_TYPE): IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>

    fun createDataType(name: String, dataType: TensorNamespace.DataType): DATA_TYPE

    fun createIRDataType(dataType: DATA_TYPE): IRDataType<DATA_TYPE>


}

interface IRTensor<TENSOR_TYPE: GeneratedMessageV3, DATA_TYPE>
        where  DATA_TYPE: ProtocolMessageEnum {
    fun shape(): List<Long>
    fun stride(): List<Long>
    fun dataType(): IRDataType<DATA_TYPE>
    fun toArgTensor(): TensorNamespace.TensorProto
    fun rawValue(): TENSOR_TYPE

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



interface MappingProcess<OP_DEF_TYPE: GeneratedMessageV3,NODE_DEF_TYPE: GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    fun lookupInputOpDef(opName: String): OP_DEF_TYPE

    fun inputOpDef(): OP_DEF_TYPE

    fun opName(): String

    fun frameworkVersion(): String

    fun inputFramework(): String

    fun attributeMappingRules(): List<AttributeMappingRule<OP_DEF_TYPE,NODE_DEF_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>

    fun tensorMappingRules():  List<TensorMappingRule<OP_DEF_TYPE,NODE_DEF_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>

    fun applyProcess(inputNode: IRNode<NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): OpNamespace.OpDescriptor

    fun applyProcessReverse(input: OpDeclarationDescriptor): IRNode<NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>

    fun serialize(): MapperNamespace.MapperDeclaration

}



enum class MappingRuleType {
    ATTRIBUTE,
    INPUT_TENSOR,
    OUTPUT_TENSOR
}


interface TensorMappingRule<OP_DEF_TYPE: GeneratedMessageV3,NODE_DEF_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {


    fun initWithMappingProcess(mappingProcess: MappingProcess<OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>)

    fun inputOpDescriptor(): OP_DEF_TYPE

    fun name(): String


    fun serialize(): MapperNamespace.MappingRule


    fun mappingNamesToPerform(): Map<String,String>

    /**
     * Convert 1 or more attributes in to a list of {@link ArgDescriptor}
     */
    fun convertInput(): List<OpNamespace.ArgDescriptor>


    fun inputArgumentMappings(): Map<String, String>

    fun convertInputsReverse(toReverse: List<OpNamespace.ArgDescriptor>): List<TENSOR_TYPE>



}


interface AttributeMappingRule<OP_DEF_TYPE: GeneratedMessageV3,NODE_DEF_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    fun initWithMappingProcess(mappingProcess: MappingProcess<OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>)


    fun  getIRAttribute(name: String, nodeWithValues: NODE_DEF_TYPE): IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>

    fun mappingNamesToPerform(): Map<String,String>

    fun mappingTransformerArgs(): Map<String, List<OpNamespace.ArgDescriptor>>

    fun name(): String

    fun serialize(): MapperNamespace.MappingRule

    fun getAttributeDefFromName(inputName: String): ATTRIBUTE_TYPE

    fun convertAttributes(inputNode: NODE_DEF_TYPE): List<OpNamespace.ArgDescriptor>

    fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>
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

interface IRDataType<DATA_TYPE> where DATA_TYPE: ProtocolMessageEnum {
    fun convertToDataType(input: DATA_TYPE): IRDataTypeValue

    fun dataType(): IRDataTypeValue

    fun internalValue(): DATA_TYPE

}

interface Mapper<NODE_TYPE: GeneratedMessageV3, ATTR_DEF_TYPE: GeneratedMessageV3, ATTR_VALUE_TYPE: GeneratedMessageV3, OP_DEF_TYPE: GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {


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

    fun typeFor(tensorflowType: DATA_TYPE): DataType
}

abstract  class AbstractMappingProcess<OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE>(inputFramework: String,
                                                              frameworkVersion: String,
                                                              inputFrameworkOpName: String,
                                                              opName: String,
                                                              tensorMappingRules: List<TensorMappingRule<OP_DEF_TYPE,NODE_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>,
                                                              attributeMappingRules: List<AttributeMappingRule<OP_DEF_TYPE,NODE_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>):
        MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    protected val inputFramework = inputFramework
    protected val frameworkVersion = frameworkVersion
    protected val inputFrameworkOpName = inputFrameworkOpName
    protected val opName = opName
    protected val tensorMappingRules = tensorMappingRules
    protected val attributeMappingRules = attributeMappingRules
    protected var opDef: OP_DEF_TYPE? = null

    init {
        tensorMappingRules.forEach {
            it.initWithMappingProcess(this)
        }

        attributeMappingRules.forEach {
            it.initWithMappingProcess(this)
        }
    }

    override fun inputOpDef(): OP_DEF_TYPE {
        if(opDef == null)
            this.opDef = lookupInputOpDef(inputFrameworkOpName)
        return opDef!!
    }

    override fun attributeMappingRules(): List<AttributeMappingRule<OP_DEF_TYPE,NODE_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>> {
        return attributeMappingRules
    }

    override fun tensorMappingRules(): List<TensorMappingRule<OP_DEF_TYPE,NODE_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>> {
        return tensorMappingRules
    }

    override fun applyProcessReverse(input: OpDeclarationDescriptor): IRNode<NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        TODO("Not yet implemented")
    }

    override fun opName(): String {
        return opName
    }

    override fun frameworkVersion(): String {
        return frameworkVersion
    }

    override fun inputFramework(): String {
        return inputFramework
    }

    override fun applyProcess(inputNode: IRNode<NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): OpNamespace.OpDescriptor {
        val descriptorBuilder = OpNamespace.OpDescriptor.newBuilder()
        descriptorBuilder.name = opName()
        tensorMappingRules.forEach {
            it.convertInput().forEach { descriptor ->
                descriptorBuilder.addArgDescriptor(descriptor)
            }
        }


        attributeMappingRules.forEach {
            it.convertAttributes(inputNode.internalValue()).forEach {
                descriptor -> descriptorBuilder.addArgDescriptor(descriptor)
            }
        }

        return descriptorBuilder.build()
    }

    override fun serialize(): MapperNamespace.MapperDeclaration {
        val retBuilder = MapperNamespace.MapperDeclaration.newBuilder()
        retBuilder.frameworkName = inputFramework()
        retBuilder.opName = opName()

        tensorMappingRules.forEach {
            retBuilder.ruleBuilderList.add(it.serialize().toBuilder())
        }

        attributeMappingRules.forEach {
            retBuilder.ruleBuilderList.add(it.serialize().toBuilder())
        }

        return retBuilder.build()
    }
}



fun convertOpDescriptorToFunction(inputDescriptor: OpNamespace.OpDescriptor, sameDiff: SameDiff): DifferentialFunction {
    val dynamicCustomOpsBuilder = DynamicCustomOp.builder(inputDescriptor.name)
    val intArgs = ArrayList<OpNamespace.ArgDescriptor>()
    val tArgs = ArrayList<OpNamespace.ArgDescriptor>()
    val inputArrays = ArrayList<OpNamespace.ArgDescriptor>()
    val outputArrays = ArrayList<OpNamespace.ArgDescriptor>()
    val boolArgs = ArrayList<OpNamespace.ArgDescriptor>()
    val sdVariables = ArrayList<SDVariable>()
    inputDescriptor.argDescriptorList.forEach { argDescriptor ->
        when(argDescriptor.argType) {
            OpNamespace.ArgDescriptor.ArgType.FLOAT, OpNamespace.ArgDescriptor.ArgType.DOUBLE -> {
                tArgs.add(argDescriptor)
            }

            OpNamespace.ArgDescriptor.ArgType.INT64, OpNamespace.ArgDescriptor.ArgType.INT32 -> {
                intArgs.add(argDescriptor)
            }

            OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR -> {
                val variable = SDVariable(argDescriptor.name, VariableType.VARIABLE,sameDiff,null,null)
                sdVariables.add(variable)
                inputArrays.add(argDescriptor)
            }

            OpNamespace.ArgDescriptor.ArgType.BOOL -> {
                boolArgs.add(argDescriptor)
            }

        }
    }

    intArgs.sortBy { it.argIndex }
    intArgs.forEach { dynamicCustomOpsBuilder.addIntegerArguments(it.int64Value) }
    inputArrays.sortBy { it.argIndex }
    tArgs.sortBy { it.argIndex }
    tArgs.forEach { dynamicCustomOpsBuilder.addFloatingPointArguments(it.doubleValue) }

    boolArgs.sortBy { it.argIndex }
    boolArgs.forEach { dynamicCustomOpsBuilder.addBooleanArguments(it.boolValue) }

    sameDiff.addArgsFor(sdVariables.toTypedArray(),dynamicCustomOpsBuilder.build())
    return dynamicCustomOpsBuilder.build()
}


fun getArgByName(name: String, descriptor: OpNamespace.OpDescriptor): OpNamespace.ArgDescriptor {
    for(i in 0 .. descriptor.argDescriptorCount) {
        if (descriptor.argDescriptorList[i].name == name)
            return descriptor.argDescriptorList[i]
    }

    return OpNamespace.ArgDescriptor.getDefaultInstance()
}


package org.nd4j.codegen.ir

import org.nd4j.gen.OpDeclarationDescriptor
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.CustomOp
import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.buffer.DataType


interface IRTensor<TENSOR_TYPE,DATA_TYPE> {
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

interface IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE> {

    fun name(): String

    fun floatValue(): Float

    fun listFloatValue(): List<Float>

    fun tensorValue(): IRTensor<TENSOR_TYPE,DATA_TYPE>

    fun listTensorValue(): List<IRTensor<TENSOR_TYPE,DATA_TYPE>>

    fun intValue(): Long

    fun listIntValue(): List<Long>

    fun boolValue(): Boolean

    fun listBoolValue(): List<Boolean>

    fun attributeValueType(): AttributeValueType

    fun internalAttributeDef(): ATTRIBUTE_TYPE

    fun internalAttributeValue(): ATTRIBUTE_VALUE_TYPE
}



interface MappingProcess<NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE> {
    fun opName(): String

    fun frameworkVersion(): String

    fun inputFramework(): String

    fun rules(): List<MappingRule<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>>


    fun applyProcess(inputNode: IRNode<NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): OpDeclarationDescriptor

    fun applyProcessReverse(input: OpDeclarationDescriptor): IRNode<NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>

    fun createDescriptor(argDescriptors: List<OpNamespace.ArgDescriptor>): OpDeclarationDescriptor
}

enum class MappingRuleType {
    ATTRIBUTE,
    INPUT_TENSOR,
    OUTPUT_TENSOR
}

interface MappingRule<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE> {
    fun name(): String

    /**
     * Convert 1 or more attributes in to a list of {@link ArgDescriptor}
     */
    fun convert(): List<OpNamespace.ArgDescriptor>

    fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>>

    fun convertInputsReverse(toReverse: List<OpNamespace.ArgDescriptor>): List<TENSOR_TYPE>

    fun inputArgumentMapping(): Map<String,String>

    fun inputAttributeMapping(): Map<String,String>

    fun opDescriptor(): OpNamespace.OpDescriptor

    fun inputTensorsToConvert(): Map<String, TENSOR_TYPE>

    fun getInputTensor(key: String): TENSOR_TYPE

    fun inputAttributeDefsToConvert(): Map<String, ATTRIBUTE_TYPE>

    fun getInputAttribute(input: String): ATTRIBUTE_TYPE

    fun inputAttributeValuesToConvert(): Map<String, ATTRIBUTE_VALUE_TYPE>

    fun getInputAttributeValue(input:String): ATTRIBUTE_VALUE_TYPE

    fun mappingType(): MappingRuleType
}



interface IRNode<NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE> {
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
    fun attributeMap(): Map<String,IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>>
    fun getAttribute(inputName: String): IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>
    fun hasAttribute(inputName: String): Boolean

    fun internalValue(): NODE_TYPE
}

interface IRArgDef<T,DATA_TYPE> {
    fun name(): String

    fun description(): String

    fun dataType(): IRDataType<DATA_TYPE>

    fun internalValue(): T

    fun indexOf(): Integer
}

interface IROpDef<T,TENSOR_TYPE,ARG_DEF_TYPE,DATA_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE> {
    fun opName(): String

    fun internalValue(): T

    fun inputArgs(): List<IRArgDef<ARG_DEF_TYPE,DATA_TYPE>>

    fun outputArgs(): List<IRArgDef<ARG_DEF_TYPE,DATA_TYPE>>

    fun attributes(): List<IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>>

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




interface Mapper<NODE_TYPE,ATTR_DEF_TYPE,ATTR_VALUE_TYPE,OP_DEF_TYPE,DATATYPE_TYPE> {


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

    fun mapAttr(input: ATTR_DEF_TYPE,inputValue: ATTR_VALUE_TYPE): OpNamespace.ArgDescriptor

    fun typeFor(input: ATTR_DEF_TYPE): OpNamespace.ArgDescriptor.ArgType

    fun nd4jOpDefList(): OpNamespace.OpDescriptorList

    fun dataTypesForArgument(input: OP_DEF_TYPE,argName: String): List<DataType>

    fun typeFor(tensorflowType: DATATYPE_TYPE): DataType
}

fun createMapperDeclaration(input: OpDeclarationDescriptor): MapperNamespace.MapperDeclaration {
    TODO("Implement")
}



package org.nd4j.codegen.ir

import org.nd4j.gen.OpDeclarationDescriptor
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.OpNamespace.ArgDescriptor.ArgType
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.CustomOp
import org.tensorflow.framework.*
import org.tensorflow.framework.OpDef.AttrDef
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import org.nd4j.linalg.factory.Nd4j

class TensorflowIRTensor(input: TensorProto): IRTensor<TensorProto,org.tensorflow.framework.DataType> {

    val tensor = input


    override fun shape(): List<Long> {
        return  tensor.tensorShape.dimList.map { it.size }

    }

    override fun stride(): List<Long> {
        return Nd4j.getStrides(shape().toTypedArray().toLongArray(),'c').asList()
    }

    override fun dataType(): IRDataType<org.tensorflow.framework.DataType> {
        return TensorflowDataType(tensor.dtype)
    }

}

class TensorflowDataType(inputDataType: org.tensorflow.framework.DataType): IRDataType<org.tensorflow.framework.DataType> {
    val dataType = inputDataType

    override fun convertToDataType(input: org.tensorflow.framework.DataType): IRDataTypeValue {
        when(input) {
            org.tensorflow.framework.DataType.DT_BOOL -> return IRDataTypeValue.DT_BOOL
            org.tensorflow.framework.DataType.DT_BFLOAT16 -> return IRDataTypeValue.DT_BFLOAT16
            org.tensorflow.framework.DataType.DT_COMPLEX128 -> return IRDataTypeValue.DT_COMPLEX128
            org.tensorflow.framework.DataType.DT_COMPLEX64 -> return IRDataTypeValue.DT_COMPLEX64
            org.tensorflow.framework.DataType.DT_DOUBLE -> return IRDataTypeValue.DT_DOUBLE
            org.tensorflow.framework.DataType.DT_FLOAT -> return IRDataTypeValue.DT_FLOAT
            org.tensorflow.framework.DataType.DT_HALF -> return IRDataTypeValue.DT_HALF
            org.tensorflow.framework.DataType.DT_INT16 -> return IRDataTypeValue.DT_INT16
            org.tensorflow.framework.DataType.DT_INT32 -> return IRDataTypeValue.DT_INT32
            org.tensorflow.framework.DataType.DT_INT64 -> return IRDataTypeValue.DT_INT64
            org.tensorflow.framework.DataType.DT_QINT8 -> return IRDataTypeValue.DT_QINT8
            org.tensorflow.framework.DataType.DT_QINT16 -> return IRDataTypeValue.DT_QINT16
            org.tensorflow.framework.DataType.DT_QINT32 -> return IRDataTypeValue.DT_QINT32
            org.tensorflow.framework.DataType.DT_STRING -> return IRDataTypeValue.DT_STRING
            org.tensorflow.framework.DataType.DT_UINT16 -> return IRDataTypeValue.DT_UINT16
            org.tensorflow.framework.DataType.DT_UINT32 -> return IRDataTypeValue.DT_UINT32
            org.tensorflow.framework.DataType.DT_UINT64 -> return IRDataTypeValue.DT_UINT64

        }

        return IRDataTypeValue.DT_INVALID
    }

    override fun dataType(): IRDataTypeValue {
        return convertToDataType(this.dataType)
    }

    override fun internalValue(): org.tensorflow.framework.DataType {
        return this.dataType
    }

}

fun attrDefaultValue(): IRAttribute<AttrDef,AttrValue,TensorProto,org.tensorflow.framework.DataType> {
    return TensorflowIRAttr(AttrDef.getDefaultInstance(), AttrValue.getDefaultInstance())
}

class TensorflowIRAttr(inputAttributeDef: AttrDef,inputAttributeValue: AttrValue): IRAttribute<AttrDef,AttrValue,TensorProto,org.tensorflow.framework.DataType> {

    private val attributeDef = inputAttributeDef
    private val attributeValue = inputAttributeValue

    override fun name(): String {
        return attributeDef.name
    }

    override fun floatValue(): Float {
        return attributeValue.f
    }

    override fun listFloatValue(): List<Float> {
        return attributeValue.list.fList
    }


    override fun intValue(): Long {
        return attributeValue.i
    }

    override fun listIntValue(): List<Long> {
        return attributeValue.list.iList
    }

    override fun boolValue(): Boolean {
        return attributeValue.b
    }

    override fun listBoolValue(): List<Boolean> {
        return attributeValue.list.bList
    }

    override fun attributeValueType(): AttributeValueType {
        when(attributeDef.type) {
            "string" -> return AttributeValueType.STRING
            "list(string)" -> return AttributeValueType.LIST_STRING
            "int" -> return AttributeValueType.INT
            "list(int)" -> return AttributeValueType.LIST_INT
            "float" -> return AttributeValueType.FLOAT
            "list(float)" -> return AttributeValueType.LIST_FLOAT
            "tensor" -> return AttributeValueType.TENSOR
            "list(tensor)" -> return AttributeValueType.LIST_TENSOR
        }

        return AttributeValueType.INVALID
    }



    override fun internalAttributeDef(): AttrDef {
        return attributeDef
    }

    override fun internalAttributeValue(): AttrValue {
        return attributeValue
    }

    override fun listTensorValue(): List<IRTensor<TensorProto, org.tensorflow.framework.DataType>> {
        return attributeValue.list.tensorList.map {
            input -> TensorflowIRTensor(input)
        }
    }

    override fun tensorValue(): IRTensor<TensorProto, org.tensorflow.framework.DataType> {
        return TensorflowIRTensor(attributeValue.tensor)
    }

}

class TensorflowIRArgDef(input: OpDef.ArgDef): IRArgDef<OpDef.ArgDef,org.tensorflow.framework.DataType> {
    val argDefValue = input
    override fun dataType(): IRDataType<org.tensorflow.framework.DataType> {
        return TensorflowIRArgDef(argDefValue).dataType()
    }

    override fun name(): String {
        return argDefValue.name
    }

    override fun description(): String {
        return argDefValue.description
    }

    override fun internalValue(): OpDef.ArgDef {
        return argDefValue
    }

    override fun indexOf(): Integer {
        TODO("Not yet implemented")
    }

}

class TensorflowIROp(input: OpDef): IROpDef<OpDef,TensorProto,OpDef.ArgDef,org.tensorflow.framework.DataType,AttrDef,AttrValue> {

    val opDef = input

    override fun attributes(): List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> {
        return opDef.attrList.map {
            TensorflowIRAttr(it, AttrValue.getDefaultInstance())
        }
    }

    override fun opName(): String {
        return opDef.name
    }

    override fun internalValue(): OpDef {
        return opDef
    }

    override fun inputArgs(): List<IRArgDef<OpDef.ArgDef, org.tensorflow.framework.DataType>> {
        return opDef.inputArgList.map {
            TensorflowIRArgDef(it)
        }
    }

    override fun outputArgs(): List<IRArgDef<OpDef.ArgDef, org.tensorflow.framework.DataType>> {
        return opDef.outputArgList.map {
            TensorflowIRArgDef(it)
        }
    }

}

class TensorflowIRNode<NodeDef,TensorflowIRTensor,TensorflowIRAttr>(inputNode: org.tensorflow.framework.NodeDef,inputOpDef: OpDef): IRNode<org.tensorflow.framework.NodeDef,TensorProto,AttrDef,AttrValue,org.tensorflow.framework.DataType> {

    private val nodeDef = inputNode
    private val opDef = inputOpDef
    private val attrDefsMap = attrDefsByName(inputOpDef.attrList)
    private val attrMap: Map<String, IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> = initAttrMapFromNode(inputNode)

    init {

    }

    private fun attrDefsByName(input: List<AttrDef>): Map<String,AttrDef> {
        val ret = HashMap<String,AttrDef>()
        input.forEach {
            ret[it.name] = it
        }
        return ret
    }

    private fun initAttrMapFromNode(input: org.tensorflow.framework.NodeDef): Map<String, IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> {
        val ret = HashMap<String,IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>>()
        input.attrMap.forEach { (key, value) ->
            ret[key] =  TensorflowIRAttr(attrDefsMap.getOrDefault(key, AttrDef.getDefaultInstance()),value)
        }

        return ret
    }

    override fun opName(): String {
        return nodeDef.op
    }

    override fun nodeName(): String {
        return nodeDef.name
    }

    override fun inputAt(index: Int): String {
        return nodeDef.getInput(index)
    }

    override fun outputAt(index: Int): String {
        return opDef.getOutputArg(index).name
    }



    override fun hasAttribute(inputName: String): Boolean {
        return nodeDef.containsAttr(inputName)
    }

    override fun attributeMap(): Map<String, IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> {
        return attrMap
    }

    override fun createInputsFrom(inputData: List<TensorProto>): List<IRTensor<TensorProto, org.tensorflow.framework.DataType>> {
        return inputData.map { org.nd4j.codegen.ir.TensorflowIRTensor(it) }
    }

    override fun createOutputsFrom(inputValues: List<TensorProto>): List<IRTensor<TensorProto, org.tensorflow.framework.DataType>> {
        return inputValues.map { org.nd4j.codegen.ir.TensorflowIRTensor(it) }
    }

    override fun getAttribute(inputName: String): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
        return attrMap.getOrDefault(inputName,attrDefaultValue())
    }

    override fun internalValue(): org.tensorflow.framework.NodeDef {
        return nodeDef
    }

}

class TensorflowMapper: Mapper<NodeDef,AttrDef,AttrValue,OpDef,org.tensorflow.framework.DataType> {
    override fun createOpFrom(input: OpDef): CustomOp {
        TODO("Not yet implemented")
    }

    override fun mapNameOp(input: String): String {
        TODO("Not yet implemented")
    }

    override fun map(input: NodeDef): INDArray {
        TODO("Not yet implemented")
    }

    override fun map(input: OpDef): OpDeclarationDescriptor {
        TODO("Not yet implemented")
    }

    override fun opDefList(): List<OpDef> {
        TODO("Not yet implemented")
    }

    override fun mapAttr(input: AttrDef, inputValue: AttrValue): OpNamespace.ArgDescriptor {
        TODO("Not yet implemented")
    }

    override fun typeFor(input: AttrDef): ArgType {
        TODO("Not yet implemented")
    }

    override fun nd4jOpDefList(): OpNamespace.OpDescriptorList {
        TODO("Not yet implemented")
    }

    override fun dataTypesForArgument(input: OpDef, argName: String): List<DataType> {
        TODO("Not yet implemented")
    }

    override fun typeFor(tensorflowType: org.tensorflow.framework.DataType): DataType {
        TODO("Not yet implemented")
    }

}
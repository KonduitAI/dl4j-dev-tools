package org.nd4j.codegen.ir

import org.nd4j.gen.OpDeclarationDescriptor
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.OpNamespace.ArgDescriptor.ArgType
import org.nd4j.ir.TensorNamespace
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

    override fun toArgTensor(): TensorNamespace.TensorProto {
        val builder = TensorNamespace.TensorProto.newBuilder()
                .setDataLocation(TensorNamespace.TensorProto.DataLocation.DEFAULT)

        for(i in 0 .. tensor.tensorShape.dimCount) {
            builder.addDims(tensor.tensorShape.getDim(i).size)
        }

        when(tensor.dtype) {
            org.tensorflow.framework.DataType.DT_UINT64 -> builder.dataType = TensorNamespace.DataType.UINT64.ordinal
            org.tensorflow.framework.DataType.DT_UINT32 -> builder.dataType = TensorNamespace.DataType.UINT32.ordinal
            org.tensorflow.framework.DataType.DT_UINT16 -> builder.dataType = TensorNamespace.DataType.UINT16.ordinal
            org.tensorflow.framework.DataType.DT_HALF -> builder.dataType = TensorNamespace.DataType.FLOAT16.ordinal
            org.tensorflow.framework.DataType.DT_STRING -> builder.dataType = TensorNamespace.DataType.STRING.ordinal
            org.tensorflow.framework.DataType.DT_FLOAT -> builder.dataType  = TensorNamespace.DataType.FLOAT.ordinal
            org.tensorflow.framework.DataType.DT_DOUBLE -> builder.dataType = TensorNamespace.DataType.DOUBLE.ordinal
            org.tensorflow.framework.DataType.DT_BOOL -> builder.dataType = TensorNamespace.DataType.BOOL.ordinal
            org.tensorflow.framework.DataType.DT_INT64 -> builder.dataType = TensorNamespace.DataType.INT64.ordinal
            org.tensorflow.framework.DataType.DT_INT32 -> builder.dataType = TensorNamespace.DataType.INT32.ordinal
            org.tensorflow.framework.DataType.DT_INT16 -> builder.dataType = TensorNamespace.DataType.INT16.ordinal
            org.tensorflow.framework.DataType.DT_BFLOAT16 -> builder.dataType = TensorNamespace.DataType.BFLOAT16.ordinal
            org.tensorflow.framework.DataType.DT_COMPLEX64 -> builder.dataType = TensorNamespace.DataType.COMPLEX64.ordinal
            org.tensorflow.framework.DataType.DT_COMPLEX128 -> builder.dataType = TensorNamespace.DataType.COMPLEX128.ordinal
            org.tensorflow.framework.DataType.UNRECOGNIZED -> builder.dataType = TensorNamespace.DataType.UNRECOGNIZED.ordinal

        }


        if(tensor.doubleValList != null) {
            builder.addAllDoubleData(tensor.doubleValList)
        }

        if(tensor.stringValList != null) {
            builder.addAllStringData(tensor.stringValList)
        }

        if(tensor.floatValList != null) {
            builder.addAllFloatData(tensor.floatValList)
        }

        if(tensor.uint32ValList != null) {
            builder.addAllInt32Data(tensor.uint32ValList)
        }

        if(tensor.uint64ValList != null) {
            builder.addAllInt64Data(tensor.uint64ValList)
        }

        if(tensor.int64ValList != null) {
            builder.addAllInt64Data(tensor.int64ValList)
        }

        if(tensor.tensorContent != null) {
            builder.rawData = tensor.tensorContent
        }

        builder.dataType = tensor.dtype.ordinal

        return builder.build()
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

fun argDescriptor(name: String,inputBoolean: Boolean,inputString: String): OpNamespace.ArgDescriptor {
    return OpNamespace.ArgDescriptor.newBuilder()
            .setName(name)
            .build()
}

fun convertTensorflowTensorToArgDescriptorTensor(input: TensorProto): TensorflowIRTensor {
    return TensorflowIRTensor(input)
}

//TODO: How to handle base classes? Parameterized constructors?
//TODO: Framework specific mappers? How to share logic across different frameworks? Are generics enough?
//Init logic for mapping rules should ideally be generic where possible
class NDArrayMappingRule(opDescriptor: OpNamespace.OpDescriptor,inputTensors: Map<String, TensorProto>): MappingRule<AttrDef,AttrValue,TensorProto,org.tensorflow.framework.DataType> {

    private val opDescriptor = opDescriptor
    private val inputTensors = inputTensors

    override fun name(): String {
        return "ndarraymapping"
    }

    override fun convert(): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        val mappingsToPerform = inputArgumentMapping()
        for(i in 0.. opDescriptor.argDescriptorCount) {
            if(mappingsToPerform.containsKey(opDescriptor.getArgDescriptor(i).name)) {
                val outputName = mappingsToPerform[mappingsToPerform[opDescriptor.getArgDescriptor(i).name]]
                val builder = OpNamespace.ArgDescriptor.newBuilder()
                builder.argType = opDescriptor.argDescriptorList[i].argType
                builder.name = outputName
                require(opDescriptor.argDescriptorList[i].argType == ArgType.INPUT_TENSOR) {"Input type must be INPUT_TENSOR"}
                builder.argIndex = opDescriptor.argDescriptorList[i].argIndex
                val tensorToConvert = getInputTensor(opDescriptor.getArgDescriptor(i).name)
                builder.inputValue = TensorflowIRTensor(tensorToConvert).toArgTensor()
                ret.add(builder.build())
            }

        }

        return ret
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> {
        return emptyList()
    }

    override fun convertInputsReverse(toReverse: List<OpNamespace.ArgDescriptor>): List<TensorProto> {
        for(argument in toReverse) {
            require(argument.argType == ArgType.INPUT_TENSOR) {"Type to reverse must be an input tensor."}
        }
        TODO("Not yet implemented")
    }

    override fun inputArgumentMapping(): Map<String, String> {
        return hashMapOf(
                "input" to "input"
        )
    }

    override fun inputAttributeMapping(): Map<String, String> {
        return emptyMap()
    }

    override fun opDescriptor(): OpNamespace.OpDescriptor {
        return opDescriptor
    }

    override fun inputTensorsToConvert(): Map<String, TensorProto> {
        return inputTensors
    }

    override fun inputAttributeDefsToConvert(): Map<String, AttrDef> {
        return emptyMap()
    }

    override fun inputAttributeValuesToConvert(): Map<String, AttrValue> {
        return emptyMap()
    }

    override fun mappingType(): MappingRuleType {
        return MappingRuleType.INPUT_TENSOR
    }

    override fun getInputTensor(key: String): TensorProto {
        return inputTensorsToConvert().getOrDefault(key, TensorProto.getDefaultInstance())
    }

    override fun getInputAttribute(input: String): AttrDef {
        return inputAttributeDefsToConvert().getOrDefault(input,AttrDef.getDefaultInstance())
    }

    override fun getInputAttributeValue(input: String): AttrValue {
        return inputAttributeValuesToConvert().getOrDefault(input,AttrValue.getDefaultInstance())
    }


}

class AbsMappingProcess: MappingProcess<NodeDef,TensorProto,AttrDef,AttrValue,org.tensorflow.framework.DataType> {
    override fun opName(): String {
        return "abs"
    }

    override fun frameworkVersion(): String {
        return "1.0"
    }

    override fun inputFramework(): String {
        return "tensorflow"
    }

    override fun rules(): List<MappingRule<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> {
        TODO("Not yet implemented")
    }

    override fun applyProcess(inputNode: IRNode<NodeDef, TensorProto, AttrDef, AttrValue, org.tensorflow.framework.DataType>): OpDeclarationDescriptor {
        val builder = OpDeclarationDescriptor.builder()
        builder.name(opName())
        val attributeList = inputNode.attributeMap().entries.toList().map { it.value }

        for(rule in rules()) {
            rule.convert()
        }

        return builder.build()
    }

    override fun applyProcessReverse(input: OpDeclarationDescriptor): IRNode<NodeDef, TensorProto, AttrDef, AttrValue, org.tensorflow.framework.DataType> {
        for(rule in rules().reversed()) {

        }
        TODO("Not yet implemented")
    }

    override fun createDescriptor(argDescriptors: List<OpNamespace.ArgDescriptor>): OpDeclarationDescriptor {
        TODO("Not yet implemented")
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

    override fun map(input: OpDef): OpNamespace.OpDescriptor {
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
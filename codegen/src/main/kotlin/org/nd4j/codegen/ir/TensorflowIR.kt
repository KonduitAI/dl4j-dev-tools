package org.nd4j.codegen.ir

import org.apache.commons.io.IOUtils
import org.nd4j.common.io.ClassPathResource
import org.nd4j.gen.OpDeclarationDescriptor
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.OpNamespace.ArgDescriptor.ArgType
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.CustomOp
import org.tensorflow.framework.*
import org.tensorflow.framework.OpDef.AttrDef
import kotlin.collections.HashMap
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.shade.protobuf.TextFormat
import java.nio.charset.Charset

fun loadTensorflowOps(): OpList {
    val string = IOUtils.toString(ClassPathResource("ops.proto").inputStream, Charset.defaultCharset())
    val tfListBuilder = OpList.newBuilder()
    TextFormat.merge(string,tfListBuilder)
    return tfListBuilder.build()
}

class TensorflowIR: IR<NodeDef,TensorProto,org.tensorflow.framework.DataType,AttrDef,AttrValue,OpList> {

    private val nd4jOpList = loadNd4jOpDescriptors()
    private val tensorflowOpList = loadTensorflowOps()

    override fun createNode(name: String, inputs: List<String>, opName: String, nodeAttributes: Map<String, AttrValue>): NodeDef {
        val builder = NodeDef.newBuilder()
        builder.name = name
        for(i in 0 .. inputs.size) {
            builder.addInput(inputs[i])
        }

        val op = tensorflowOpList.opList.filter { input -> input.name == opName }
        require(op.size == 1) { "Op name $opName illegal. Name must be unique and defined within tensorflow." }
        builder.op = opName
        builder.putAllAttr(nodeAttributes)
        return builder.build()
    }

    override fun createIRNode(name: String, inputNode: NodeDef): IRNode<NodeDef, TensorProto, AttrDef, AttrValue, org.tensorflow.framework.DataType> {
        val opDef = tensorflowOpList.opList.filter { input -> input.name == inputNode.op }[0]
        return TensorflowIRNode(inputNode,opDef)
    }

    override fun <T> createTensor(name: String, dataType: org.tensorflow.framework.DataType, shape: List<Long>, inputData: T): TensorProto {
        val tensorProtoBuilder = TensorProto.newBuilder()
        tensorProtoBuilder.dtype = dataType
        val tensorShapeProtoBuilder = TensorShapeProto.newBuilder()
        for(i in 0 .. shape.size) {
            val dim = TensorShapeProto.Dim.newBuilder()
            dim.size = shape[i]
            dim.name = i.toString()
            tensorShapeProtoBuilder.addDim(dim.build())
        }

        tensorProtoBuilder.tensorShape = tensorShapeProtoBuilder.build()

        return tensorProtoBuilder.build()
    }

    override fun createIRTensor(inputTensor: TensorProto): IRTensor<TensorProto, org.tensorflow.framework.DataType> {
        return TensorflowIRTensor(inputTensor)
    }

    override fun createAttributeValue(name: String, attributeValue: AttributeValueType): AttrValue {
        val attrBuilder = AttrValue.newBuilder()
        when(attributeValue) {
            AttributeValueType.INT -> attrBuilder.type = org.tensorflow.framework.DataType.DT_INT64
            AttributeValueType.STRING -> attrBuilder.type = org.tensorflow.framework.DataType.DT_STRING
            AttributeValueType.FLOAT -> attrBuilder.type = org.tensorflow.framework.DataType.DT_FLOAT
            AttributeValueType.BOOL -> attrBuilder.type = org.tensorflow.framework.DataType.DT_BOOL
             else -> {
                 attrBuilder.type = org.tensorflow.framework.DataType.DT_VARIANT
             }

        }
        return attrBuilder.build()
    }

    override fun createAttributeDef(name: String, description: String, type: String): AttrDef {
        val attrDef = AttrDef.newBuilder()
        attrDef.name = name
        attrDef.description = description
        attrDef.type = type
        return attrDef.build()
    }

    override fun createIRAttribute(attributeDef: AttrDef, attributeValue: AttrValue): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
        return TensorflowIRAttr(attributeDef,attributeValue)
    }

    override fun createDataType(name: String, dataType: TensorNamespace.DataType): org.tensorflow.framework.DataType {
        when(dataType) {
            TensorNamespace.DataType.UINT64 -> return org.tensorflow.framework.DataType.DT_UINT64
            TensorNamespace.DataType.UINT32 -> return org.tensorflow.framework.DataType.DT_UINT32
            TensorNamespace.DataType.UINT16 -> return org.tensorflow.framework.DataType.DT_UINT16
            TensorNamespace.DataType.FLOAT16 -> return org.tensorflow.framework.DataType.DT_HALF
            TensorNamespace.DataType.STRING -> return org.tensorflow.framework.DataType.DT_STRING
            TensorNamespace.DataType.FLOAT -> return org.tensorflow.framework.DataType.DT_FLOAT
            TensorNamespace.DataType.DOUBLE -> return org.tensorflow.framework.DataType.DT_DOUBLE
            TensorNamespace.DataType.BOOL -> return org.tensorflow.framework.DataType.DT_BOOL
            TensorNamespace.DataType.INT64 -> return org.tensorflow.framework.DataType.DT_INT64
            TensorNamespace.DataType.INT32 -> return org.tensorflow.framework.DataType.DT_INT32
            TensorNamespace.DataType.INT16 -> return org.tensorflow.framework.DataType.DT_INT16
            TensorNamespace.DataType.BFLOAT16 -> return org.tensorflow.framework.DataType.DT_BFLOAT16
            TensorNamespace.DataType.COMPLEX64 -> return org.tensorflow.framework.DataType.DT_COMPLEX64
            TensorNamespace.DataType.COMPLEX128 ->  return org.tensorflow.framework.DataType.DT_COMPLEX128
            else -> {
                return org.tensorflow.framework.DataType.UNRECOGNIZED
            }

        }

    }

    override fun createIRDataType(dataType: org.tensorflow.framework.DataType): IRDataType<org.tensorflow.framework.DataType> {
        return TensorflowIRDataType(dataType)
    }

    override fun nd4jListOps(): OpNamespace.OpDescriptorList {
        return nd4jOpList
    }

    override fun inputFrameworkOpDefs(): OpList {
        return tensorflowOpList
    }

}


class TensorflowIRTensor(input: TensorProto): IRTensor<TensorProto,org.tensorflow.framework.DataType> {

    val tensor = input


    override fun shape(): List<Long> {
        return  tensor.tensorShape.dimList.map { it.size }

    }

    override fun stride(): List<Long> {
        return Nd4j.getStrides(shape().toTypedArray().toLongArray(),'c').asList()
    }

    override fun dataType(): IRDataType<org.tensorflow.framework.DataType> {
        return TensorflowIRDataType(tensor.dtype)
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

    override fun rawValue(): TensorProto {
        return tensor
    }


}

class TensorflowIRDataType(inputDataType: org.tensorflow.framework.DataType): IRDataType<org.tensorflow.framework.DataType> {
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

    override fun stringValue(): String {
        return attributeValue.s.toStringUtf8()
    }

    override fun listStringValue(): List<String> {
        return attributeValue.list.sList.map { it.toStringUtf8() }
    }

}

class TensorflowIRArgDef(input: OpDef.ArgDef): IRArgDef<OpDef.ArgDef,org.tensorflow.framework.DataType> {
    private val argDefValue = input

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

class TensorflowIRNode(inputNode: NodeDef,inputOpDef: OpDef): IRNode<NodeDef,TensorProto,AttrDef,AttrValue,org.tensorflow.framework.DataType> {

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

    private fun initAttrMapFromNode(input: NodeDef): Map<String, IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> {
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

    override fun internalValue(): NodeDef {
        return nodeDef
    }

}

fun argDescriptor(name: String,inputBoolean: Boolean,inputString: String): OpNamespace.ArgDescriptor {
    return OpNamespace.ArgDescriptor.newBuilder()
            .setName(name)
            .build()
}




class NDArrayMappingRule(opDescriptor: OpNamespace.OpDescriptor, inputTensors: Map<String, TensorProto>,
                         mappingNamesToPerform: Map<String,String>,
                         transformerArgs: Map<String,
                                 List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>>>):
        BaseNDArrayMappingRule<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>(opDescriptor, inputTensors,mappingNamesToPerform, transformerArgs) {
    override fun getInputTensor(key: String): TensorProto {
        return inputTensorsToConvert().getOrDefault(key, TensorProto.getDefaultInstance())
    }

    override fun getInputAttribute(input: String): AttrDef {
        return inputAttributeDefsToConvert().getOrDefault(input,AttrDef.getDefaultInstance())
    }

    override fun getInputAttributeValue(input: String): AttrValue {
        return inputAttributeValuesToConvert().getOrDefault(input,AttrValue.getDefaultInstance())
    }

    override fun createTensorProto(input: TensorProto): TensorNamespace.TensorProto {
        return TensorflowIRTensor(input).toArgTensor()
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
            rule.convertInput()
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


class TensorflowNDArrayMappingRule(opDescriptor: OpNamespace.OpDescriptor, inputTensors: Map<String, TensorProto>, mappingNamesToPerform: Map<String, String>,
                                   transformerArgs: Map<String, List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>>>)
    : BaseNDArrayMappingRule<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>(opDescriptor, inputTensors, mappingNamesToPerform, transformerArgs) {
    override fun createTensorProto(input: TensorProto): TensorNamespace.TensorProto {
        return TensorflowIRTensor(input).toArgTensor()
    }
}

class TensorflowStringEqualsAdapterRule(opDescriptor: OpNamespace.OpDescriptor, mappingNamesToPerform: Map<String, String>, inputAttributeDef: AttrDef, inputAttributeValue: AttrValue, transformerArgs: Map<String, List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>>>) : StringEqualsAdapterRule<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>(opDescriptor, mappingNamesToPerform, inputAttributeDef, inputAttributeValue, transformerArgs) {

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> {
        TODO("Not yet implemented")
    }

    override fun convertInput(): List<OpNamespace.ArgDescriptor> {
        TODO("Not yet implemented")
    }

    override fun convertInputsReverse(toReverse: List<OpNamespace.ArgDescriptor>): List<TensorProto> {
        TODO("Not yet implemented")
    }

    override fun inputTensorsToConvert(): Map<String, TensorProto> {
        TODO("Not yet implemented")
    }

    override fun getInputTensor(key: String): TensorProto {
        TODO("Not yet implemented")
    }

    override fun inputAttributeDefsToConvert(): Map<String, AttrDef> {
        TODO("Not yet implemented")
    }

    override fun getInputAttribute(input: String): AttrDef {
        TODO("Not yet implemented")
    }

    override fun inputAttributeValuesToConvert(): Map<String, AttrValue> {
        TODO("Not yet implemented")
    }

    override fun getInputAttributeValue(input: String): AttrValue {
        TODO("Not yet implemented")
    }

    override fun createIRAttribute(name: String, attrDef: AttrDef, attributeValueType: AttrValue): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
        TODO("Not yet implemented")
    }

    override fun typeForAttribute(name: String): AttributeValueType {
        TODO("Not yet implemented")
    }

}

class TensorflowSizeThresholdIntArrayIntIndexRule(opDescriptor: OpNamespace.OpDescriptor, mappingNamesToPerform: Map<String, String>, inputAttributeDef: AttrDef, inputAttributeValue: AttrValue, transformerArgs: Map<String, List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>>>) : SizeThresholdIntArrayIntIndexRule<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>(opDescriptor, mappingNamesToPerform, inputAttributeDef, inputAttributeValue, transformerArgs) {
    override fun createIRAttribute(name: String, attrDef: AttrDef, attributeValueType: AttrValue): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
        TODO("Not yet implemented")
    }

    override fun convertInput(): List<OpNamespace.ArgDescriptor> {
        TODO("Not yet implemented")
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> {
        TODO("Not yet implemented")
    }

    override fun convertInputsReverse(toReverse: List<OpNamespace.ArgDescriptor>): List<TensorProto> {
        TODO("Not yet implemented")
    }

    override fun inputTensorsToConvert(): Map<String, TensorProto> {
        TODO("Not yet implemented")
    }

    override fun getInputTensor(key: String): TensorProto {
        TODO("Not yet implemented")
    }

    override fun inputAttributeDefsToConvert(): Map<String, AttrDef> {
        TODO("Not yet implemented")
    }

    override fun getInputAttribute(input: String): AttrDef {
        TODO("Not yet implemented")
    }

    override fun inputAttributeValuesToConvert(): Map<String, AttrValue> {
        TODO("Not yet implemented")
    }

    override fun getInputAttributeValue(input: String): AttrValue {
        TODO("Not yet implemented")
    }

    override fun typeForAttribute(name: String): AttributeValueType {
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
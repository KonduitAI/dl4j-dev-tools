package org.nd4j.codegen.ir.tensorflow

import org.apache.commons.io.IOUtils
import org.nd4j.codegen.ir.*
import org.nd4j.common.io.ClassPathResource
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

val tensorflowOps = loadTensorflowOps()

fun lookupOpDef(name: String): OpDef {
    return tensorflowOps.opList.filter { it.name == name }[0]
}


class TensorflowIR: IR<NodeDef, TensorProto, org.tensorflow.framework.DataType, AttrDef, AttrValue, OpList> {

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


class TensorflowIRTensor(input: TensorProto): IRTensor<TensorProto, org.tensorflow.framework.DataType> {

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

fun attrDefaultValue(): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
    return TensorflowIRAttr(AttrDef.getDefaultInstance(), AttrValue.getDefaultInstance())
}

class TensorflowIRAttr(inputAttributeDef: AttrDef,inputAttributeValue: AttrValue): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {

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

class TensorflowIRArgDef(input: OpDef.ArgDef): IRArgDef<OpDef.ArgDef, org.tensorflow.framework.DataType> {
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

class TensorflowIROp(input: OpDef): IROpDef<OpDef, TensorProto, OpDef.ArgDef, org.tensorflow.framework.DataType, AttrDef, AttrValue> {

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

class TensorflowIRNode(inputNode: NodeDef,inputOpDef: OpDef): IRNode<NodeDef, TensorProto, AttrDef, AttrValue, org.tensorflow.framework.DataType> {

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
        val ret = HashMap<String, IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>>()
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
        return inputData.map { TensorflowIRTensor(it) }
    }

    override fun createOutputsFrom(inputValues: List<TensorProto>): List<IRTensor<TensorProto, org.tensorflow.framework.DataType>> {
        return inputValues.map { TensorflowIRTensor(it) }
    }

    override fun getAttribute(inputName: String): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
        return attrMap.getOrDefault(inputName, attrDefaultValue())
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




class NDArrayMappingRule(mappingNamesToPerform: Map<String,String>,
                         transformerArgs: Map<String,
                                 List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>>> = emptyMap()):
        BaseNDArrayMappingRule<OpDef,NodeDef,AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>(mappingNamesToPerform = mappingNamesToPerform,transformerArgs = transformerArgs) {



    override fun createTensorProto(input: TensorProto): TensorNamespace.TensorProto {
        return TensorflowIRTensor(input).toArgTensor()
    }
}

abstract class AbstractTensorflowMappingProcess(inputFramework: String = "tensorflow",
                                                frameworkVersion: String = "2.3",
                                                inputFrameworkOpName: String,
                                                opName: String,
                                                tensorMappingRules: List<out TensorMappingRule<OpDef, NodeDef, AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> = emptyList(),
                                                attributeMappingRules: List<out AttributeMappingRule<OpDef, NodeDef, AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> = emptyList())
    : AbstractMappingProcess<OpDef,NodeDef, TensorProto, AttrDef, AttrValue, org.tensorflow.framework.DataType>(
        inputFramework,
        frameworkVersion,
        inputFrameworkOpName,
        opName,
        tensorMappingRules,
        attributeMappingRules) {

    override fun lookupInputOpDef(opName: String): OpDef {
        return tensorflowOps.opList.filter { it.name == opName }.first()
    }
}



//val listOfRules =  listOf(NDArrayMappingRule("abs",mapOf("x" to "x", "y" to "y"))
class AbsMappingProcess: AbstractTensorflowMappingProcess(
        opName = "abs",
        inputFrameworkOpName = "Abs", tensorMappingRules =  listOf(NDArrayMappingRule(
        mappingNamesToPerform = mapOf("x" to "x", "y" to "y"))))


/**
 * opList {
name: "conv2d"
argDescriptor {
name: "sW"
argType: INT64
}
argDescriptor {
name: "dH"
argType: INT64
}
argDescriptor {
name: "wFormat"
argType: INT64
}
argDescriptor {
name: "pW"
argType: INT64
}
argDescriptor {
name: "isNCHW"
argType: INT64
}
argDescriptor {
name: "weights"
argType: INPUT_TENSOR
}
argDescriptor {
name: "kW"
argType: INT64
}
argDescriptor {
name: "output"
argType: OUTPUT_TENSOR
}
argDescriptor {
name: "input"
argType: INPUT_TENSOR
}
argDescriptor {
name: "dW"
argType: INT64
}
argDescriptor {
name: "sH"
argType: INT64
}
argDescriptor {
name: "bias"
argType: INPUT_TENSOR
}
argDescriptor {
name: "pH"
argType: INT64
}
argDescriptor {
name: "isSameMode"
argType: INT64
}
argDescriptor {
name: "kH"
argType: INT64
}
}
 */


/**
 * op {
name: "Conv2D"
input_arg {
name: "input"
type_attr: "T"
}
input_arg {
name: "filter"
type_attr: "T"
}
output_arg {
name: "output"
type_attr: "T"
}
attr {
name: "T"
type: "type"
allowed_values {
list {
type: DT_HALF
type: DT_BFLOAT16
type: DT_FLOAT
type: DT_DOUBLE
}
}
}
attr {
name: "strides"
type: "list(int)"
}
attr {
name: "use_cudnn_on_gpu"
type: "bool"
default_value {
b: true
}
}
attr {
name: "padding"
type: "string"
allowed_values {
list {
s: "SAME"
s: "VALID"
}
}
}
attr {
name: "data_format"
type: "string"
default_value {
s: "NHWC"
}
allowed_values {
list {
s: "NHWC"
s: "NCHW"
}
}
}
attr {
name: "dilations"
type: "list(int)"
default_value {
list {
i: 1
i: 1
i: 1
i: 1
}
}
}
}
 */

val tfIr = TensorflowIR()

class Conv2DMappingProcess: AbstractTensorflowMappingProcess(
        inputFramework = "tensorflow",
        inputFrameworkOpName = "Conv2D",
        opName = "conv2d",
        tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mapOf(
                "input" to "input","filter" to "weights"
        ),transformerArgs = emptyMap())),
        attributeMappingRules = listOf(
                TensorflowStringEqualsAdapterRule(
                        mappingNamesToPerform = mapOf("isNCHW" to "data_format"),
                        transformerArgs = mapOf("isNCHW" to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                            name = "data_format"
                            stringValue = "NCHW"
                        }.build()))
                ),
                TensorflowStringEqualsAdapterRule(
                        mappingNamesToPerform = mapOf("isSameMode" to "padding"),
                        transformerArgs = mapOf("isSameMode" to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                            name = "padding"
                            stringValue = "SAME"
                        }.build()))
                ),
                TensorflowConditionalFieldValueIntIndexArrayRule(
                        mappingNamesToPerform = mapOf("sH" to "strides"),
                        transformerArgs = mapOf("sH" to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                            name = "targetValue"
                            stringValue = "NCHW"
                        }.build(),
                                OpNamespace.ArgDescriptor.newBuilder().apply {
                                    name = "trueIndex"
                                    int32Value = 2
                                }.build(),
                                OpNamespace.ArgDescriptor.newBuilder().apply {
                                    name = "falseIndex"
                                    int32Value = 1
                                }.build()))
                ),
                TensorflowConditionalFieldValueIntIndexArrayRule(
                        mappingNamesToPerform = mapOf("sW" to "strides"),
                        transformerArgs = mapOf("sW" to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                            name = "targetValue"
                            stringValue = "NCHW"
                        }.build(),OpNamespace.ArgDescriptor.newBuilder().apply {
                            name = "trueIndex"
                            int32Value = 3
                        }.build(),
                                OpNamespace.ArgDescriptor.newBuilder().apply {
                                    name = "falseIndex"
                                    int32Value = 2
                                }.build()
                        ))
                ),
                TensorflowConditionalFieldValueIntIndexArrayRule(
                        mappingNamesToPerform = mapOf("dH" to "dilations"),
                        transformerArgs = mapOf("dH" to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                            name = "targetValue"
                            stringValue = "NCHW"
                        }.build(),OpNamespace.ArgDescriptor.newBuilder().apply {
                            name = "trueIndex"
                            int32Value = 2
                        }.build(),
                                OpNamespace.ArgDescriptor.newBuilder().apply {
                                    name = "falseIndex"
                                    int32Value = 1
                                }.build()))
                ),
                TensorflowConditionalFieldValueIntIndexArrayRule(
                        mappingNamesToPerform = mapOf("dW" to "dilations"),
                        transformerArgs = mapOf("dW" to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
                            name = "targetValue"
                            stringValue = "NCHW"
                        }.build(),OpNamespace.ArgDescriptor.newBuilder().apply {
                            name = "trueIndex"
                            int32Value = 3
                        }.build(),
                                OpNamespace.ArgDescriptor.newBuilder().apply {
                                    name = "falseIndex"
                                    int32Value = 2
                                }.build()))
                ))
)




class TensorflowConditionalFieldValueIntIndexArrayRule(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : ConditionalFieldValueIntIndexArrayRule<OpDef, NodeDef, AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>(mappingNamesToPerform, transformerArgs) {
    override fun createIRAttribute(name: String, attrDef: AttrDef, attributeValueType: AttrValue): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
        return TensorflowIRAttr(attrDef,attributeValueType)
    }

    override fun getIRAttribute(name: String, nodeWithValues: NodeDef): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
        return createIRAttribute(name,opDef!!.attrList.first { it.name == name },nodeWithValues.getAttrOrThrow(name))
    }

    override fun getAttributeDefFromName(inputName: String): AttrDef {
        return opDef!!.attrList.first { it.name == inputName }
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> {
        TODO("Not yet implemented")
    }

}

class TensorflowStringEqualsAdapterRule(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                        transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
        StringEqualsAdapterRule<OpDef,NodeDef,AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: AttrDef, attributeValueType: AttrValue): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
        return TensorflowIRAttr(attrDef,attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> {
        TODO("Not yet implemented")
    }

    override fun getAttributeDefFromName(inputName: String): AttrDef {
        return opDef!!.attrList.first { it.name == inputName }
    }

    override fun getIRAttribute(name: String, nodeWithValues: NodeDef): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
        return createIRAttribute(name,opDef!!.attrList.first { it.name == name },nodeWithValues.getAttrOrThrow(name))
    }
}

class TensorflowSizeThresholdIntArrayIntIndexRule(mappingNamesToPerform: Map<String, String>,
                                                  transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : SizeThresholdIntArrayIntIndexRule<OpDef, NodeDef, AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>(mappingNamesToPerform, transformerArgs) {
    override fun createIRAttribute(name: String, attrDef: AttrDef, attributeValueType: AttrValue): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
        return TensorflowIRAttr(attrDef,attributeValueType)
    }

    override fun getAttributeDefFromName(inputName: String): AttrDef {
        return opDef!!.attrList.first { it.name == inputName }
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>> {
        TODO("Not yet implemented")
    }

    override fun getIRAttribute(name: String, nodeWithValues: NodeDef): IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType> {
        return createIRAttribute(name,opDef!!.attrList.first { it.name == name },nodeWithValues.getAttrOrThrow(name))
    }

}



class TensorflowMapper: Mapper<NodeDef, AttrDef, AttrValue, OpDef, org.tensorflow.framework.DataType> {
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
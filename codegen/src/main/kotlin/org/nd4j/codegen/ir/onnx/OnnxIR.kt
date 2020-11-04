package org.nd4j.codegen.ir.onnx

import onnx.Onnx
import org.nd4j.codegen.ir.*
import org.nd4j.common.io.ClassPathResource
import org.nd4j.ir.TensorNamespace

import kotlin.collections.HashMap
import org.nd4j.linalg.factory.Nd4j

fun loadOnnxOps(): List<Onnx.NodeProto> {
    val graphProto = Onnx.GraphProto.parseFrom(ClassPathResource("onnx-op-defs.pb").inputStream)
    return graphProto.nodeList
}

val onnxops = loadOnnxOps()

class OnnxIRTensor(input: Onnx.TensorProto): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {

    val tensor = input


    override fun shape(): List<Long> {
        return tensor.dimsList
    }

    override fun stride(): List<Long> {
        return Nd4j.getStrides(shape().toTypedArray().toLongArray(),'c').asList()
    }

    override fun dataType(): IRDataType<Onnx.TensorProto.DataType> {
        return OnnxIRDataType(Onnx.TensorProto.DataType.values()[tensor.dataType])
    }

    override fun toArgTensor(): TensorNamespace.TensorProto {
        val builder = TensorNamespace.TensorProto.newBuilder()
                .setDataLocation(TensorNamespace.TensorProto.DataLocation.DEFAULT)

        for(i in 0 .. tensor.dimsCount) {
            builder.addDims(tensor.getDims(i))
        }

        when(tensor.dataType) {
            Onnx.TensorProto.DataType.UINT64.ordinal -> builder.dataType = TensorNamespace.DataType.UINT64.ordinal
            Onnx.TensorProto.DataType.UINT32.ordinal -> builder.dataType = TensorNamespace.DataType.UINT32.ordinal
            Onnx.TensorProto.DataType.UINT16.ordinal -> builder.dataType = TensorNamespace.DataType.UINT16.ordinal
            Onnx.TensorProto.DataType.FLOAT16.ordinal -> builder.dataType = TensorNamespace.DataType.FLOAT16.ordinal
            Onnx.TensorProto.DataType.STRING.ordinal -> builder.dataType = TensorNamespace.DataType.STRING.ordinal
            Onnx.TensorProto.DataType.FLOAT.ordinal -> builder.dataType  = TensorNamespace.DataType.FLOAT.ordinal
            Onnx.TensorProto.DataType.DOUBLE.ordinal -> builder.dataType = TensorNamespace.DataType.DOUBLE.ordinal
            Onnx.TensorProto.DataType.BOOL.ordinal -> builder.dataType = TensorNamespace.DataType.BOOL.ordinal
            Onnx.TensorProto.DataType.INT64.ordinal -> builder.dataType = TensorNamespace.DataType.INT64.ordinal
            Onnx.TensorProto.DataType.INT32.ordinal -> builder.dataType = TensorNamespace.DataType.INT32.ordinal
            Onnx.TensorProto.DataType.INT16.ordinal -> builder.dataType = TensorNamespace.DataType.INT16.ordinal
            Onnx.TensorProto.DataType.COMPLEX64.ordinal -> builder.dataType = TensorNamespace.DataType.COMPLEX64.ordinal
            Onnx.TensorProto.DataType.COMPLEX128.ordinal -> builder.dataType = TensorNamespace.DataType.COMPLEX128.ordinal
            Onnx.TensorProto.DataType.UNDEFINED.ordinal,Onnx.TensorProto.DataType.UNRECOGNIZED.ordinal ->  TensorNamespace.DataType.UNRECOGNIZED.ordinal

        }


        if(tensor.doubleDataList != null) {
            builder.addAllDoubleData(tensor.doubleDataList)
        }

        if(tensor.stringDataList != null) {
            builder.addAllStringData(tensor.stringDataList)
        }

        if(tensor.floatDataList != null) {
            builder.addAllFloatData(tensor.floatDataList)
        }

        if(tensor.int32DataList != null) {
            builder.addAllInt32Data(tensor.int32DataList)
        }

        if(tensor.uint64DataList != null) {
            builder.addAllInt64Data(tensor.uint64DataList)
        }

        if(tensor.rawData != null) {
            builder.rawData = tensor.rawData
        }

        builder.dataType = tensor.dataType

        return builder.build()
    }

    override fun rawValue(): Onnx.TensorProto {
        return tensor
    }


}

class OnnxIRDataType(inputDataType: Onnx.TensorProto.DataType): IRDataType<Onnx.TensorProto.DataType> {
    val dataType = inputDataType

    override fun convertToDataType(input: Onnx.TensorProto.DataType): IRDataTypeValue {
        when(input) {
            Onnx.TensorProto.DataType.UINT64 ->  return IRDataTypeValue.DT_UINT64
            Onnx.TensorProto.DataType.UINT32 ->  return IRDataTypeValue.DT_UINT32
            Onnx.TensorProto.DataType.UINT16 ->  return IRDataTypeValue.DT_UINT16
            Onnx.TensorProto.DataType.FLOAT16 -> return IRDataTypeValue.DT_HALF
            Onnx.TensorProto.DataType.STRING -> return IRDataTypeValue.DT_STRING
            Onnx.TensorProto.DataType.FLOAT ->  return IRDataTypeValue.DT_FLOAT
            Onnx.TensorProto.DataType.DOUBLE -> return IRDataTypeValue.DT_DOUBLE
            Onnx.TensorProto.DataType.BOOL -> return IRDataTypeValue.DT_BOOL
            Onnx.TensorProto.DataType.INT64 -> return IRDataTypeValue.DT_INT64
            Onnx.TensorProto.DataType.INT32 ->  return IRDataTypeValue.DT_INT32
            Onnx.TensorProto.DataType.INT16 -> return IRDataTypeValue.DT_INT16
            Onnx.TensorProto.DataType.COMPLEX64 ->  return IRDataTypeValue.DT_COMPLEX64
            Onnx.TensorProto.DataType.COMPLEX128 ->  return IRDataTypeValue.DT_COMPLEX128
            Onnx.TensorProto.DataType.UNDEFINED,Onnx.TensorProto.DataType.UNRECOGNIZED ->  TensorNamespace.DataType.UNRECOGNIZED.ordinal

        }

        return IRDataTypeValue.DT_INVALID
    }

    override fun dataType(): IRDataTypeValue {
        return convertToDataType(this.dataType)
    }

    override fun internalValue(): Onnx.TensorProto.DataType {
        return this.dataType
    }

}

fun attrDefaultValue(): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
    return OnnxIRAttr(Onnx.AttributeProto.getDefaultInstance(), Onnx.AttributeProto.getDefaultInstance())
}

class OnnxIRAttr(inputAttributeDef: Onnx.AttributeProto, inputAttributeValue: Onnx.AttributeProto):
        IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {

    private val attributeDef = inputAttributeDef
    private val attributeValue = inputAttributeValue

    override fun name(): String {
        return attributeDef.name
    }

    override fun floatValue(): Float {
        return attributeValue.f
    }

    override fun listFloatValue(): List<Float> {
        return attributeValue.floatsList
    }


    override fun intValue(): Long {
        return attributeValue.i
    }

    override fun listIntValue(): List<Long> {
        return attributeValue.intsList
    }

    override fun boolValue(): Boolean {
        TODO("Implement")
    }

    override fun listBoolValue(): List<Boolean> {
        TODO("Implement")
    }

    override fun attributeValueType(): AttributeValueType {
        when(attributeDef.type) {
            Onnx.AttributeProto.AttributeType.STRING -> return AttributeValueType.STRING
            Onnx.AttributeProto.AttributeType.STRINGS -> return AttributeValueType.LIST_STRING
            Onnx.AttributeProto.AttributeType.INT-> return AttributeValueType.INT
            Onnx.AttributeProto.AttributeType.INTS -> return AttributeValueType.LIST_INT
            Onnx.AttributeProto.AttributeType.FLOAT -> return AttributeValueType.FLOAT
            Onnx.AttributeProto.AttributeType.FLOATS -> return AttributeValueType.LIST_FLOAT
            Onnx.AttributeProto.AttributeType.TENSOR -> return AttributeValueType.TENSOR
            Onnx.AttributeProto.AttributeType.TENSORS -> return AttributeValueType.LIST_TENSOR
        }

        return AttributeValueType.INVALID
    }



    override fun internalAttributeDef(): Onnx.AttributeProto {
        return attributeDef
    }

    override fun internalAttributeValue(): Onnx.AttributeProto {
        return attributeValue
    }

    override fun listTensorValue(): List<IRTensor<Onnx.TensorProto,Onnx.TensorProto.DataType>> {
        return attributeValue.tensorsList.map {
            input -> OnnxIRTensor(input)
        }
    }

    override fun tensorValue(): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRTensor(attributeValue.t)
    }

    override fun stringValue(): String {
        return attributeValue.s.toStringUtf8()
    }

    override fun listStringValue(): List<String> {
        return attributeValue.stringsList.map { it.toStringUtf8() }
    }

}

class OnnxIRArgDef(input: Onnx.NodeProto): IRArgDef<Onnx.NodeProto,Onnx.TensorProto.DataType> {
    private val argDefValue = input

    override fun dataType(): IRDataType<Onnx.TensorProto.DataType> {
        return OnnxIRArgDef(argDefValue).dataType()
    }

    override fun name(): String {
        return argDefValue.name
    }

    override fun description(): String {
        return argDefValue.docString
    }

    override fun internalValue(): Onnx.NodeProto {
        return argDefValue
    }

    override fun indexOf(): Integer {
        TODO("Not yet implemented")
    }

}

class OnnxIROp(input: Onnx.NodeProto): IROpDef<Onnx.NodeProto, Onnx.TensorProto, Onnx.NodeProto,Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto> {

    val opDef = input

    override fun attributes(): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto,Onnx.TensorProto.DataType>> {
        return opDef.attributeList.map {
            OnnxIRAttr(it, Onnx.AttributeProto.getDefaultInstance())
        }
    }

    override fun opName(): String {
        return opDef.name
    }

    override fun internalValue(): Onnx.NodeProto {
        return opDef
    }

    override fun inputArgs(): List<IRArgDef<Onnx.NodeProto, Onnx.TensorProto.DataType>> {
        return opDef.inputList.map {
            OnnxIRArgDef(opDef)
        }
    }

    override fun outputArgs(): List<IRArgDef<Onnx.NodeProto,Onnx.TensorProto.DataType>> {
        return opDef.outputList.map {
            OnnxIRArgDef(opDef)
        }
    }

}

class OnnxIRNode(inputNode: Onnx.NodeProto, inputOpDef: Onnx.NodeProto): IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType> {

    private val nodeDef = inputNode
    private val opDef = inputOpDef
    private val attrDefsMap = attrDefsByName(inputOpDef.attributeList)
    private val attrMap: Map<String, IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto,Onnx.TensorProto.DataType>> =
            initAttrMapFromNode(inputNode)

    init {

    }

    private fun attrDefsByName(input: List<Onnx.AttributeProto>): Map<String,Onnx.AttributeProto> {
        val ret = HashMap<String,Onnx.AttributeProto>()
        input.forEach {
            ret[it.name] = it
        }
        return ret
    }

    private fun initAttrMapFromNode(input: Onnx.NodeProto): Map<String, IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        val ret = HashMap<String, IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto,Onnx.TensorProto.DataType>>()
        input.attributeList.forEach {
            ret[it.name] = OnnxIRAttr(it,it)

        }
        return ret
    }

    override fun opName(): String {
        return nodeDef.opType
    }

    override fun nodeName(): String {
        return nodeDef.name
    }

    override fun inputAt(index: Int): String {
        return nodeDef.getInput(index)
    }

    override fun outputAt(index: Int): String {
        return opDef.getOutput(index)
    }



    override fun hasAttribute(inputName: String): Boolean {
        return nodeDef.attributeList.filter { it.name == inputName }.size > 0
    }

    override fun attributeMap(): Map<String, IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        return attrMap
    }

    override fun createInputsFrom(inputData: List<Onnx.TensorProto>): List<IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        return inputData.map { OnnxIRTensor(it) }
    }

    override fun createOutputsFrom(inputValues: List<Onnx.TensorProto>): List<IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        return inputValues.map { OnnxIRTensor(it) }
    }

    override fun getAttribute(inputName: String): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return attrMap.getOrDefault(inputName, attrDefaultValue())
    }

    override fun internalValue(): Onnx.NodeProto {
        return nodeDef
    }

}


fun Onnx.GraphProto.nodeByName(name: String): Onnx.NodeProto {
    return this.nodeList.first { it.name == name }!!
}

class OnnxIRGraph(graphDef: Onnx.GraphProto): IRGraph<Onnx.NodeProto,
        Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType> {

    val graphDef = graphDef
    val opList = graphDef.nodeList
    override fun nodeByName(input: String): Onnx.NodeProto {
        return graphDef.nodeByName(input)
    }

    override fun nodeList(): List<Onnx.NodeProto> {
        return graphDef.nodeList
    }

    override fun opDefFor(name: String): Onnx.NodeProto {
        return opList.first { it.name == name }!!
    }

    fun graphDef(): Onnx.GraphProto {
        return graphDef
    }

}


class OnnxMappingContext(opDef: Onnx.NodeProto, node: Onnx.NodeProto, graph:
IRGraph<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto,
        Onnx.AttributeProto,
        Onnx.AttributeProto, Onnx.TensorProto.DataType>) :
        AbstractMappingContext<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>(opDef, node, graph) {

    override fun attrDef(name: String): Onnx.AttributeProto {
        val ret = opDef().attributeList.firstOrNull { it.name == name }
        return ret!!
    }

    override fun irAttributeValueForNode(valueName: String): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        val attrDef = attrDef(valueName)
        val attrValue = node.attributeList.first { it.name == valueName }
        return OnnxIRAttr(inputAttributeDef = attrDef,inputAttributeValue = attrValue)

    }

    override fun tensorInputFor(name: String): IRTensor<Onnx.TensorProto,Onnx.TensorProto.DataType> {
        var foundIndex = -1
        /**
         * Use op definition name as 1 unified reference name in rules for static purposes, but
         * look up via index for specific node mappings.
         *
         * This is equivalent to the tf input position attribute value in the previous tensorflow import.
         */
        opDef.inputList.forEachIndexed {
            index,argDef -> if(argDef == name) foundIndex = index
        }


        val targetName = opDef.getInput(foundIndex)
        val graphNode = graph.nodeByName(targetName)
        val castedGraph = graph as OnnxIRGraph
        val graphDef = castedGraph.graphDef()
        graphDef.initializerList.filter { it.name == name }
        //value nodes are the values of attributes that are input nodes in a frozen graph
        return OnnxIRTensor(graphDef.initializerList.first { it.name == name })
    }

}

fun NodeProto(block: Onnx.NodeProto.Builder.() -> Unit): Onnx.NodeProto {
    return Onnx.NodeProto.newBuilder().apply(block).build()
}


fun Onnx.NodeProto.Builder.Input(inputName: String) {
    this.addInput(inputName)
}

fun GraphProto(block: Onnx.GraphProto.Builder.() -> Unit): Onnx.GraphProto {
    return Onnx.GraphProto.newBuilder().apply(block).build()
}

fun Onnx.GraphProto.Builder.Node(inputNode: Onnx.NodeProto) {
    this.addNode(inputNode)
}





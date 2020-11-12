package org.nd4j.codegen.ir.tensorflow

import org.apache.commons.io.IOUtils
import org.nd4j.codegen.ir.*
import org.nd4j.common.io.ClassPathResource
import org.nd4j.imports.graphmapper.tf.tensors.TFTensorMappers
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.shade.protobuf.TextFormat
import org.tensorflow.framework.*
import org.tensorflow.framework.OpDef.AttrDef
import java.lang.IllegalArgumentException
import java.nio.charset.Charset

fun loadTensorflowOps(): OpList {
    val string = IOUtils.toString(ClassPathResource("ops.proto").inputStream, Charset.defaultCharset())
    val tfListBuilder = OpList.newBuilder()
    TextFormat.merge(string, tfListBuilder)
    return tfListBuilder.build()
}

val tensorflowOps = loadTensorflowOps()




class TensorflowIRTensor(input: TensorProto): IRTensor<TensorProto, DataType> {

    val tensor = input


    override fun shape(): List<Long> {
        return  tensor.tensorShape.dimList.map { it.size }

    }

    override fun stride(): List<Long> {
        return Nd4j.getStrides(shape().toTypedArray().toLongArray(), 'c').asList()
    }

    override fun dataType(): IRDataType<DataType> {
        return TensorflowIRDataType(tensor.dtype)
    }

    override fun toArgTensor(): TensorNamespace.TensorProto {
        val builder = TensorNamespace.TensorProto.newBuilder()
                .setDataLocation(TensorNamespace.TensorProto.DataLocation.DEFAULT)

        for(i in 0 .. tensor.tensorShape.dimCount) {
            builder.addDims(tensor.tensorShape.getDim(i).size)
        }

        when(tensor.dtype) {
            DataType.DT_UINT64 -> builder.dataType = TensorNamespace.DataType.UINT64.ordinal
            DataType.DT_UINT32 -> builder.dataType = TensorNamespace.DataType.UINT32.ordinal
            DataType.DT_UINT16 -> builder.dataType = TensorNamespace.DataType.UINT16.ordinal
            DataType.DT_HALF -> builder.dataType = TensorNamespace.DataType.FLOAT16.ordinal
            DataType.DT_STRING -> builder.dataType = TensorNamespace.DataType.STRING.ordinal
            DataType.DT_FLOAT -> builder.dataType = TensorNamespace.DataType.FLOAT.ordinal
            DataType.DT_DOUBLE -> builder.dataType = TensorNamespace.DataType.DOUBLE.ordinal
            DataType.DT_BOOL -> builder.dataType = TensorNamespace.DataType.BOOL.ordinal
            DataType.DT_INT64 -> builder.dataType = TensorNamespace.DataType.INT64.ordinal
            DataType.DT_INT32 -> builder.dataType = TensorNamespace.DataType.INT32.ordinal
            DataType.DT_INT16 -> builder.dataType = TensorNamespace.DataType.INT16.ordinal
            DataType.DT_BFLOAT16 -> builder.dataType = TensorNamespace.DataType.BFLOAT16.ordinal
            DataType.DT_COMPLEX64 -> builder.dataType = TensorNamespace.DataType.COMPLEX64.ordinal
            DataType.DT_COMPLEX128 -> builder.dataType = TensorNamespace.DataType.COMPLEX128.ordinal
            DataType.UNRECOGNIZED -> builder.dataType = TensorNamespace.DataType.UNRECOGNIZED.ordinal

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

    override fun toNd4jNDArray(): INDArray {
        return TFTensorMappers.newMapper(tensor).toNDArray()
    }
}

class TensorflowIRDataType(inputDataType: DataType): IRDataType<DataType> {
    val dataType = inputDataType

    override fun convertToDataType(input: DataType): IRDataTypeValue {
        when(input) {
            DataType.DT_BOOL -> return IRDataTypeValue.DT_BOOL
            DataType.DT_BFLOAT16 -> return IRDataTypeValue.DT_BFLOAT16
            DataType.DT_COMPLEX128 -> return IRDataTypeValue.DT_COMPLEX128
            DataType.DT_COMPLEX64 -> return IRDataTypeValue.DT_COMPLEX64
            DataType.DT_DOUBLE -> return IRDataTypeValue.DT_DOUBLE
            DataType.DT_FLOAT -> return IRDataTypeValue.DT_FLOAT
            DataType.DT_HALF -> return IRDataTypeValue.DT_HALF
            DataType.DT_INT16 -> return IRDataTypeValue.DT_INT16
            DataType.DT_INT32 -> return IRDataTypeValue.DT_INT32
            DataType.DT_INT64 -> return IRDataTypeValue.DT_INT64
            DataType.DT_QINT8 -> return IRDataTypeValue.DT_QINT8
            DataType.DT_QINT16 -> return IRDataTypeValue.DT_QINT16
            DataType.DT_QINT32 -> return IRDataTypeValue.DT_QINT32
            DataType.DT_STRING -> return IRDataTypeValue.DT_STRING
            DataType.DT_UINT16 -> return IRDataTypeValue.DT_UINT16
            DataType.DT_UINT32 -> return IRDataTypeValue.DT_UINT32
            DataType.DT_UINT64 -> return IRDataTypeValue.DT_UINT64

        }

        return IRDataTypeValue.DT_INVALID
    }



    override fun dataType(): IRDataTypeValue {
        return convertToDataType(this.dataType)
    }

    override fun internalValue(): DataType {
        return this.dataType
    }

    override fun nd4jDataType(): org.nd4j.linalg.api.buffer.DataType {
        when(this.dataType) {
            DataType.DT_BOOL -> return org.nd4j.linalg.api.buffer.DataType.BOOL
            DataType.DT_FLOAT -> return org.nd4j.linalg.api.buffer.DataType.FLOAT
            DataType.DT_STRING -> return org.nd4j.linalg.api.buffer.DataType.UTF8
            DataType.DT_BFLOAT16 -> return org.nd4j.linalg.api.buffer.DataType.BFLOAT16
            DataType.DT_INT64 -> return org.nd4j.linalg.api.buffer.DataType.INT64
            DataType.DT_HALF -> return org.nd4j.linalg.api.buffer.DataType.FLOAT16
            DataType.DT_INT16 -> return org.nd4j.linalg.api.buffer.DataType.INT16
            DataType.DT_INT32 -> return org.nd4j.linalg.api.buffer.DataType.INT32
            DataType.DT_DOUBLE -> return org.nd4j.linalg.api.buffer.DataType.DOUBLE
            DataType.DT_UINT16 -> return org.nd4j.linalg.api.buffer.DataType.UINT16
            DataType.DT_UINT32 -> return org.nd4j.linalg.api.buffer.DataType.UINT32
            DataType.DT_UINT64 -> return org.nd4j.linalg.api.buffer.DataType.UINT64
        }

        return org.nd4j.linalg.api.buffer.DataType.UNKNOWN
    }

}

fun attrDefaultValue(): IRAttribute<AttrDef, AttrValue, TensorProto, DataType> {
    return TensorflowIRAttr(AttrDef.getDefaultInstance(), AttrValue.getDefaultInstance())
}

class TensorflowIRAttr(inputAttributeDef: AttrDef, inputAttributeValue: AttrValue): IRAttribute<AttrDef, AttrValue, TensorProto, DataType> {

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
            "list(bool)" -> return AttributeValueType.LIST_BOOL
            "bool" -> return AttributeValueType.BOOL
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

    override fun listTensorValue(): List<IRTensor<TensorProto, DataType>> {
        return attributeValue.list.tensorList.map { input -> TensorflowIRTensor(input)
        }
    }

    override fun tensorValue(): IRTensor<TensorProto, DataType> {
        return TensorflowIRTensor(attributeValue.tensor)
    }

    override fun stringValue(): String {
        return attributeValue.s.toStringUtf8()
    }

    override fun listStringValue(): List<String> {
        return attributeValue.list.sList.map { it.toStringUtf8() }
    }

}

class TensorflowIRArgDef(input: OpDef.ArgDef): IRArgDef<OpDef.ArgDef, DataType> {
    private val argDefValue = input

    override fun dataType(): IRDataType<DataType> {
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

class TensorflowIROp(input: OpDef): IROpDef<OpDef, TensorProto, OpDef.ArgDef, DataType, AttrDef, AttrValue> {

    val opDef = input

    override fun attributes(): List<IRAttribute<AttrDef, AttrValue, TensorProto, DataType>> {
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

    override fun inputArgs(): List<IRArgDef<OpDef.ArgDef, DataType>> {
        return opDef.inputArgList.map {
            TensorflowIRArgDef(it)
        }
    }

    override fun outputArgs(): List<IRArgDef<OpDef.ArgDef, DataType>> {
        return opDef.outputArgList.map {
            TensorflowIRArgDef(it)
        }
    }

}

class TensorflowIRNode(inputNode: NodeDef, inputOpDef: OpDef): IRNode<NodeDef, TensorProto, AttrDef, AttrValue, DataType> {

    private val nodeDef = inputNode
    private val opDef = inputOpDef
    private val attrDefsMap = attrDefsByName(inputOpDef.attrList)
    private val attrMap: Map<String, IRAttribute<AttrDef, AttrValue, TensorProto, DataType>> = initAttrMapFromNode(inputNode)

    init {

    }

    private fun attrDefsByName(input: List<AttrDef>): Map<String, AttrDef> {
        val ret = HashMap<String, AttrDef>()
        input.forEach {
            ret[it.name] = it
        }
        return ret
    }

    private fun initAttrMapFromNode(input: NodeDef): Map<String, IRAttribute<AttrDef, AttrValue, TensorProto, DataType>> {
        val ret = HashMap<String, IRAttribute<AttrDef, AttrValue, TensorProto, DataType>>()
        input.attrMap.forEach { (key, value) ->
            ret[key] =  TensorflowIRAttr(attrDefsMap.getOrDefault(key, AttrDef.getDefaultInstance()), value)
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

    override fun attributeMap(): Map<String, IRAttribute<AttrDef, AttrValue, TensorProto, DataType>> {
        return attrMap
    }

    override fun createInputsFrom(inputData: List<TensorProto>): List<IRTensor<TensorProto, DataType>> {
        return inputData.map { TensorflowIRTensor(it) }
    }

    override fun createOutputsFrom(inputValues: List<TensorProto>): List<IRTensor<TensorProto, DataType>> {
        return inputValues.map { TensorflowIRTensor(it) }
    }

    override fun getAttribute(inputName: String): IRAttribute<AttrDef, AttrValue, TensorProto, DataType> {
        return attrMap.getOrDefault(inputName, attrDefaultValue())
    }

    override fun internalValue(): NodeDef {
        return nodeDef
    }

}


class TensorflowIRGraph(graphDef: GraphDef, opDef: OpList): IRGraph<NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType> {

    val graphDef = graphDef
    val opList = opDef
    override fun nodeByName(input: String): NodeDef {
        return graphDef.nodeByName(input)
    }

    override fun nodeList(): List<NodeDef> {
        return graphDef.nodeList
    }

    override fun opDefFor(name: String): OpDef {
        return opList.opList.first { it.name == name }!!
    }

}


class TensorflowImportProcess(inputFramework: String = "tensorflow") : AbstractImportProcess<OpDef, NodeDef, TensorProto, AttrDef, AttrValue, DataType>(inputFramework) {
    override fun createMappingContext(graph: IRGraph<NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType>, node: NodeDef): MappingContext<NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType> {
        val opDef = tensorflowOps.findOp(node.op)
        return TensorflowMappingContext(graph = graph, node = node, opDef = opDef)
    }

    override fun createImportContext(mappingProcess: MappingProcess<OpDef, NodeDef, TensorProto, AttrDef, AttrValue, DataType>, mappingContext: MappingContext<NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType>):
            ImportContext<OpDef, NodeDef, TensorProto, AttrDef, AttrValue, DataType> {
        return TensorflowImportContext(mappingContext = mappingContext, process = mappingProcess)
    }

}

class TensorflowImportContext(process: MappingProcess<OpDef, NodeDef, TensorProto, AttrDef, AttrValue, DataType>, mappingContext: MappingContext<NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType>) : AbstractImportContext<OpDef, NodeDef, TensorProto, AttrDef, AttrValue, DataType>(process, mappingContext) {

    override fun process(): MappingProcess<OpDef, NodeDef, TensorProto, AttrDef, AttrValue, DataType> {
        return process
    }

    override fun mappingContext(): MappingContext<NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType> {
        return mappingContext
    }

}


fun convertToDataType(dataType: org.nd4j.linalg.api.buffer.DataType): DataType {
    return when (dataType) {
        org.nd4j.linalg.api.buffer.DataType.UINT16 -> DataType.DT_UINT16
        org.nd4j.linalg.api.buffer.DataType.UINT32 -> DataType.DT_UINT32
        org.nd4j.linalg.api.buffer.DataType.UINT64 -> DataType.DT_UINT64
        org.nd4j.linalg.api.buffer.DataType.BOOL -> DataType.DT_BOOL
        org.nd4j.linalg.api.buffer.DataType.BFLOAT16 -> DataType.DT_BFLOAT16
        org.nd4j.linalg.api.buffer.DataType.FLOAT -> DataType.DT_FLOAT
        org.nd4j.linalg.api.buffer.DataType.INT -> DataType.DT_INT32
        org.nd4j.linalg.api.buffer.DataType.LONG -> DataType.DT_INT64
        org.nd4j.linalg.api.buffer.DataType.BYTE -> DataType.DT_INT8
        org.nd4j.linalg.api.buffer.DataType.SHORT -> DataType.DT_INT16
        org.nd4j.linalg.api.buffer.DataType.DOUBLE -> DataType.DT_DOUBLE
        org.nd4j.linalg.api.buffer.DataType.UBYTE -> DataType.DT_UINT8
        org.nd4j.linalg.api.buffer.DataType.HALF -> DataType.DT_HALF
        org.nd4j.linalg.api.buffer.DataType.UTF8 -> DataType.DT_STRING
        else -> throw UnsupportedOperationException("Unknown TF data type: [" + dataType.name + "]")
    }
}


class TensorflowMappingContext(opDef: OpDef, node: NodeDef, graph: IRGraph<NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType>) :
        AbstractMappingContext<NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType>(opDef, node, graph) {

    override fun attrDef(name: String): AttrDef {
        if(opDef().attrCount < 1) {
            throw IllegalArgumentException("No attributes found for op def with name ${opDef.name}")
        }

        val ret =  opDef().attrList.firstOrNull { it.name == name } ?: error("No attribute found with name $name")
        return ret!!
    }

    override fun irAttributeValueForNode(valueName: String): IRAttribute<AttrDef, AttrValue, TensorProto, DataType> {
        val attrDef = attrDef(valueName)
        val attrValue = node.getAttrOrDefault(valueName, AttrValue.getDefaultInstance())
        return TensorflowIRAttr(inputAttributeDef = attrDef, inputAttributeValue = attrValue)

    }

    override fun tensorInputFor(name: String): IRTensor<TensorProto, DataType> {
        var foundIndex = -1
        /**
         * Use op definition name as 1 unified reference name in rules for static purposes, but
         * look up via index for specific node mappings.
         *
         * This is equivalent to the tf input position attribute value in the previous tensorflow import.
         */
        opDef.inputArgList.forEachIndexed { index, argDef ->
            if(argDef.name == name)
                foundIndex = index
        }

        val targetName = opDef.getInputArg(foundIndex).name
        val graphNode = graph.nodeByName(targetName)
        //value nodes are the values of attributes that are input nodes in a frozen graph
        return TensorflowIRTensor(graphNode.getAttrOrThrow("value").tensor)
    }

    override fun opName(): String {
        return node.op
    }

    override fun nodeName(): String {
        return node.name
    }

    override fun nd4jDataTypeFor(input: IRTensor<TensorProto, DataType>): org.nd4j.linalg.api.buffer.DataType {
        return input.dataType().nd4jDataType()
    }

    override fun createIRTensorFromNDArray(ndarray: INDArray): IRTensor<TensorProto, DataType> {
        val tensorProto = TensorProto {
            RawData(ndarray.data().asBytes())
            Shape(ndarray.shape().toList())
            DataType(convertToDataType(ndarray.dataType()))
        }

        return TensorflowIRTensor(tensorProto)
    }

}





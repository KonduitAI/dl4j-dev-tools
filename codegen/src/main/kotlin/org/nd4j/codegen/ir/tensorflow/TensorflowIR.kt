package org.nd4j.codegen.ir.tensorflow

import org.apache.commons.io.IOUtils
import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.VariableType
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.autodiff.samediff.internal.Variable
import org.nd4j.codegen.ir.*
import org.nd4j.common.base.Preconditions
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.io.ReflectionUtils
import org.nd4j.graph.OpType
import org.nd4j.imports.converters.DifferentialFunctionClassHolder
import org.nd4j.imports.graphmapper.tf.TFGraphMapper
import org.nd4j.imports.graphmapper.tf.tensors.TFTensorMappers
import org.nd4j.imports.tensorflow.TFImportOverride
import org.nd4j.imports.tensorflow.TFOpImportFilter
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.Op
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge
import org.nd4j.linalg.api.shape.Shape
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.shade.protobuf.TextFormat
import org.tensorflow.framework.*
import org.tensorflow.framework.OpDef.AttrDef
import java.nio.charset.Charset
import java.util.*
import kotlin.collections.HashMap
import kotlin.collections.HashSet

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

        for(i in 0 until tensor.tensorShape.dimCount) {
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
        val attrValue = node.getAttrOrDefault(valueName, attrDef.defaultValue)
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

        val graphNode = node.getInput(foundIndex)
        val searchedNode = graph.nodeByName(graphNode)
        //no value to be found on placeholder, return default instance
        //if no value exists it's an output from another node
        if("Placeholder" in searchedNode.op || !searchedNode.containsAttr("value")) {
            return TensorflowIRTensor(TensorProto.getDefaultInstance())
        }

        //value nodes are the values of attributes that are input nodes in a frozen graph
        return TensorflowIRTensor(searchedNode.getAttrOrThrow("value").tensor)
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

fun tensorflowAttributeValueTypeFor(attributeName: String, opDef: OpDef): AttributeValueType {
    val names = opDef.attrList.map { attrDef -> attrDef.name }
    if(!names.contains(attributeName) && !isTensorflowTensorName(attributeName,opDef)) {
        throw java.lang.IllegalArgumentException("Tensorflow op ${opDef.name} does not have attribute name $attributeName")
    } else if(isTensorflowTensorName(attributeName,opDef)) {
      //note we allows tensors here since sometimes input tensors in tensorflow become attributes in nd4j
        return AttributeValueType.TENSOR
    }
    val attrDef = opDef.attrList.first { attrDef -> attrDef.name == attributeName }
    return TensorflowIRAttr(attrDef, AttrValue.getDefaultInstance()).attributeValueType()
}



fun isTensorflowTensorName(name: String, opDef: OpDef): Boolean {
    return opDef.inputArgList.map {inputDef -> inputDef.name }.contains(name)
}


fun isTensorflowAttributeName(name: String, opDef: OpDef): Boolean {
    return opDef.attrList.map { attrDef -> attrDef.name }.contains(name)
}




/**
 * Import a TensorFlow model from a GraphDef, with optional import overrides
 *
 * @param tfGraph        TensorFlow model GraphDef
 * @param importOverride Optional import override for specific ops, keyed by op name
 * @param opFilter       Optional filter - ops to exclude/ignore
 * @return Imported model
 */
fun importGraph(tfGraph: GraphDef, importOverride: Map<String?, TFImportOverride?>?,
                opFilter: TFOpImportFilter?): SameDiff? {

    /*
        First, build an in-memory representation of the graph that allows us to build the graph incrementally
        If we can build the graph incrementally, we can make sure that the added variables are set up with the correct
        datatype and (once implemented) greedy shape inference
         */
    val availableToAddSet: MutableSet<String> = HashSet() //TODO maybe unnecessary?
    val availableToAdd: Queue<NodeDef> = LinkedList()
    val remainingNodes: MutableMap<String, NodeDef> = HashMap() //All other nodes, not in availableToAdd
    val nodeInputTo: MutableMap<String, MutableSet<String>?> = HashMap() // For op x -> y, x is key, y is value. Note that these are OP names not VARIABLE names
    val nNodes = tfGraph.nodeCount
    val tfGraph2 = TensorflowIRGraph(graphDef = tfGraph,opDef = tensorflowOps)

    //val mappingContext = TensorflowMappingContext()
    //First, add any constants, placeholders, and zero-input ops
    val sd = SameDiff.create()
    for (i in 0 until nNodes) {
        val nd = tfGraph.getNode(i)
        val op = nd.op

        val name = nd.name
        val nInputs = nd.inputCount
        if ("Const" == op || "Placeholder" == op || nInputs == 0) {
            availableToAdd.add(nd)
            availableToAddSet.add(name)
        } else {
            remainingNodes[name] = nd
            for (`in` in 0 until nInputs) {
                var inOpName = stripControl(nd.getInput(`in`))
                inOpName = stripVarSuffix(inOpName)
                if (!nodeInputTo.containsKey(inOpName)) {
                    nodeInputTo[inOpName] = HashSet()
                }
                nodeInputTo[inOpName]!!.add(name)
            }
        }
    }


    val mergeOpsPostProcess: MutableMap<String, String> = HashMap()

    //Go through ops in order, and add to the graph
    val constControlDeps: MutableMap<String, List<String>> = HashMap() //Key: constant name. Value: control dependencies
    while (!availableToAdd.isEmpty()) {
        val nd = availableToAdd.remove()
        val name = nd.name
        val opName = nd.op
        val nIn = nd.inputCount
        val opDefLookup = tfGraph2.opDefFor(opName)

        availableToAddSet.remove(name)
        println("Adding operation to graph: $opName (name=$name)")
        var skipCase = false
        if (opFilter != null && opFilter.skipOp(nd, sd, nd.attrMap, tfGraph)) {
            println("Skipping op $name of type $opName due to op filter")
            //Don't continue at this point - we still need to process what this feeds into...
            skipCase = true
        } else {
            if (importOverride == null || !importOverride.containsKey(name)) {
                //Standard case
                if ("Const" == opName) {
                    //Get array, create a constant
                    val tfTensor = nd.getAttrOrThrow("value").tensor
                    val m = TFTensorMappers.newMapper(tfTensor)
                    val arr = m.toNDArray()
                    sd.constant(name, arr)
                    val inputCount = nd.inputCount
                    if (inputCount > 0) {
                        //Very likely control dependency. i.e., "we must execute op X before the constant is really available to be used"
                        val l: MutableList<String> = ArrayList(inputCount)
                        for (i in 0 until inputCount) {
                            val n = nd.getInput(i)
                            check(isControlDep(n)) { "Found non-control dependency input \"$n\" for constant \"$name\"" }
                            val n2 = stripControl(n)
                            l.add(n2)
                        }
                        constControlDeps[name] = l
                    }
                } else if ("Placeholder" == opName || "PlaceholderWithDefault" == opName) {
                    //TODO support the "WithDefault" array
                    val attrMap = nd.attrMap
                    val shapeAvailable = attrMap.containsKey("shape")
                    var shape: LongArray?
                    shape = if (shapeAvailable) {
                        val shapeProto = attrMap["shape"]!!.shape
                        shapeFromShapeProto(shapeProto)
                    } else {
                        //Some placeholders don't have any shape restrictions - i.e., accept anything...
                        null
                    }
                    val tfDtype = attrMap["dtype"]!!.type
                    val dt = convertType(tfDtype)
                    if(shape != null)
                        sd.placeHolder(name, dt, *shape)
                    else
                        sd.placeHolder(name, dt)
                } else {
                    /*
                        Normal ops. Process in the following order:
                        1. Create the op instance
                        2. Add op to graph
                        3. Import from TF (to set attributes)
                        4. Calculate output dtypes
                        5. Create and add output variables to graph

                        Note: one constraint on this order is that some ops import modify the graph structure.
                        Notable example: concat op - it removes the axis op and converts the value to an iArg
                        https://github.com/eclipse/deeplearning4j/issues/8285
                         */
                    val dfInstance = DifferentialFunctionClassHolder.getInstance().getOpWithTensorflowName(opName)
                    Preconditions.checkState(dfInstance != null, "Could not find class for TF Ops: %s", opName)
                    var df: DifferentialFunction
                    df = try {
                        dfInstance.javaClass.newInstance()
                    } catch (t: Throwable) {
                        //Should never happen because function was already created via no-arg constructor earlier
                        throw RuntimeException(t)
                    }
                    df.sameDiff = sd
                    df.ownName = name

                    //Process inputs
                    val inNames: MutableList<String> = ArrayList(nIn)
                    var controlDeps: MutableList<String?>? = null
                    for (i in 0 until nIn) {
                        val origInName = nd.getInput(i)
                        var inName = stripControl(origInName)
                        if (inName.endsWith(":0")) {
                            //Strip ":0" suffix. Some ops can depend on placeholders, like "image_tensor:0" but in SameDiff this is a variable called "image_tensor"
                            inName = inName.substring(0, inName.length - 2)
                        }
                        val isControlDep = isControlDep(origInName)
                        if (isControlDep) {
                            if (controlDeps == null) controlDeps = ArrayList()
                            controlDeps.add(inName)
                        }
                        if (!isControlDep) {
                            inNames.add(inName)
                        }

                        //Update Variable.inputsForOp for all variables that feed into this op
                        // Such variables must have already been created, given we process in order
                        val v = sd.variables[inName]
                        if (v == null && df is Merge) {
                            //Edge case for import - we allow merge ops to be added before both inputs are available
                            //This is to break the cycles in loops, otherwise we can't process anything in order
                            mergeOpsPostProcess[df.getOwnName()] = inName
                            continue
                        }
                        if (!isControlDep && (v!!.inputsForOp == null || !v.inputsForOp.contains(name))) {
                            //May already be present - for example, add(x,x)
                            if (v.inputsForOp == null) v.inputsForOp = ArrayList()
                            v.inputsForOp.add(name)
                        } else if (isControlDep) {
                            if (v!!.controlDepsForOp == null) v.controlDepsForOp = ArrayList()
                            if (!v.controlDepsForOp.contains(name)) {
                                v.controlDepsForOp.add(name)
                            }
                        }
                    }

                    val mappingContext = TensorflowMappingContext(graph = tfGraph2,opDef = opDefLookup,node = nd)
                    val mappingProcess = tensorflowOpRegistry.lookupOpMappingProcess(opName)
                    val applied = mappingProcess.applyProcess(mappingContext)


                    //Create SameDiffOp instance and add to graph
                    val op = SameDiffOp.builder()
                            .name(name)
                            .op(df)
                            .inputsToOp(inNames) //.outputsOfOp(outNames)    //We'll set this later
                            .controlDeps(controlDeps)
                            .build()
                    sd.ops[name] = op
                    val attrMap = nd.attrMap
                    when(df.opType()) {
                        Op.Type.CUSTOM -> {
                            val  dynamicCustomOp =  df as DynamicCustomOp
                            val grouped = applied.second.argDescriptorList.groupBy { descriptor ->
                                descriptor.argType
                            }

                            val sortedMap = HashMap<OpNamespace.ArgDescriptor.ArgType,List<OpNamespace.ArgDescriptor>>()
                            grouped.forEach { (argType,list) ->
                                sortedMap[argType] = list.sortedBy { arg -> arg.argIndex }
                            }

                            sortedMap.forEach { (argType, listOfArgsSortedByIndex) ->
                                when(argType) {
                                    OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR -> {
                                        val args = dynamicCustomOp.args()
                                        listOfArgsSortedByIndex.forEachIndexed { index, argDescriptor ->
                                            val convertedTensor = ndarrayFromNameSpaceTensor(argDescriptor.inputValue)
                                            val arg = args[index]
                                            if(arg.variableType != VariableType.ARRAY) {
                                                if(arg.shape == null) {
                                                    val emptyLongArray = LongArray(0)
                                                    arg.setShape(*emptyLongArray)
                                                }

                                                if(!Shape.shapeEquals(arg.shape,convertedTensor.shape())) {
                                                    arg.setShape(*convertedTensor.shape())
                                                    dynamicCustomOp.addInputArgument(convertedTensor)
                                                }
                                                else {
                                                    //dynamicCustomOp.addInputArgument(convertedTensor)
                                                }
                                            }

                                        }

                                    }

                                    OpNamespace.ArgDescriptor.ArgType.INT64, OpNamespace.ArgDescriptor.ArgType.INT32 -> {
                                        listOfArgsSortedByIndex.forEach { dynamicCustomOp.addIArgument(it.int64Value) }
                                    }

                                    OpNamespace.ArgDescriptor.ArgType.DOUBLE, OpNamespace.ArgDescriptor.ArgType.FLOAT -> {
                                        listOfArgsSortedByIndex.forEach { dynamicCustomOp.addTArgument(it.doubleValue) }
                                    }

                                    OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR -> {
                                        listOfArgsSortedByIndex.forEach {
                                            val convertedTensor = ndarrayFromNameSpaceTensor(it.inputValue)
                                            dynamicCustomOp.addOutputArgument(convertedTensor)
                                        }
                                    }

                                    OpNamespace.ArgDescriptor.ArgType.BOOL -> {
                                        listOfArgsSortedByIndex.forEach {
                                            dynamicCustomOp.addBArgument(it.boolValue)
                                        }
                                    }

                                    OpNamespace.ArgDescriptor.ArgType.DATA_TYPE -> {
                                        listOfArgsSortedByIndex.forEach {
                                            val dtype = convertNd4jDataTypeFromNameSpaceTensorDataType(it.dataTypeValue!!)
                                            dynamicCustomOp.addDArgument(dtype)
                                        }
                                    }
                                    else -> {
                                        throw java.lang.IllegalArgumentException("Illegal type")
                                    }

                                }
                            }


                        }
                        else -> {
                            //TODO: still need to finish mapping, mainly need to see how to add the op
                            //in to the graph (this is techincally done down below but still need to verify and test this
                            applied.second.argDescriptorList.forEach { argDescriptor ->
                                val field = ReflectionUtils.findField(df.javaClass, argDescriptor.name)
                                if (field != null) {
                                    field.isAccessible = true
                                    when (argDescriptor.name) {
                                        "x", "y", "z" -> {
                                            val createdNDArray = mappingContext.tensorInputFor(argDescriptor.name).toNd4jNDArray()

                                            ReflectionUtils.setField(field, df, createdNDArray)
                                            val variable = createVariable(varName = argDescriptor.name,
                                                    shape = createdNDArray.shape().toList(), dataType = createdNDArray.dataType(),
                                                    sameDiff = sd,
                                                    varType = VariableType.ARRAY)
                                            //add var to graph

                                        }
                                        "keepDims" -> ReflectionUtils.setField(field, df, argDescriptor.boolValue)
                                        else -> {
                                        }
                                    }

                                }
                            }

                        }
                    }


                    //DType calculate for output variables (set/correct if necessary)
                    val newInNames = sd.ops[name]!!.inputsToOp //Just in case import has modified this, like for concat case
                    val newInDtypes: MutableList<org.nd4j.linalg.api.buffer.DataType> = ArrayList(newInNames.size)
                    if (df is Merge) {
                        //Merge op: as noted elsewhere, we allow merge to be processed when only one of the inputs is available
                        // to break cycles for loops
                        //We know that Merge op has the restriction of the same datatype for both inputs, so we'll
                        val v1 = sd.getVariable(newInNames[0])
                        val v2 = sd.getVariable(newInNames[1])
                        val dt1 = if (v1 == null) v2!!.dataType() else v1.dataType()
                        val dt2 = if (v2 == null) v1!!.dataType() else v2.dataType()
                        newInDtypes.add(dt1)
                        newInDtypes.add(dt2)
                    } else {
                        for (s in newInNames) {
                            val v = sd.getVariable(s)
                            newInDtypes.add(v.dataType())
                        }
                    }

                    /**
                     * TODO: Note in order to generalize variable management, we may want to
                     * pull in a "variable accessor" for tensorflow and onnx.
                     *
                     * Tensorflow is found in constants, onnx is found in an initializers section of the actual
                     * proto file.
                     *
                     */
                    val outDTypes = df.calculateOutputDataTypes(newInDtypes)
                    val outSDVars = arrayOfNulls<SDVariable>(outDTypes.size)
                    val outVars = arrayOfNulls<Variable>(outDTypes.size)
                    val outNames: MutableList<String> = ArrayList(outDTypes.size)

                    //Create output variables and add to graph
                    for (i in outDTypes.indices) {
                        val dt = outDTypes[i]
                        val varName = name + if (i == 0) "" else ":$i"
                        //TODO: handle variadic type in kotlin
                        outSDVars[i] = sd.`var`(varName, VariableType.ARRAY, null, dt)
                        outNames.add(varName)
                        outVars[i] = Variable.builder()
                                .name(varName)
                                .variable(outSDVars[i])
                                .inputsForOp(null) //This is updated incrementally as other ops are added
                                .controlDepsForOp(null) //Control deps are handled later
                                .controlDepsForVar(null)
                                .outputOfOp(name)
                                .build()
                        sd.variables[varName] = outVars[i]
                        println("Added variable to graph: $varName (output of op $name)")
                    }
                    sd.ops[name]!!.outputsOfOp = outNames
                    println("Imported op: $opName (name=$name)")
                }
            } else {
                //Import override case
                val o = importOverride[name]
                println("Importing op $opName using override $importOverride")

                //First, get inputs:
                val inputs: MutableList<SDVariable> = ArrayList(nIn)
                var controlDeps: MutableList<SDVariable?>? = null
                for (i in 0 until nIn) {
                    val inName = nd.getInput(i)
                    val controlDep = isControlDep(inName)
                    val v = sd.getVariable(name)
                    if (controlDep) {
                        if (controlDeps == null) controlDeps = ArrayList()
                        controlDeps.add(v)
                    } else {
                        inputs.add(v)
                    }
                    o!!.initFromTensorFlow(inputs, controlDeps, nd, sd, nd.attrMap, tfGraph)
                }
            }
        }


        //Now that we have just added an op (or variable) - check what this feeds into, and see what we can now process
        // as a result
        if (nodeInputTo.containsKey(name)) {
            val set: Set<String>? = nodeInputTo[name]
            for (nextOp in set!!) {
                val nextOpDef = remainingNodes[nextOp]
                if (nextOpDef == null) {
                    if (sd.ops.containsKey(nextOp)) {
                        //Already processed this.
                        //Almost certainly the close of a loop - like NextIteration -> Merge case
                        continue
                    }
                    throw IllegalStateException("Could not find op definition for op to import: $nextOp")
                }
                val nInNext = nextOpDef.inputCount
                var allAlreadyInGraph = true
                var nonControlSeenCount = 0
                for (i in 0 until nInNext) {
                    val s = nextOpDef.getInput(i)
                    var inName = stripControl(nextOpDef.getInput(i))
                    if (inName.endsWith(":0")) {
                        //Strip ":0" suffix. Some ops can depend on placeholders, like "image_tensor:0" but in SameDiff this is a variable called "image_tensor"
                        inName = inName.substring(0, inName.length - 2)
                    }

//                        log.info("Input: {}, {}", s, inName);
                    if (!sd.hasVariable(inName) && !skipCase) {
//                            log.info("Not found: {} for op {}", inName, nextOpDef.getName());
                        allAlreadyInGraph = false
                        break
                    } else if (!isControlDep(s)) {
                        nonControlSeenCount++
                    }
                }

                //Merge ops are an edge case. We'll allow these to be executed with just ONE input, to break
                // the cycle in loops. In loops, generally we have (Enter, NextIteration) -> Merge, which
                // of course can't be done if we strictly require all inputs to be available
                val mergeCase = nonControlSeenCount > 0 && "Merge" == nextOpDef.op
                if (allAlreadyInGraph || mergeCase) {
                    //Can process this op, add it to the queue for processing
                    if (!availableToAddSet.contains(nextOp)) {
                        //Avoid processing same op multiple times, for repeated inputs to one op, etc
                        availableToAdd.add(nextOpDef)
                        availableToAddSet.add(nextOp)
                        println("Added to processing queue: ${nextOpDef.op} (name=$nextOp)")
                    }
                }
            }
        }

        //Finally, remove the just processed op from remainingNodes map:
        remainingNodes.remove(name)
    }

    //Post process the control dependencies, if any (done after because dependencies may not exist when imported)
    for ((varName, cdOpNames) in constControlDeps) {
        sd.variables[varName]!!.controlDeps = cdOpNames
        for (s in cdOpNames) {
            val sdo = sd.ops[s]
            if (sdo!!.controlDepFor == null) sdo.controlDepFor = ArrayList()
            val l = sdo.controlDepFor
            if (!l.contains(s)) l.add(varName)
        }
    }

    //Post process the merge ops - all we are missing is a Variable.getInputsForOp().add(mergeOpName);
    for ((key, value) in mergeOpsPostProcess) {
        val v = sd.variables[value]
        if (v!!.inputsForOp == null) v.inputsForOp = ArrayList()
        v.inputsForOp.add(key)
    }
    Preconditions.checkState(remainingNodes.isEmpty(), "%s Unprocessed nodes: %s", remainingNodes.size, remainingNodes.keys)
    return sd
}


/**
 * Get the shape from a TensorShapeProto
 *
 * @param tensorShapeProto Shape
 * @return Shape as long[]
 */
private fun shapeFromShapeProto(tensorShapeProto: TensorShapeProto): LongArray? {
    val shape = LongArray(tensorShapeProto.dimList.size)
    for (i in shape.indices) {
        shape[i] = tensorShapeProto.getDim(i).size
    }
    return shape
}

/**
 * Convert from TF proto datatype to ND4J datatype
 *
 * @param tfType TF datatype
 * @return ND4J datatype
 */
fun convertType(tfType: DataType?): org.nd4j.linalg.api.buffer.DataType {
    return when (tfType) {
        DataType.DT_DOUBLE -> org.nd4j.linalg.api.buffer.DataType.DOUBLE
        DataType.DT_FLOAT -> org.nd4j.linalg.api.buffer.DataType.FLOAT
        DataType.DT_HALF -> org.nd4j.linalg.api.buffer.DataType.HALF
        DataType.DT_BFLOAT16 -> org.nd4j.linalg.api.buffer.DataType.BFLOAT16
        DataType.DT_INT8 -> org.nd4j.linalg.api.buffer.DataType.BYTE
        DataType.DT_INT16 -> org.nd4j.linalg.api.buffer.DataType.SHORT
        DataType.DT_INT32 -> org.nd4j.linalg.api.buffer.DataType.INT
        DataType.DT_INT64 -> org.nd4j.linalg.api.buffer.DataType.LONG
        DataType.DT_UINT8 -> org.nd4j.linalg.api.buffer.DataType.UBYTE
        DataType.DT_STRING -> org.nd4j.linalg.api.buffer.DataType.UTF8
        DataType.DT_BOOL -> org.nd4j.linalg.api.buffer.DataType.BOOL
        else -> org.nd4j.linalg.api.buffer.DataType.UNKNOWN
    }
}

/**
 * @return True if the specified name represents a control dependency (starts with "^")
 */
fun isControlDep(name: String): Boolean {
    return name.startsWith("^")
}

/**
 * @return The specified name without the leading "^" character (if any) that appears for control dependencies
 */
fun stripControl(name: String): String {
    return if (name.startsWith("^")) {
        name.substring(1)
    } else name
}

/**
 * Remove the ":1" etc suffix for a variable name to get the op name
 *
 * @param varName Variable name
 * @return Variable name without any number suffix
 */
fun stripVarSuffix(varName: String): String {
    if (varName.matches(regex = Regex(".*:\\d+"))) {
        val idx = varName.lastIndexOf(':')
        return varName.substring(0, idx)
    }
    return varName
}

/**
 * Convert the tensor to an NDArray (if possible and if array is available)
 *
 * @param node Node to get NDArray from
 * @return NDArray
 */
fun getNDArrayFromTensor(node: NodeDef): INDArray? {
    //placeholder of some kind
    if (!node.attrMap.containsKey("value")) {
        return null
    }
    val tfTensor = node.getAttrOrThrow("value").tensor
    return mapTensorProto(tfTensor)
}

/**
 * Convert a TensorProto to an INDArray
 *
 * @param tfTensor Tensor proto
 * @return INDArray
 */
fun mapTensorProto(tfTensor: TensorProto): INDArray {
    val m = TFTensorMappers.newMapper(tfTensor) ?: throw RuntimeException("Not implemented datatype: " + tfTensor.dtype)
    return m.toNDArray()
}





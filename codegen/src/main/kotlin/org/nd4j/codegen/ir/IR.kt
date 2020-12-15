package org.nd4j.codegen.ir

import io.github.classgraph.ClassGraph
import org.apache.commons.io.IOUtils
import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.VariableType
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.autodiff.samediff.internal.Variable
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.codegen.ir.tensorflow.*
import org.nd4j.common.base.Preconditions
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.io.ReflectionUtils
import org.nd4j.common.util.ArrayUtil
import org.nd4j.gen.OpDeclarationDescriptor
import org.nd4j.imports.converters.DifferentialFunctionClassHolder
import org.nd4j.imports.graphmapper.OpImportFilter
import org.nd4j.imports.graphmapper.OpImportOverride
import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.Op
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge
import org.nd4j.linalg.api.shape.Shape
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.shade.protobuf.ByteString
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.nd4j.shade.protobuf.TextFormat
import java.lang.IllegalArgumentException
import java.nio.ByteBuffer
import java.nio.charset.Charset
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.HashSet


fun loadNd4jOpDescriptors(): OpNamespace.OpDescriptorList {
    val nd4jOpDescriptorResourceStream = ClassPathResource("nd4j-op-defs-2.proto").inputStream
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

fun nd4jDifferentialFunctions(): List<Class<out DifferentialFunction>> {
    return ClassGraph().enableAllInfo()
        .scan().getSubclasses("org.nd4j.autodiff.functions.DifferentialFunction").filter {
                clazz-> !clazz.isAbstract && !clazz.isAnnotation && !clazz.isInterface
        }.map { clazz -> Class.forName(clazz.name) as Class<out DifferentialFunction> }
}

val differentialFunctionClasses = nd4jDifferentialFunctions()

fun cachedOpInstances2(): List<DifferentialFunction> {
    return differentialFunctionClasses.map { clazz -> clazz.newInstance() as DifferentialFunction}.filter {
        it.opName() != null
    }
}

val cachedOpInstances = cachedOpInstances2()


fun createDifferentialFunctionInstanceForName(name: String): DifferentialFunction {
    return cachedOpInstances.first { op -> op.opName() == name }.javaClass.newInstance()
}

fun isOutputFrameworkAttributeName(name: String,opDescriptor: OpNamespace.OpDescriptor): Boolean {
    return opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType != OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
            && argDescriptor.argType != OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR }
        .map { inputArg -> inputArg.name }.contains(name)
}

fun isNd4jTensorName(name: String,opDescriptor: OpNamespace.OpDescriptor): Boolean {
    return opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
        .map { inputArg -> inputArg.name }
        .contains(name)
}


fun argDescriptorType(name: String, opDescriptor: OpNamespace.OpDescriptor): OpNamespace.ArgDescriptor.ArgType {
    return opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.name == name }[0].argType
}

val nd4jOpDescriptors = loadNd4jOpDescriptors()

fun OpNamespace.OpDescriptorList.findOp(opName: String): OpNamespace.OpDescriptor {
    return this.opListList.first { it.name == opName }
}


interface IRTensor<TENSOR_TYPE: GeneratedMessageV3, DATA_TYPE>
        where  DATA_TYPE: ProtocolMessageEnum {
    fun shape(): List<Long>
    fun stride(): List<Long>
    fun dataType(): IRDataType<DATA_TYPE>
    fun toArgTensor(): TensorNamespace.TensorProto
    fun rawValue(): TENSOR_TYPE
    fun toNd4jNDArray(): INDArray

}


enum class AttributeValueType {
    FLOAT,
    LIST_FLOAT,
    INT,
    LIST_INT,
    BOOL,
    LIST_BOOL,
    STRING,
    LIST_STRING,
    TENSOR,
    LIST_TENSOR,
    DATA_TYPE,
    INVALID
}

interface IRAttribute<ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE : ProtocolMessageEnum> {

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

    fun dataTataTypeValue(): IRDataType<DATA_TYPE>

    fun internalAttributeDef(): ATTRIBUTE_TYPE


    fun internalAttributeValue(): ATTRIBUTE_VALUE_TYPE
}



interface MappingProcess<
        GRAPH_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> {



    fun opName(): String

    fun frameworkVersion(): String

    fun inputFramework(): String

    fun inputFrameworkOpName(): String

    fun attributeMappingRules(): List<AttributeMappingRule<GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_DEF_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            TENSOR_TYPE,
            DATA_TYPE>>

    fun tensorMappingRules():  List<TensorMappingRule<GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_DEF_TYPE,
            ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>

    fun applyProcess(mappingCtx: MappingContext<GRAPH_TYPE,NODE_DEF_TYPE, OP_DEF_TYPE,
            TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE,DATA_TYPE>):
            Pair<MappingContext<GRAPH_TYPE,
                    NODE_DEF_TYPE,
                    OP_DEF_TYPE,
                    TENSOR_TYPE, ATTRIBUTE_TYPE,
                    ATTRIBUTE_VALUE_TYPE,
                    DATA_TYPE>, OpNamespace.OpDescriptor>

    fun applyProcessReverse(input: OpDeclarationDescriptor): IRNode<NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>

    fun serialize(): MapperNamespace.MapperDeclaration


}


interface TensorMappingRule<GRAPH_TYPE: GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,NODE_DEF_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {


    fun initWithMappingProcess(mappingProcess: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>)


    fun name(): String


    fun serialize(): MapperNamespace.MappingRule


    fun mappingNamesToPerform(): Map<String,String>

    /**
     * Convert 1 or more attributes in to a list of {@link ArgDescriptor}
     */
    fun convertInput(mappingContext: MappingContext<GRAPH_TYPE,NODE_DEF_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor>


    fun inputArgumentMappings(): Map<String, String>

    fun convertInputsReverse(toReverse: List<OpNamespace.ArgDescriptor>): List<TENSOR_TYPE>

    fun isInputTensorName(inputName: String): Boolean

    fun isOutputTensorName(outputName: String): Boolean

}


interface AttributeMappingRule<GRAPH_TYPE: GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,NODE_DEF_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    fun initWithMappingProcess(mappingProcess: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>)

    fun mappingNamesToPerform(): Map<String,String>

    fun mappingTransformerArgs(): Map<String, List<OpNamespace.ArgDescriptor>>

    fun name(): String

    fun serialize(): MapperNamespace.MappingRule

    fun convertAttributes(mappingCtx: MappingContext<GRAPH_TYPE,NODE_DEF_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): List<OpNamespace.ArgDescriptor>

    fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>


    fun isInputFrameworkTensorName(name: String,mappingProcess: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): Boolean

    fun isNd4jTensorName(name: String,mappingProcess: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): Boolean

    fun isInputFrameworkAttributeName(name: String,mappingProcess: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): Boolean

    fun isOutputFrameworkAttributeName(name: String,mappingProcess: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): Boolean

    fun argDescriptorType(name: String,mappingProcess: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): OpNamespace.ArgDescriptor.ArgType

    fun acceptsInputType(argDescriptorType: AttributeValueType): Boolean

    fun outputsType(argDescriptorType: List<OpNamespace.ArgDescriptor.ArgType>): Boolean

    fun attributeValueTypeFor(name: String,mappingProcess: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): AttributeValueType

    fun argDescriptorTypesForOutputName(
        name: String, mappingProcess:
        MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,
                ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): List<OpNamespace.ArgDescriptor.ArgType>
}


interface MappingContext<GRAPH_TYPE: GeneratedMessageV3,NODE_TYPE: GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE: GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,DATA_TYPE: ProtocolMessageEnum> {
    fun node(): NODE_TYPE

    fun opDef(): OP_DEF_TYPE

    fun opName(): String

    fun nodeName(): String

    fun attrDef(name: String): ATTRIBUTE_TYPE

    fun tensorInputFor(name: String): IRTensor<TENSOR_TYPE,DATA_TYPE>

    fun tensorAttributeFor(name: String): IRTensor<TENSOR_TYPE,DATA_TYPE>


    fun createIRTensorFromNDArray(ndaray:INDArray): IRTensor<TENSOR_TYPE,DATA_TYPE>

    fun nd4jDataTypeFor(input: IRTensor<TENSOR_TYPE,DATA_TYPE>): DataType

    fun irAttributeValueForNode(valueName: String): IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>

    fun argDescriptorTypeForName(nd4jName: String): OpNamespace.ArgDescriptor.ArgType

    fun graph(): IRGraph<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,DATA_TYPE>

    fun nd4jOpName(): String

}

abstract class AbstractMappingContext<GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE: GeneratedMessageV3,
        ATTRIBUTE_TYPE: GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum>(opDef: OP_DEF_TYPE,
                                        node: NODE_TYPE,
                                        graph:
                                        IRGraph<GRAPH_TYPE,
                                                NODE_TYPE,
                                                OP_DEF_TYPE,
                                                TENSOR_TYPE,
                                                ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>):
    MappingContext<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE> {

    val opDef = opDef
    val node = node
    val graph = graph

    override fun node(): NODE_TYPE {
        return node
    }

    override fun opDef(): OP_DEF_TYPE {
        return opDef
    }

    override fun graph(): IRGraph<GRAPH_TYPE,NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        return graph
    }

    override fun argDescriptorTypeForName(nd4jName: String): OpNamespace.ArgDescriptor.ArgType {
        val opDescriptor = nd4jOpDescriptors.findOp(graph.nd4jNameForInternalOpName(opName()))
        return opDescriptor.argDescriptorList.first { argDescriptor -> argDescriptor.name == nd4jName }!!.argType
    }

    override fun nd4jOpName(): String {
        return nd4jOpDescriptors.findOp(graph.nd4jNameForInternalOpName(opName())).name
    }
}


interface IRGraphRunner<
        GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE: GeneratedMessageV3,
        ATTRIBUTE_TYPE: GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,
        DATA_TYPE : ProtocolMessageEnum> {

    fun graph(): IRGraph<GRAPH_TYPE,
            NODE_TYPE,
            OP_DEF_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE>

    fun run(inputs: Map<String,INDArray>): Map<String,INDArray>
}

interface IRGraph<
        GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE: GeneratedMessageV3,
        ATTRIBUTE_TYPE: GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,
        DATA_TYPE : ProtocolMessageEnum> {

    fun shapeOfInput(varName: String): LongArray?

    fun dataTypeForVariable(varName: String): IRDataType<DATA_TYPE>

    fun isConstant(opName: String): Boolean

    fun isPlaceHolder(opName: String): Boolean

    fun isConstantOpName(name: String): Boolean

    fun nodeByName(input: String): NODE_TYPE

    fun nodeList(): List<IRNode<NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>>

    fun internalValue(): GRAPH_TYPE

    fun createMappingContext(opDef: OP_DEF_TYPE, node: NODE_TYPE): MappingContext<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>

    fun frameworkName(): String

    fun nd4jNameForInternalOpName(name: String): String
}

interface IRNode<NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE>
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

    fun numInputs(): Int

    fun numOutputs(): Int

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

interface IROpDef<
        GRAPH_DEF: GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ARG_DEF_TYPE : GeneratedMessageV3, DATA_TYPE,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3>
        where DATA_TYPE: ProtocolMessageEnum  {
    fun opName(): String

    fun internalValue(): OP_DEF_TYPE

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

    fun nd4jDataType(): DataType

    fun nameSpaceDataType(): TensorNamespace.DataType
}



abstract  class AbstractMappingProcess<
        GRAPH_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(inputFramework: String,
                                                                                   frameworkVersion: String,
                                                                                   inputFrameworkOpName: String,
                                                                                   opName: String,
                                                                                   opMappingRegistry: OpMappingRegistry<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,DATA_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE>,
                                                                                   tensorMappingRules: List<out TensorMappingRule<GRAPH_TYPE,OP_DEF_TYPE,NODE_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>,
                                                                                   attributeMappingRules: List<out AttributeMappingRule<GRAPH_TYPE,OP_DEF_TYPE,NODE_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>):
    MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,
            NODE_TYPE,TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,DATA_TYPE> {

    protected val inputFramework = inputFramework
    protected val frameworkVersion = frameworkVersion
    protected val inputFrameworkOpName = inputFrameworkOpName
    protected val opName = opName
    protected val tensorMappingRules = tensorMappingRules
    protected val attributeMappingRules = attributeMappingRules
    protected var opDef: IROpDef<GRAPH_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,DATA_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE>? = null
    protected val opMappingRegistry = opMappingRegistry

    init {
        tensorMappingRules.forEach { tensorMappingRule ->
            tensorMappingRule.initWithMappingProcess(this)
            tensorMappingRule.mappingNamesToPerform().forEach { (nd4jName, inputFrameworkName) ->
                if(!tensorMappingRule.isInputTensorName(inputFrameworkName)) {
                    throw IllegalArgumentException("Found invalid input tensor named ${inputFrameworkName} for rule ${tensorMappingRule.name()} and mapping process for op ${opName} and input framework name ${inputFrameworkOpName}")
                }

                if(!tensorMappingRule.isOutputTensorName(nd4jName)) {
                    throw IllegalArgumentException("Found invalid output tensor named ${nd4jName} for rule ${tensorMappingRule.name()} and mapping process for op ${opName} and input framework name ${inputFrameworkOpName}")
                }

            }
        }

        attributeMappingRules.forEach {
            it.initWithMappingProcess(this)
            attributeMappingRules.forEach { attributeMappingRule ->
                attributeMappingRule.mappingNamesToPerform().forEach { (nd4jName, inputFrameworkName) ->
                    val inputType = attributeMappingRule.attributeValueTypeFor(inputFrameworkName,this)
                    if(!attributeMappingRule.acceptsInputType(inputType)) {
                        throw IllegalArgumentException("Rule ${attributeMappingRule.name()} for framework $inputFramework does not accept input type ${inputType} for attribute name ${inputFrameworkName} and mapping process for op ${opName} and input framework name ${inputFrameworkOpName}")
                    }

                    val outputType = attributeMappingRule.argDescriptorTypesForOutputName(nd4jName,this)
                    if(!attributeMappingRule.outputsType(outputType)) {
                        throw IllegalArgumentException("Rule ${attributeMappingRule.name()} for framework $inputFramework does not accept output type ${outputType} for attribute name ${nd4jName} and mapping process for op ${opName}")
                    }

                }
            }
        }


        opMappingRegistry.registerMappingProcess(
            inputFrameworkOpName = inputFrameworkOpName,
            processToRegister = this
        )


    }


    override fun attributeMappingRules(): List<AttributeMappingRule<GRAPH_TYPE,OP_DEF_TYPE,NODE_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>> {
        return attributeMappingRules
    }

    override fun tensorMappingRules(): List<TensorMappingRule<GRAPH_TYPE,OP_DEF_TYPE,NODE_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>> {
        return tensorMappingRules
    }

    override fun applyProcessReverse(input: OpDeclarationDescriptor): IRNode<NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        TODO("Not yet implemented")
    }

    override fun inputFrameworkOpName(): String {
        return inputFrameworkOpName
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

    override fun applyProcess(mappingCtx: MappingContext<GRAPH_TYPE,
            NODE_TYPE,
            OP_DEF_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): Pair<MappingContext<
            GRAPH_TYPE,
            NODE_TYPE,
            OP_DEF_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE>, OpNamespace.OpDescriptor> {
        val descriptorBuilder = OpNamespace.OpDescriptor.newBuilder()
        descriptorBuilder.name = opName()
        tensorMappingRules.forEach {
            it.convertInput(mappingCtx).forEach { descriptor ->
                descriptorBuilder.addArgDescriptor(descriptor)
            }
        }


        attributeMappingRules.forEach {
            it.convertAttributes(mappingCtx).forEach {
                    descriptor -> descriptorBuilder.addArgDescriptor(descriptor)
            }
        }

        val fullDescriptor = nd4jOpDescriptors.findOp(opName())
        descriptorBuilder.opDeclarationType = fullDescriptor.opDeclarationType

        return Pair(mappingCtx,descriptorBuilder.build())
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


fun ArgDescriptor(block: OpNamespace.ArgDescriptor.Builder.() -> Unit): OpNamespace.ArgDescriptor {
    return OpNamespace.ArgDescriptor.newBuilder()
        .apply(block).build()
}

fun NameSpaceTensor(block: TensorNamespace.TensorProto.Builder.() -> Unit): TensorNamespace.TensorProto {
    return TensorNamespace.TensorProto.newBuilder()
        .apply(block).build()
}



fun TensorNamespace.TensorProto.Builder.RawData(rawData: ByteArray) {
    this.rawData = ByteString.copyFrom(rawData)
}

fun TensorNamespace.TensorProto.Builder.IntData(intData: List<Int>) {
    this.addAllInt32Data(intData)
}

fun TensorNamespace.TensorProto.Builder.FloatData(floatData: List<Float>) {
    this.addAllFloatData(floatData)
}

fun TensorNamespace.TensorProto.Builder.DoubleData(doubleData: List<Double>) {
    this.addAllDoubleData(doubleData)
}

fun TensorNamespace.TensorProto.Builder.Int64Data(intData: List<Long>) {
    this.addAllInt64Data(intData)
}

fun TensorNamespace.TensorProto.Builder.Dims(shape: List<Long>) {
    shape.forEach { this.addDims(it) }
}


fun convertNd4jDataTypeFromNameSpaceTensorDataType(dataType: TensorNamespace.DataType): DataType {
    return when(dataType) {
        TensorNamespace.DataType.UINT32 -> return DataType.UINT32
        TensorNamespace.DataType.INT64 -> return DataType.INT64
        TensorNamespace.DataType.INT16 -> return DataType.INT16
        TensorNamespace.DataType.UINT64 ->  return DataType.UINT64
        TensorNamespace.DataType.DOUBLE ->  return DataType.DOUBLE
        TensorNamespace.DataType.FLOAT ->  return DataType.FLOAT
        TensorNamespace.DataType.FLOAT16 ->  return DataType.FLOAT16
        TensorNamespace.DataType.FLOAT16 -> return  DataType.FLOAT16
        TensorNamespace.DataType.INT32,DataType.INT ->  return DataType.INT32
        TensorNamespace.DataType.STRING ->  return DataType.UTF8
        TensorNamespace.DataType.BOOL -> return  DataType.BOOL
        TensorNamespace.DataType.BFLOAT16 -> return  DataType.BFLOAT16
        TensorNamespace.DataType.INT8 -> return DataType.INT8
        TensorNamespace.DataType.UINT16 -> return DataType.UINT16
        TensorNamespace.DataType.UNDEFINED,TensorNamespace.DataType.UNRECOGNIZED -> return DataType.UNKNOWN
        else -> {
            throw IllegalArgumentException("Illegal data type $dataType")
        }
    }
}

fun convertNameSpaceTensorDataTypeFromNd4jDataType(dataType: DataType): TensorNamespace.DataType {
    return when(dataType) {
        DataType.UINT32 ->  return TensorNamespace.DataType.UINT32
        DataType.INT64,DataType.LONG ->  return TensorNamespace.DataType.INT64
        DataType.UINT64 ->  return TensorNamespace.DataType.UINT64
        DataType.DOUBLE ->  return TensorNamespace.DataType.DOUBLE
        DataType.FLOAT ->  return TensorNamespace.DataType.FLOAT
        DataType.FLOAT16,DataType.HALF ->  return TensorNamespace.DataType.FLOAT16
        DataType.HALF -> return  TensorNamespace.DataType.FLOAT16
        DataType.INT32,DataType.INT ->  return TensorNamespace.DataType.INT32
        DataType.UTF8 ->  return TensorNamespace.DataType.STRING
        DataType.BOOL -> return  TensorNamespace.DataType.BOOL
        DataType.BFLOAT16 -> return  TensorNamespace.DataType.BFLOAT16
        DataType.SHORT,DataType.INT8 -> return TensorNamespace.DataType.INT8
        DataType.UINT16 -> return TensorNamespace.DataType.UINT16
        DataType.BYTE,DataType.UINT8,DataType.UBYTE -> return TensorNamespace.DataType.UINT8
        else -> {
            throw IllegalArgumentException("Illegal data type $dataType")
        }
    }
}


fun ndarrayFromNameSpaceTensor(inputTensor: TensorNamespace.TensorProto): INDArray {
    val dtype = convertNd4jDataTypeFromNameSpaceTensorDataType(TensorNamespace.DataType.values()[inputTensor.dataType])
    val shape = inputTensor.dimsList.toLongArray()
    when(dtype) {
        DataType.FLOAT -> {
            val floatArray = inputTensor.floatDataList.toFloatArray()
            if(floatArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)
            val dataBuffer = Nd4j.createBuffer(floatArray)
            return Nd4j.create(dataBuffer).reshape(*shape)
        }
        DataType.DOUBLE -> {
            val doubleArray = inputTensor.doubleDataList.toDoubleArray()
            if(doubleArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)

            val dataBuffer = Nd4j.createBuffer(doubleArray)
            return Nd4j.create(dataBuffer).reshape(*shape)
        }
        DataType.INT64 -> {
            val longArray = inputTensor.int64DataList.toLongArray()
            if(longArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)
            val dataBuffer = Nd4j.createBuffer(longArray)
            return Nd4j.create(dataBuffer).reshape(*shape)
        }
        DataType.INT32 -> {
            val intArray = inputTensor.int32DataList.toIntArray()
            if(intArray.isEmpty())
                return loadDataBufferFromRawData(inputTensor)

            val dataBuffer = Nd4j.createBuffer(intArray)
            return Nd4j.create(dataBuffer).reshape(*shape)
        }
        DataType.UNKNOWN -> {
            val ret =  Nd4j.empty()
            return ret
        }

        else -> {
            return loadDataBufferFromRawData(inputTensor)
        }

    }

    throw IllegalArgumentException("Illegal type found for conversion ${dtype}")
}

fun loadDataBufferFromRawData(inputTensor: TensorNamespace.TensorProto): INDArray {
    val shape = inputTensor.dimsList.toLongArray()
    val dtype = convertNd4jDataTypeFromNameSpaceTensorDataType(TensorNamespace.DataType.values()[inputTensor.dataType])
    val byteArray = inputTensor.rawData.toByteArray()
    val totalLen = ArrayUtil.prod(*shape)
    val byteBuffer = ByteBuffer.allocateDirect(totalLen * dtype.width())
    byteBuffer.put(byteArray)
    byteBuffer.rewind()
    val rawDataBuffer = Nd4j.createBuffer(byteBuffer,dtype,totalLen,0)
    return Nd4j.create(rawDataBuffer).reshape(*shape)
}



fun nameSpaceTensorFromNDarray(ndarray:INDArray): TensorNamespace.TensorProto {
    val nameSpaceDataType = convertNameSpaceTensorDataTypeFromNd4jDataType(ndarray.dataType()).ordinal
    when(ndarray.dataType()) {
        DataType.INT64 -> {
            return NameSpaceTensor {
                dataType = nameSpaceDataType
                Int64Data(ndarray.data().asLong().toList())
                Dims(ndarray.shape().asList())
            }
        }

        DataType.INT32 -> {
            return NameSpaceTensor {
                dataType = nameSpaceDataType
                IntData(ndarray.data().asInt().toList())
                Dims(ndarray.shape().asList())
            }
        }

        DataType.DOUBLE -> {
            return NameSpaceTensor {
                dataType = nameSpaceDataType
                DoubleData(ndarray.data().asDouble().toList())
                Dims(ndarray.shape().asList())
            }
        }

        DataType.FLOAT -> {
            return NameSpaceTensor {
                dataType = nameSpaceDataType
                FloatData(ndarray.data().asFloat().toList())
                Dims(ndarray.shape().asList())
            }
        }

        else -> {
            throw IllegalArgumentException("Illegal data type ${ndarray.dataType()}")
        }
    }

}




interface ImportContext<
        GRAPH_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> {

    fun process(): MappingProcess<GRAPH_TYPE,OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>

    fun mappingContext(): MappingContext<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>

}

abstract class AbstractImportContext<
        GRAPH_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum>
    (process: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>,
     mappingContext: MappingContext<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,
             ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>):
    ImportContext<GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>
{
    val process = process
    val mappingContext = mappingContext

    override fun process(): MappingProcess<
            GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE> {
        return process
    }

    override fun mappingContext(): MappingContext<GRAPH_TYPE,
            NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE,
            ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        return mappingContext
    }
}


interface ImportProcess<
        GRAPH_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> {

    fun createMappingProcesses(graph: IRGraph<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,
            TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>)
            : List<MappingProcess<GRAPH_TYPE,
            OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>>

    fun createMappingContext(graph:
                             IRGraph<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,
                                     ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,
                                     DATA_TYPE>,node: IRNode<NODE_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE>):
            MappingContext<GRAPH_TYPE,NODE_TYPE,
                    OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>


    fun createImportContext(mappingProcess: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>,mappingContext:
    MappingContext<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>)
            : ImportContext<GRAPH_TYPE,OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>

    fun runImportProcess(mappingProcesses: List<ImportContext<GRAPH_TYPE,OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>>): SameDiff

}

abstract class AbstractImportProcess<
        GRAPH_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        TENSOR_TYPE: GeneratedMessageV3,
        ATTRIBUTE_TYPE: GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum>
    (inputFramework: String):
    ImportProcess<GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE> {

    val inputFramework = inputFramework


    override fun createMappingProcesses(graph: IRGraph<GRAPH_TYPE,NODE_TYPE,
            OP_DEF_TYPE, TENSOR_TYPE,
            ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>):
            List<MappingProcess<GRAPH_TYPE,
                    OP_DEF_TYPE,
                    NODE_TYPE,
                    TENSOR_TYPE,
                    ATTRIBUTE_TYPE,
                    ATTRIBUTE_VALUE_TYPE,
                    DATA_TYPE>> {
        return graph.nodeList().map {
            val mappingContext = createMappingContext(graph = graph, node = it)
            val opName = mappingContext.opName()
            OpRegistryHolder.lookupOpMappingProcess<
                    GRAPH_TYPE,
                    NODE_TYPE,
                    OP_DEF_TYPE,
                    TENSOR_TYPE,
                    DATA_TYPE,
                    ATTRIBUTE_TYPE,
                    ATTRIBUTE_VALUE_TYPE>(inputFrameworkOpName = opName, inputFrameworkName = inputFramework)
        }
    }


    override fun runImportProcess(importContexts: List<ImportContext<GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE>>): SameDiff {
        val sameDiff = SameDiff.create()
        importContexts.map {
            it.process().applyProcess(it.mappingContext())
        }.forEach {
                pair ->
            val variables2 = ArrayList<SDVariable>()
            val opDescriptor = pair.second
            val mappingContext = pair.first
            when(opDescriptor.opDeclarationType) {
                OpNamespace.OpDescriptor.OpDeclarationType.LEGACY_XYZ -> {
                    val createdOp = createDifferentialFunctionInstanceForName(opDescriptor.name)
                    opDescriptor.argDescriptorList.forEach {
                            argDescriptor ->
                        val field = ReflectionUtils.findField(createdOp.javaClass,argDescriptor.name)
                        field.isAccessible = true
                        when(argDescriptor.name) {
                            "x","y","z" ->  {
                                val createdNDArray = mappingContext.tensorInputFor(argDescriptor.name).toNd4jNDArray()

                                ReflectionUtils.setField(field,createdOp,createdNDArray)
                                val variable = createVariable(varName = argDescriptor.name, shape = createdNDArray.shape().toList(),dataType = createdNDArray.dataType(),sameDiff = sameDiff,varType = VariableType.ARRAY)
                                //add var to graph
                                sameDiff.`var`(variable)
                                variables2.add(variable)
                            }
                            "keepDims" ->  ReflectionUtils.setField(field,createdOp, argDescriptor.boolValue)
                            else -> { }
                        }

                    }

                    sameDiff.addArgsFor(variables2.toTypedArray(),createdOp)

                }

                OpNamespace.OpDescriptor.OpDeclarationType.CUSTOM_OP_IMPL -> {
                    val dynamicCustomOp = DynamicCustomOp.builder(opDescriptor.name)
                    opDescriptor.argDescriptorList.forEach {
                            argDescriptor ->
                        when(argDescriptor.argType) {
                            OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR -> {
                                val arr = mappingContext.tensorInputFor(argDescriptor.name)
                                val nd4jDataType = mappingContext.nd4jDataTypeFor(arr)
                                val variable = createVariable(varName = argDescriptor.name, shape = arr.shape(),dataType = nd4jDataType,sameDiff = sameDiff,varType = VariableType.ARRAY)
                                variables2.add(variable)
                                dynamicCustomOp.addInputs(arr.toNd4jNDArray())
                            }
                            OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR -> {

                            }

                            OpNamespace.ArgDescriptor.ArgType.INT32 -> {
                                dynamicCustomOp.addIntegerArguments(argDescriptor.int32Value)
                            }

                            OpNamespace.ArgDescriptor.ArgType.INT64 -> {
                                dynamicCustomOp.addIntegerArguments(argDescriptor.int64Value)
                            }

                            OpNamespace.ArgDescriptor.ArgType.FLOAT -> {
                                dynamicCustomOp.addFloatingPointArguments(argDescriptor.floatValue.toDouble())
                            }

                            OpNamespace.ArgDescriptor.ArgType.DOUBLE -> {
                                dynamicCustomOp.addFloatingPointArguments(argDescriptor.doubleValue)
                            }

                            OpNamespace.ArgDescriptor.ArgType.STRING -> {

                            }

                        }
                    }

                    sameDiff.addArgsFor(variables2.toTypedArray(),dynamicCustomOp.build())
                }
            }


        }

        return sameDiff

    }
}

fun lookupIndexForArgDescriptor(
    argDescriptorName: String,
    opDescriptorName: String,
    argDescriptorType: OpNamespace.ArgDescriptor.ArgType
): Int {
    println("Trying to find arg descriptor index for op $opDescriptorName and descriptor name $argDescriptorName with type $argDescriptorType")
    val op = nd4jOpDescriptors.findOp(opDescriptorName)
    val names = op.argDescriptorList.map { argDescriptor -> argDescriptor.name }
    if(!names.contains(argDescriptorName)) {
        throw IllegalArgumentException("Invalid name $argDescriptorName for op $opDescriptorName passed in. $argDescriptorName not found in $opDescriptorName. Available names were ${names}")
    }
    val ret =  op
        .argDescriptorList.firstOrNull { argDescriptor -> argDescriptor.name == argDescriptorName &&
                argDescriptor.argType == argDescriptorType }
    if(ret == null)
        return -1
    else return ret.argIndex
}

fun createVariable(varName: String,varType: VariableType,sameDiff: SameDiff,shape: List<Long>,dataType: DataType): SDVariable {
    return SDVariable(varName,varType, sameDiff, shape.toLongArray(), dataType)
}

/**
 * TODO: Refactor to make agnostic to TF or onnx
 * TODO: implement test reflecting this behavior
 * Focus on making testable version of this allowing for quick creation of all op
 * instances and verify that the number of expected parameters are present in each op list
 * as well as relative to op type ensuring at least input fields are implemented.
 * Also check for certain common attributes like dimensions, axis keepDims
 * and check those are set properly
 *
 * There are also different parameter type assertions we could do such as
 * list int -> int conversion having the right ints in place
 * int -> ndarray ensuring input is present
 * etc
 * Relative to the mappings we also need to ensure that each arg descriptor mapped
 * has the right name and right index mapped and it each arg descriptor is mapped to 1 to 1
 * in the tests. This would be different for each op type, but for custom ops maps 1 to 1 for each arg descriptor
 * mapping to 1 argument
 *
 * For certain classes of ops this means making sure that certain default values are preset to their correct values (eg: x,y, inPlace, keepDims,..)
 */
fun <GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTR_DEF_TYPE : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> initAttributes(
    df: DifferentialFunction,
    frameworkName: String,
    mappingContext: MappingContext<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>,
    sd: SameDiff,
    inputFrameworkOpName: String
) {

    val opMappingProcess =  OpRegistryHolder.lookupOpMappingProcess<
            GRAPH_TYPE,
            NODE_TYPE,
            OP_DEF_TYPE,
            TENSOR_TYPE,
            DATA_TYPE,
            ATTR_DEF_TYPE,
            ATTR_VALUE_TYPE>(inputFrameworkOpName = inputFrameworkOpName, inputFrameworkName = frameworkName)

    val applied = opMappingProcess.applyProcess(mappingContext)

    when (df.opType()) {
        Op.Type.CUSTOM -> {
            val dynamicCustomOp = df as DynamicCustomOp
            val grouped = applied.second.argDescriptorList.groupBy { descriptor ->
                descriptor.argType
            }

            val sortedMap = HashMap<OpNamespace.ArgDescriptor.ArgType, List<OpNamespace.ArgDescriptor>>()
            grouped.forEach { (argType, list) ->
                sortedMap[argType] = list.sortedBy { arg -> arg.argIndex }
            }

            sortedMap.forEach { (argType, listOfArgsSortedByIndex) ->
                when (argType) {
                    OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR -> {
                        val args = dynamicCustomOp.args()
                        val arraysToAdd = ArrayList<INDArray>()
                        listOfArgsSortedByIndex.forEachIndexed { index, argDescriptor ->
                            val convertedTensor = ndarrayFromNameSpaceTensor(argDescriptor.inputValue)
                            if(index < args.size) {
                                val arg = args[index]
                                if (arg.variableType != VariableType.ARRAY) {
                                    if (arg.shape == null) {
                                        val emptyLongArray = LongArray(0)
                                        arg.setShape(*emptyLongArray)
                                    }

                                    if (arg.shape.size != convertedTensor.rank() || !Shape.shapeEquals(arg.shape, convertedTensor.shape())) {
                                        println("Shape of constant argument ${arg.shape} did not match actual input tensor. Overriding shape to be the same shape as ${convertedTensor.shape()}")
                                        arg.setShape(*convertedTensor.shape())
                                        arg.setArray(convertedTensor)
                                        val variable = Variable.builder()
                                            .variable(arg)
                                            .name(arg.name())
                                            .build()
                                        sd.variables[arg.name()] = variable
                                        arraysToAdd.add(convertedTensor)
                                        df.args()
                                    } else {
                                        arraysToAdd.add(convertedTensor)
                                    }
                                }
                            } else {
                                sd.constant(sd.generateNewVarName(argDescriptor.name,0),convertedTensor)
                                arraysToAdd.add(convertedTensor)
                            }

                        }

                        //note we don't add arrays one at a time because addInputArgument requires all the input arrays to be added at once
                        //dynamicCustomOp.addInputArgument(*arraysToAdd.toTypedArray())


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
                            val tensorName = opMappingProcess.tensorMappingRules().filter { mappingRule -> mappingRule.mappingNamesToPerform()
                                .containsKey(argDescriptor.name)}
                                .map { rule -> rule.mappingNamesToPerform()[argDescriptor.name] }.first()!!
                            val createdNDArray = mappingContext.tensorInputFor(tensorName).toNd4jNDArray()
                            ReflectionUtils.setField(field, df, createdNDArray)
                        }
                        "keepDims" -> ReflectionUtils.setField(field, df, argDescriptor.boolValue)
                        else -> {
                        }
                    }

                }
            }

        }
    }
}


/**
 * Import a TensorFlow model from a GraphDef, with optional import overrides
 *
 * @param irGraph        TensorFlow model GraphDef
 * @param importOverride Optional import override for specific ops, keyed by op name
 * @param opFilter       Optional filter - ops to exclude/ignore
 * @return Imported model
 */
fun  <GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTR_DEF_TYPE : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum>
        importGraph(irGraph: IRGraph<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTR_DEF_TYPE,ATTR_VALUE_TYPE,DATA_TYPE>, importOverride: Map<String?, OpImportOverride<GRAPH_TYPE, NODE_TYPE, ATTR_VALUE_TYPE>?>?,
                    opFilter: OpImportFilter<GRAPH_TYPE,NODE_TYPE,ATTR_VALUE_TYPE>?): SameDiff? {

    /*
        First, build an in-memory representation of the graph that allows us to build the graph incrementally
        If we can build the graph incrementally, we can make sure that the added variables are set up with the correct
        datatype and (once implemented) greedy shape inference
         */
    val variableTracker = VariableStateTracker()
    val availableToAddSet: MutableSet<String> = HashSet() //TODO maybe unnecessary?
    val availableToAdd: Queue<IRNode<NODE_TYPE,TENSOR_TYPE,ATTR_DEF_TYPE,ATTR_VALUE_TYPE,DATA_TYPE>> = LinkedList()
    val remainingNodes: MutableMap<String, IRNode<NODE_TYPE,TENSOR_TYPE,ATTR_DEF_TYPE,ATTR_VALUE_TYPE,DATA_TYPE>> = HashMap() //All other nodes, not in availableToAdd
    val nodeInputTo: MutableMap<String, MutableList<String>> = HashMap() // For op x -> y, x is key, y is value. Note that these are OP names not VARIABLE names
    val nNodes = irGraph.nodeList().size

    //First, add any constants, placeholders, and zero-input ops
    val sd = SameDiff.create()
    val opRegistryHolder = if(irGraph.frameworkName() == "tensorflow")  OpRegistryHolder.tensorflow() else  OpRegistryHolder.onnx()
    irGraph.nodeList().forEach { node ->
        val nd4jOpName = opRegistryHolder.lookupOpMappingProcess(node.opName()).opName()
        variableTracker.addOp(nd4jOpName)
        val opDescriptor = opRegistryHolder.lookupNd4jOpDef(nd4jOpName)
        val opInputs = opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR  }
        val numInputs = node.numInputs()
        variableTracker.addOp(nd4jOpName)
        val nodeInputs = ArrayList<String>()
        val name = node.nodeName()

        for(inputIdx in 0 until numInputs) {
            var inOpName = if(inputIdx < numInputs) stripVarSuffix(stripControl(node.inputAt(inputIdx))) else variableTracker.generateNewVarName(opInputs[inputIdx].name,inputIdx)
            nodeInputs.add(inOpName)
            if (!nodeInputTo.containsKey(inOpName)) {
                nodeInputTo[inOpName!!] = ArrayList()

            }

            nodeInputTo[inOpName]!!.add(name)
            variableTracker.addVariable(name)
        }


    }

    for (i in 0 until nNodes) {
        val nd = irGraph.nodeList()[i]
        val op = nd.opName()
        val nd4jOpName = opRegistryHolder.lookupOpMappingProcess(op).opName()
        variableTracker.addOp(nd4jOpName)
        val opDescriptor = opRegistryHolder.lookupNd4jOpDef(nd4jOpName)
        val opInputs = opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR  }
        val numInputs = nd.numInputs()

        val name = nd.nodeName()
        Preconditions.checkState(name.isNotEmpty(),"Node name was empty!")
        if (irGraph.isConstantOpName(op)|| numInputs == 0) {
            availableToAdd.add(nd)
            availableToAddSet.add(name)
            variableTracker.addVariable(name!!)
        } else {
            remainingNodes[name] = nd

            for (inputIdx in 0 until numInputs) {
                /**
                 * Sometimes import and nd4j might not map 1 to 1.
                 * This handles the edge case where extra inputs may exist in nd4j
                 * but not from the input framework.
                 */
                var inOpName = if(inputIdx < numInputs) stripVarSuffix(stripControl(nd.inputAt(inputIdx))) else variableTracker.generateNewVarName(opInputs[i].name,i)
                variableTracker.addVariable(inOpName!!)

                if (!nodeInputTo.containsKey(inOpName)) {
                    nodeInputTo[inOpName!!] = ArrayList()
                }
                nodeInputTo[inOpName]!!.add(name)
                variableTracker.addVariable(name)
            }
        }
    }


    val mergeOpsPostProcess: MutableMap<String, String> = HashMap()

    //Go through ops in order, and add to the graph
    val constControlDeps: MutableMap<String, List<String>> = HashMap() //Key: constant name. Value: control dependencies
    while (!availableToAdd.isEmpty()) {
        val nd = availableToAdd.remove()
        val name = nd.nodeName()
        val opName = nd.opName()

        /**
         * How do we handle flexible inputs here?
         * Things may not map 1 to 1.
         * We can optionally add attributes or
         * inputs, sometimes we might try to do both.
         */

        availableToAddSet.remove(name)
        println("Adding operation to graph: $opName (name=$name)")
        var skipCase = false
        val rawAttrMap = HashMap<String,ATTR_VALUE_TYPE>()
        nd.attributeMap().forEach { (name, def) ->
            rawAttrMap[name] = def.internalAttributeValue()
        }


        if (opFilter != null && opFilter.skipOp(
                nd.internalValue(),
                sd,rawAttrMap, irGraph.internalValue())) {
            println("Skipping op $name of type $opName due to op filter")
            //Don't continue at this point - we still need to process what this feeds into...
            skipCase = true
        } else {
            if (importOverride == null || !importOverride.containsKey(name)) {
                //Standard case
                if (irGraph.isConstant(opName)) {
                    //Get array, create a constant
                    val tfTensor = nd.getAttribute("value").tensorValue()
                    val arr = tfTensor.toNd4jNDArray()
                    sd.constant(name, arr)
                    variableTracker.addVariable(name)
                    val inputCount = nd.numInputs()
                    if (inputCount > 0) {
                        //Very likely control dependency. i.e., "we must execute op X before the constant is really available to be used"
                        val l: MutableList<String> = java.util.ArrayList(inputCount)
                        for (i in 0 until inputCount) {
                            val n = nd.inputAt(i)
                            check(isControlDep(n)) { "Found non-control dependency input \"$n\" for constant \"$name\"" }
                            val n2 = stripControl(n)
                            l.add(n2)
                        }
                        constControlDeps[name] = l
                    }
                } else if (irGraph.isPlaceHolder(opName)) {
                    //TODO support the "WithDefault" array
                    val attrMap = nd.attributeMap()
                    var shape = irGraph.shapeOfInput(nd.nodeName())

                    variableTracker.addVariable(name)

                    val dt = irGraph.dataTypeForVariable(nd.nodeName()).nd4jDataType()
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


                    val nd4jOpName = opRegistryHolder.lookupOpMappingProcess(opName).opName()

                    val dfInstance = if( DifferentialFunctionClassHolder.getInstance().hasName(nd4jOpName))DifferentialFunctionClassHolder.getInstance().getInstance(nd4jOpName)
                    else DynamicCustomOp.builder(nd4jOpName).build()
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
                    var controlDeps: MutableList<String?>? = null
                    val opDescriptor = opRegistryHolder.lookupNd4jOpDef(nd4jOpName)
                    val opInputs = opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR  }
                    val numInputs = nd.numInputs()
                    val inNames: MutableList<String> = java.util.ArrayList(numInputs)
                    for (i in 0 until numInputs) {
                        //use input name if it exists and matches, otherwise if the input names do not map 1 to 1 for import
                        //use samediff to generate a unique name
                        val origInName = nd.inputAt(i)
                        variableTracker.addVariable(origInName)
                        var inName = stripControl(origInName)
                        if (inName.endsWith(":0")) {
                            //Strip ":0" suffix. Some ops can depend on placeholders, like "image_tensor:0" but in SameDiff this is a variable called "image_tensor"
                            inName = inName.substring(0, inName.length - 2)
                        }
                        val isControlDep = isControlDep(origInName)
                        if (isControlDep) {
                            if (controlDeps == null) controlDeps = java.util.ArrayList()
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
                            if (v.inputsForOp == null) v.inputsForOp = java.util.ArrayList()
                            v.inputsForOp.add(name)
                        } else if (isControlDep) {
                            if (v!!.controlDepsForOp == null) v.controlDepsForOp = java.util.ArrayList()
                            if (!v.controlDepsForOp.contains(name)) {
                                v.controlDepsForOp.add(name)
                            }
                        }
                    }

                    val opDefLookup = opRegistryHolder.lookupInputFrameworkOpDef(opName) as OP_DEF_TYPE
                    val mappingContext = irGraph.createMappingContext(opDef = opDefLookup,node = irGraph.nodeByName(name))

                    //Create SameDiffOp instance and add to graph
                    val op = SameDiffOp.builder()
                        .name(name)
                        .op(df)
                        .inputsToOp(inNames) //.outputsOfOp(outNames)    //We'll set this later
                        .controlDeps(controlDeps)
                        .build()
                    sd.ops[name] = op
                    initAttributes(df, irGraph.frameworkName(), mappingContext, sd,opName)


                    //DType calculate for output variables (set/correct if necessary)
                    val newInNames = sd.ops[name]!!.inputsToOp //Just in case import has modified this, like for concat case
                    val newInDtypes: MutableList<DataType> =
                        java.util.ArrayList(newInNames.size)
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


                    val numOutputs = nd4jOpDescriptors.findOp(nd4jOpName).argDescriptorList.filter { input -> input.argType == OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR }.size
                    //println("Out dtypes size ${outDTypes.size} and numOutputs $numOutputs")
                    val outSDVars = arrayOfNulls<SDVariable>(numOutputs)
                    val outVars = arrayOfNulls<Variable>(numOutputs)
                    val outNames: MutableList<String> = java.util.ArrayList(numOutputs)
                    val outputDataTypes = df.calculateOutputDataTypes(newInDtypes)
                    //Create output variables and add to graph
                    for (i in 0 until numOutputs) {
                        val dt = outputDataTypes[i]
                        val varName = name + if (i == 0) "" else ":$i"
                        //TODO: handle variadic type in kotlin
                        /**
                         * TODO: handle data type import
                         */
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
                val inputs: MutableList<SDVariable> = java.util.ArrayList()
                var controlDeps: MutableList<SDVariable?>? = null
                val nd4jOpName = opRegistryHolder.lookupOpMappingProcess(opName).opName()
                val opDescriptor = opRegistryHolder.lookupNd4jOpDef(nd4jOpName)
                val opInputs = opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR  }
                val numInputs = opInputs.size

                for (i in 0 until numInputs) {
                    val inName = nodeInputTo[nd.nodeName()]!![i]!!
                    val controlDep = isControlDep(inName)
                    val v = sd.getVariable(name)
                    if (controlDep) {
                        if (controlDeps == null) controlDeps = java.util.ArrayList()
                        controlDeps.add(v)
                    } else {
                        inputs.add(v)
                    }

                    //o!!.initFromTensorFlow(inputs, controlDeps, nd, sd, nd.attributeMap(), irGraph)
                }
            }
        }


        //Now that we have just added an op (or variable) - check what this feeds into, and see what we can now process
        // as a result
        if (nodeInputTo.containsKey(name)) {
            val set: List<String>? = nodeInputTo[name]
            for (nextOp in set!!) {
                val nextOpDef = remainingNodes[nextOp]
                val nextOpDefDescriptor = opRegistryHolder.lookupNd4jOpDef(irGraph.nd4jNameForInternalOpName(nextOpDef!!.opName()))
                val opInputs = nextOpDefDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR  }
                val nInNext = nextOpDef.numInputs()

                if (nextOpDef == null) {
                    if (sd.ops.containsKey(nextOp)) {
                        //Already processed this.
                        //Almost certainly the close of a loop - like NextIteration -> Merge case
                        continue
                    }
                    throw IllegalStateException("Could not find op definition for op to import: $nextOp")
                }
                var allAlreadyInGraph = true
                var nonControlSeenCount = 0
                /**
                 * TODO: the new input name breaks early and thus the node gets removed,
                 * need to figure out how to make the nodesToRemove converge to empty.
                 * Right now, we're adding variables that aren't specified for processing.
                 * We need to make the inbound input names line up in order for the graph to converge.
                 */
                for (i in 0 until nInNext) {
                    val s = nextOpDef.inputAt(i)
                    var inName = stripControl(nextOpDef.inputAt(i))
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
                val mergeCase = nonControlSeenCount > 0 && "Merge" == nextOpDef.opName()
                if (allAlreadyInGraph || mergeCase) {
                    //Can process this op, add it to the queue for processing
                    if (!availableToAddSet.contains(nextOp)) {
                        //Avoid processing same op multiple times, for repeated inputs to one op, etc
                        availableToAdd.add(nextOpDef)
                        availableToAddSet.add(nextOp)
                        println("Added to processing queue: ${nextOpDef.opName()} (name=$nextOp)")
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
            if (sdo!!.controlDepFor == null) sdo.controlDepFor = java.util.ArrayList()
            val l = sdo.controlDepFor
            if (!l.contains(s)) l.add(varName)
        }
    }

    //Post process the merge ops - all we are missing is a Variable.getInputsForOp().add(mergeOpName);
    for ((key, value) in mergeOpsPostProcess) {
        val v = sd.variables[value]
        if (v!!.inputsForOp == null) v.inputsForOp = java.util.ArrayList()
        v.inputsForOp.add(key)
    }
    Preconditions.checkState(remainingNodes.isEmpty(), "%s Unprocessed nodes: %s", remainingNodes.size, remainingNodes.keys)
    return sd
}
package org.nd4j.codegen.ir

import io.github.classgraph.ClassGraph
import org.apache.commons.io.IOUtils
import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.VariableType
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.io.ReflectionUtils
import org.nd4j.gen.OpDeclarationDescriptor
import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.shade.protobuf.ByteString
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.nd4j.shade.protobuf.TextFormat
import java.lang.IllegalArgumentException
import java.nio.charset.Charset



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


    fun inputOpDef(graphDef: IRGraph<NODE_DEF_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): OP_DEF_TYPE

    fun opName(): String

    fun frameworkVersion(): String

    fun inputFramework(): String

    fun inputFrameworkOpName(): String

    fun attributeMappingRules(): List<AttributeMappingRule<OP_DEF_TYPE,NODE_DEF_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>

    fun tensorMappingRules():  List<TensorMappingRule<OP_DEF_TYPE,NODE_DEF_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>

    fun applyProcess(mappingCtx: MappingContext<NODE_DEF_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): Pair<MappingContext<NODE_DEF_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>, OpNamespace.OpDescriptor>

    fun applyProcessReverse(input: OpDeclarationDescriptor): IRNode<NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>

    fun serialize(): MapperNamespace.MapperDeclaration


}


interface TensorMappingRule<OP_DEF_TYPE: GeneratedMessageV3,NODE_DEF_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {


    fun initWithMappingProcess(mappingProcess: MappingProcess<OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>)


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

    fun initWithMappingProcess(mappingProcess: MappingProcess<OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>)

    fun mappingNamesToPerform(): Map<String,String>

    fun mappingTransformerArgs(): Map<String, List<OpNamespace.ArgDescriptor>>

    fun name(): String

    fun serialize(): MapperNamespace.MappingRule

    fun convertAttributes(mappingCtx: MappingContext<NODE_DEF_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): List<OpNamespace.ArgDescriptor>

    fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>
}


interface MappingContext<NODE_TYPE: GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE: GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,DATA_TYPE: ProtocolMessageEnum> {
    fun node(): NODE_TYPE

    fun opDef(): OP_DEF_TYPE

    fun opName(): String

    fun nodeName(): String

    fun attrDef(name: String): ATTRIBUTE_TYPE

    fun tensorInputFor(name: String): IRTensor<TENSOR_TYPE,DATA_TYPE>

    fun createIRTensorFromNDArray(ndaray:INDArray): IRTensor<TENSOR_TYPE,DATA_TYPE>

    fun nd4jDataTypeFor(input: IRTensor<TENSOR_TYPE,DATA_TYPE>): DataType

    fun irAttributeValueForNode(valueName: String): IRAttribute<ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,TENSOR_TYPE,DATA_TYPE>

    fun graph(): IRGraph<NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>

}

abstract class AbstractMappingContext<NODE_TYPE: GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3,
        ATTRIBUTE_TYPE: GeneratedMessageV3,ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(opDef: OP_DEF_TYPE,
                                                                                                                     node: NODE_TYPE,
                                                                                                                     graph:IRGraph<NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>):

        MappingContext<NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE> {

    val opDef = opDef
    val node = node
    val graph = graph

    override fun node(): NODE_TYPE {
        return node
    }

    override fun opDef(): OP_DEF_TYPE {
        return opDef
    }

    override fun graph(): IRGraph<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        return graph
    }



}

interface IRGraph<NODE_TYPE: GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE: GeneratedMessageV3,ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    fun opDefFor(name: String): OP_DEF_TYPE

    fun nodeByName(input: String): NODE_TYPE

    fun nodeList(): List<NODE_TYPE>

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

interface IROpDef<OP_DEF_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, ARG_DEF_TYPE : GeneratedMessageV3, DATA_TYPE, ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3>
        where DATA_TYPE: ProtocolMessageEnum {
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
}



abstract  class AbstractMappingProcess<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE>(inputFramework: String,
                                                              frameworkVersion: String,
                                                              inputFrameworkOpName: String,
                                                              opName: String,
                                                              opMappingRegistry: OpMappingRegistry<NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,DATA_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE>,
                                                              tensorMappingRules: List<out TensorMappingRule<OP_DEF_TYPE,NODE_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>,
                                                              attributeMappingRules: List<out AttributeMappingRule<OP_DEF_TYPE,NODE_TYPE,ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>):
        MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    protected val inputFramework = inputFramework
    protected val frameworkVersion = frameworkVersion
    protected val inputFrameworkOpName = inputFrameworkOpName
    protected val opName = opName
    protected val tensorMappingRules = tensorMappingRules
    protected val attributeMappingRules = attributeMappingRules
    protected var opDef: OP_DEF_TYPE? = null
    protected val opMappingRegistry = opMappingRegistry

    init {
        tensorMappingRules.forEach {
            it.initWithMappingProcess(this)
        }

        attributeMappingRules.forEach {
            it.initWithMappingProcess(this)
        }


        opMappingRegistry.registerMappingProcess(
                inputFrameworkOpName = inputFrameworkOpName,
                processToRegister = this
        )


    }

    override fun inputOpDef(graphDef: IRGraph<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): OP_DEF_TYPE {
        if(opDef == null)
            this.opDef = graphDef.opDefFor(inputFrameworkOpName)
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

    override fun applyProcess(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): Pair<MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>, OpNamespace.OpDescriptor> {
        val descriptorBuilder = OpNamespace.OpDescriptor.newBuilder()
        descriptorBuilder.name = opName()
        tensorMappingRules.forEach {
            it.convertInput().forEach { descriptor ->
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
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> {

    fun process(): MappingProcess<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>

    fun mappingContext(): MappingContext<NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>

}

abstract class AbstractImportContext<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum>
(process: MappingProcess<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>,
 mappingContext: MappingContext<NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>): ImportContext<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>
{
    val process = process
    val mappingContext = mappingContext

    override fun process(): MappingProcess<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        return process
    }

    override fun mappingContext(): MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        return mappingContext
    }
}
interface ImportProcess<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> {

    fun createMappingProcesses(graph: IRGraph<NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>)
            : List<MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>>

    fun createMappingContext(graph:
                             IRGraph<NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,
                                     DATA_TYPE>,node: NODE_TYPE): MappingContext<NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>


    fun createImportContext(mappingProcess: MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>,mappingContext: MappingContext<NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>)
            : ImportContext<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>

    fun runImportProcess(mappingProcesses: List<ImportContext<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>>): SameDiff

}

abstract class AbstractImportProcess<OP_DEF_TYPE: GeneratedMessageV3,NODE_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE: GeneratedMessageV3,ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,DATA_TYPE: ProtocolMessageEnum>
(inputFramework: String):
        ImportProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE> {

    val inputFramework = inputFramework


    override fun createMappingProcesses(graph: IRGraph<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): List<MappingProcess<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>> {
        return graph.nodeList().map {
            val mappingContext = createMappingContext(graph = graph, node = it)
            val opName = mappingContext.opName()
            OpRegistryHolder.lookupOpMappingProcess<
                    NODE_TYPE,
                    OP_DEF_TYPE,
                    TENSOR_TYPE,
                    DATA_TYPE,
                    ATTRIBUTE_TYPE,
                    ATTRIBUTE_VALUE_TYPE>(inputFrameworkOpName = opName, inputFrameworkName = inputFramework)
        }
    }


    override fun runImportProcess(importContexts: List<ImportContext<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>>): SameDiff {
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



fun createVariable(varName: String,varType: VariableType,sameDiff: SameDiff,shape: List<Long>,dataType: DataType): SDVariable {
    return SDVariable(varName,varType, sameDiff, shape.toLongArray(), dataType)
}
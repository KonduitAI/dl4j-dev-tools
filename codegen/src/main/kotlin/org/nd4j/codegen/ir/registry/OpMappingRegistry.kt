package org.nd4j.codegen.ir.registry

import org.apache.commons.collections4.MultiSet
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.apache.commons.collections4.MultiValuedMap
import org.apache.commons.collections4.multimap.HashSetValuedHashMap
import org.nd4j.codegen.ir.MappingProcess
import org.nd4j.codegen.ir.findOp
import org.nd4j.codegen.ir.nd4jOpDescriptors
import org.nd4j.ir.OpNamespace
import java.lang.IllegalArgumentException


class OpMappingRegistry<GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE: GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3>(inputFrameworkName: String) {

    val registeredOps: MultiValuedMap<String,MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_TYPE,
            TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>> = HashSetValuedHashMap<
            String,MappingProcess<GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE>>()

    val opDefList = HashMap<String,OP_DEF_TYPE>()
    val nd4jOpDefs = HashMap<String,OpNamespace.OpDescriptor>()

    fun mappingProcessNames(): MultiSet<String> {
        return registeredOps.keys()!!
    }

    fun nd4jOpNames(): Set<String> {
        return nd4jOpDefs.keys
    }

    fun inputFrameworkOpNames(): Set<String> {
        return opDefList.keys
    }

    fun lookupNd4jOpDef(name:String): OpNamespace.OpDescriptor {
        return nd4jOpDefs[name]!!
    }


    fun registerNd4jOpDef(name:String, opDef: OpNamespace.OpDescriptor) {
        nd4jOpDefs[name] = opDef
    }

    fun lookupInputFrameworkOpDef(name:String): OP_DEF_TYPE {
        return opDefList[name]!!
    }

    fun registerInputFrameworkOpDef(name: String,opDef: OP_DEF_TYPE) {
        opDefList[name] = opDef
    }

    fun registerMappingProcess(inputFrameworkOpName: String, processToRegister: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>) {
        registeredOps.put(inputFrameworkOpName,processToRegister)
    }

    fun hasMappingOpProcess(inputFrameworkOpName: String): Boolean {
        return registeredOps.containsKey(inputFrameworkOpName)
    }


    fun  lookupOpMappingProcess(inputFrameworkOpName: String): MappingProcess<
            GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE> {
        if(!registeredOps.containsKey(inputFrameworkOpName)) {
            throw IllegalArgumentException("No import process defined for $inputFrameworkOpName")
        }
        return registeredOps[inputFrameworkOpName]!!.first()
    }

    fun opTypeForName(nd4jOpName: String): OpNamespace.OpDescriptor.OpDeclarationType {
        val descriptor = nd4jOpDescriptors.findOp(nd4jOpName)
        return descriptor.opDeclarationType
    }

    fun createMappingContext(frameworkName: String) {

    }

}







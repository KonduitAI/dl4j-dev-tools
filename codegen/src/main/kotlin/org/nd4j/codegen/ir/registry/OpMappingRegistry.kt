package org.nd4j.codegen.ir.registry

import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.apache.commons.collections4.MultiValuedMap
import org.apache.commons.collections4.multimap.HashSetValuedHashMap
import org.nd4j.codegen.ir.MappingProcess



class OpMappingRegistry<NODE_TYPE : GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3>(inputFrameworkName: String) {
    val registeredOps: MultiValuedMap<String,MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>> = HashSetValuedHashMap<String,MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>>()

    fun registerMappingProcess(inputFrameworkOpName: String, processToRegister: MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>) {
        registeredOps.put(inputFrameworkOpName,processToRegister)
    }

    fun  lookupOpMappingProcess(inputFrameworkOpName: String): MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE> {
        return registeredOps[inputFrameworkOpName].first()
    }

}


object OpRegistryHolder {

    val registeredOps = HashSetValuedHashMap<String, OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>>()


    fun <NODE_TYPE : GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3>
            registerMappingProcess(inputFrameworkOpName: String, processToRegister: MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>) {
        registeredOps.put(inputFrameworkOpName,processToRegister as OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>)
    }

    fun  <NODE_TYPE : GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3> lookupOpMappingProcess(inputFrameworkName: String,inputFrameworkOpName: String): MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE> {
        val mappingRegistry = registeredOps[inputFrameworkName].first()
        val lookup = mappingRegistry.lookupOpMappingProcess(inputFrameworkOpName = inputFrameworkOpName)
        return lookup  as MappingProcess<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>
    }
}



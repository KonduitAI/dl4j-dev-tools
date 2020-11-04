package org.nd4j.codegen.ir.registry

import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.apache.commons.collections4.MultiValuedMap
import org.apache.commons.collections4.multimap.HashSetValuedHashMap
import org.nd4j.codegen.ir.MappingProcess
import org.nd4j.codegen.ir.findOp
import org.nd4j.codegen.ir.nd4jOpDescriptors
import org.nd4j.ir.OpNamespace


class OpMappingRegistry<NODE_TYPE : GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3>(inputFrameworkName: String) {
    val registeredOps: MultiValuedMap<String,MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>> = HashSetValuedHashMap<String,MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>>()

    fun registerMappingProcess(inputFrameworkOpName: String, processToRegister: MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>) {
        registeredOps.put(inputFrameworkOpName,processToRegister)
    }

    fun  lookupOpMappingProcess(inputFrameworkOpName: String): MappingProcess<OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE> {
        return registeredOps[inputFrameworkOpName].first()
    }

    fun opTypeForName(nd4jOpName: String): OpNamespace.OpDescriptor.OpDeclarationType {
        val descriptor = nd4jOpDescriptors.findOp(nd4jOpName)
        return descriptor.opDeclarationType
    }

}







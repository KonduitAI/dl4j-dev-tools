package org.nd4j.codegen.ir.registry

import org.apache.commons.collections4.multimap.HashSetValuedHashMap
import org.nd4j.codegen.ir.MappingProcess
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.tensorflow.framework.*

object OpRegistryHolder {

    private val registeredOps = HashSetValuedHashMap<String, OpMappingRegistry<out GeneratedMessageV3, out GeneratedMessageV3, out GeneratedMessageV3, out ProtocolMessageEnum, out GeneratedMessageV3, out GeneratedMessageV3>>()


    fun tensorflow(): OpMappingRegistry<NodeDef,OpDef,TensorProto, DataType,OpDef.AttrDef, AttrValue> {
        return registeredOps["tensorflow"].first() as OpMappingRegistry<NodeDef, OpDef, TensorProto, DataType, OpDef.AttrDef, AttrValue>
    }

    fun registerOpMappingRegistry(framework: String, registry: OpMappingRegistry<out GeneratedMessageV3, out GeneratedMessageV3, out GeneratedMessageV3, out ProtocolMessageEnum, out GeneratedMessageV3, out GeneratedMessageV3>) {
        registeredOps.put(framework,registry)
    }

    fun <NODE_TYPE : GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3>
            registerMappingProcess(inputFrameworkOpName: String, processToRegister: MappingProcess<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>) {
        registeredOps.put(inputFrameworkOpName,processToRegister as OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>)
    }

    fun  <NODE_TYPE : GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3> lookupOpMappingProcess(inputFrameworkName: String, inputFrameworkOpName: String): MappingProcess<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        val mappingRegistry = registeredOps[inputFrameworkName].first()
        val lookup = mappingRegistry.lookupOpMappingProcess(inputFrameworkOpName = inputFrameworkOpName)
        return lookup  as MappingProcess<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>
    }
}

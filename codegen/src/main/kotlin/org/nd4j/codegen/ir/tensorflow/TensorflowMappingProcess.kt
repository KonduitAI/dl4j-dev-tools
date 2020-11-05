package org.nd4j.codegen.ir.tensorflow

import org.nd4j.codegen.ir.AbstractMappingProcess
import org.nd4j.codegen.ir.AttributeMappingRule
import org.nd4j.codegen.ir.TensorMappingRule
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.tensorflow.framework.*

open class TensorflowMappingProcess(inputFramework: String = "tensorflow",
                                           frameworkVersion: String = "2.3",
                                           inputFrameworkOpName: String,
                                           opName: String,
                                           opMappingRegistry: OpMappingRegistry<NodeDef,OpDef,TensorProto,DataType, OpDef.AttrDef,AttrValue>,
                                           tensorMappingRules: List<TensorMappingRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>> = emptyList(),
                                           attributeMappingRules: List<AttributeMappingRule<OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>> = emptyList())
    : AbstractMappingProcess<OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>(
        inputFramework,
        frameworkVersion,
        inputFrameworkOpName,
        opName,
        opMappingRegistry,
        tensorMappingRules,
        attributeMappingRules) {

}



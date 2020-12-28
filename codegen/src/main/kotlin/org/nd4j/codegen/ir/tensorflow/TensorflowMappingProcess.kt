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
                                    opMappingRegistry: OpMappingRegistry<GraphDef,
                                            NodeDef,OpDef,
                                            TensorProto,DataType, OpDef.AttrDef,AttrValue>,
                                    tensorMappingRules: List<TensorMappingRule<GraphDef,
                                            OpDef, NodeDef,
                                            OpDef.AttrDef,
                                            AttrValue, TensorProto, DataType>> = emptyList(),
                                    attributeMappingRules: List<AttributeMappingRule<GraphDef,
                                            OpDef, NodeDef,
                                            OpDef.AttrDef,
                                            AttrValue,
                                            TensorProto, DataType>> = emptyList(),
                                    inputIndexOverrides: Map<Int,Int> = emptyMap())
    : AbstractMappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef,
        AttrValue, DataType>(
    inputFramework,
    frameworkVersion,
    inputFrameworkOpName,
    inputIndexOverrides,
    opName,
    opMappingRegistry,
    tensorMappingRules,
    attributeMappingRules) {

}



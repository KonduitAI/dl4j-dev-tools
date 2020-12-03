package org.nd4j.codegen.ir.onnx

import onnx.Onnx
import org.nd4j.codegen.ir.AbstractMappingProcess
import org.nd4j.codegen.ir.AttributeMappingRule
import org.nd4j.codegen.ir.TensorMappingRule
import org.nd4j.codegen.ir.registry.OpMappingRegistry

open class OnnxMappingProcess(inputFramework: String = "onnx",
                              frameworkVersion: String = "1.4",
                              inputFrameworkOpName: String,
                              opName: String,
                              opMappingRegistry: OpMappingRegistry<Onnx.NodeProto,
                                      Onnx.NodeProto,
                                      Onnx.TensorProto,
                                      Onnx.TensorProto.DataType,
                                      Onnx.AttributeProto,
                                      Onnx.AttributeProto>,
                              tensorMappingRules: List<TensorMappingRule<Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto,Onnx.TensorProto,Onnx.TensorProto.DataType>> = emptyList(),
                              attributeMappingRules: List<out AttributeMappingRule<Onnx.NodeProto, Onnx.NodeProto,Onnx.AttributeProto, Onnx.AttributeProto,
                                      Onnx.TensorProto, Onnx.TensorProto.DataType>> = emptyList())
    : AbstractMappingProcess<Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>(
        inputFramework,
        frameworkVersion,
        inputFrameworkOpName,
        opName,
        opMappingRegistry,
        tensorMappingRules,
        attributeMappingRules) {

}


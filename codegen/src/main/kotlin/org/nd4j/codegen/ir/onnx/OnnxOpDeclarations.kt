package org.nd4j.codegen.ir.onnx

import onnx.Onnx
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder

val onnxOpRegistry = OpMappingRegistry<Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto>("onnx")
class AbsMappingProcess: OnnxMappingProcess(
        opName = "abs", tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("x" to "x"))),
        inputFrameworkOpName = "Abs",
        inputFramework = "onnx",
        opMappingRegistry = onnxOpRegistry)

class ConstMappingProcess: OnnxMappingProcess(
        opName = "identity",
        inputFrameworkOpName = "Constant",
        opMappingRegistry = onnxOpRegistry
)



class Conv2DMappingProcess: OnnxMappingProcess(
        inputFramework = "onnx",
        inputFrameworkOpName = "Conv",
        opName = "conv2d",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","filter" to "weights"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW"),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "auto_pad",valueToTest = "SAME"),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "pH",inputFrameworkAttributeName = "padding",targetValue = "NCHW",trueIndex = 2,falseIndex = 1),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "pW",inputFrameworkAttributeName = "padding",targetValue = "NCHW",trueIndex = 3,falseIndex = 2),
                sizeThreshold(outputAttribute = "dH",index = 0,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "dilations"),
                sizeThreshold(outputAttribute = "dW",index = 1,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "dilations"),
                sizeThreshold(outputAttribute = "sH",index = 0,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "strides"),
                sizeThreshold(outputAttribute = "sW",index = 0,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "strides"),
                sizeThreshold(outputAttribute = "kH",index = 0,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "kernel_shape"),
                sizeThreshold(outputAttribute = "kW",index = 1,sizeThreshold =  2,fallbackIndex = 0,inputFrameworkAttributeName = "kernel_shape")
        ),opMappingRegistry = onnxOpRegistry)



object OnnxOpDeclarations {
        init {
                OpRegistryHolder.registerOpMappingRegistry("onnx", onnxOpRegistry)
                AbsMappingProcess()
                Conv2DMappingProcess()
                ConstMappingProcess()
        }
}



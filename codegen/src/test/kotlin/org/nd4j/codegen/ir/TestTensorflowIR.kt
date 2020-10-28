package org.nd4j.codegen.ir

import org.junit.jupiter.api.Test
import org.nd4j.codegen.ir.tensorflow.*
import org.nd4j.ir.OpNamespace
import org.tensorflow.framework.NodeDef
import org.tensorflow.framework.OpDef

class TestTensorflowIR {

    @Test
    fun testTensorflowAbs() {
        val xArgBuilder = OpNamespace.ArgDescriptor.newBuilder().apply {
            name = "x"
            argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
        }

        val yArgBuilder = OpNamespace.ArgDescriptor.newBuilder().apply {
            name = "y"
            argType = OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR

        }

        val absOpDescriptorBuilder = OpNamespace.OpDescriptor.newBuilder().apply {
            name = "abs"
            addArgDescriptor(xArgBuilder.build())
            addArgDescriptor(yArgBuilder.build())
        }

        val absDescriptor = absOpDescriptorBuilder.build()
        val opDef = tensorflowOps.opList.filter { it.name == "Abs" }[0]
        val tensorflowAbsMapper = TensorflowNDArrayMappingRule(
                nd4jOpName = "abs",
                mappingNamesToPerform = mapOf("x" to "x","y" to "y"))

        val nodeDef = NodeDef.newBuilder()
                .addInput("x").addInput("y")
                .setOp("Abs")
                .setName("test")
                .build()




        val tensorflowNode = TensorflowIRNode(nodeDef,opDef)
        val absMappingProcess = AbsMappingProcess()

        val input = absMappingProcess.applyProcess(tensorflowNode)
        System.out.println(input)

    }

    @Test
    fun testStringEqualsMapper() {
        val opDescriptorList = loadNd4jOpDescriptors()
        val targetOpDescriptor = opDescriptorList.opListList.filter { opDescriptor -> opDescriptor.name == "conv2d" }[0]
        val tensorflowOp = loadTensorflowOps().opList.filter { opDef -> opDef.name == "Conv2D" }[0]
        val attrDef = tensorflowOp.attrList.filter { attrDef: OpDef.AttrDef? -> attrDef!!.name == "data_format" }[0]
        val tfIr = TensorflowIR()
        val attrValue = tfIr.createAttributeValue("NCHW",AttributeValueType.STRING)
        val otherInputValue = tfIr.createAttributeValue("NCHW",AttributeValueType.STRING)
        val irAttribute = listOf(TensorflowIRAttr(attrDef,attrValue), TensorflowIRAttr(attrDef,otherInputValue))
        val tensorflowStringEquals = TensorflowStringEqualsAdapterRule(
                targetOpDescriptor,
                hashMapOf("isNCHW" to "data_format"),
                attrDef,
                attrValue,
                hashMapOf("isNCHW" to irAttribute))
        val convertedRule = tensorflowStringEquals.serialize()
        val convertedAttributes = tensorflowStringEquals.convertAttributes()


    }
}
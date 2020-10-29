package org.nd4j.codegen.ir

import org.junit.jupiter.api.Test
import org.nd4j.codegen.ir.tensorflow.*
import org.nd4j.shade.protobuf.ByteString
import org.tensorflow.framework.AttrValue
import org.tensorflow.framework.NodeDef
import org.tensorflow.framework.OpDef
import java.nio.charset.Charset

class TestTensorflowIR {

    @Test
    fun testTensorflowAbs() {
        val opDef = tensorflowOps.opList.filter { it.name == "Abs" }[0]
        val nodeDef = NodeDef.newBuilder()
                .addInput("x").addInput("y")
                .setOp("Abs")
                .setName("test")
                .build()




        val tensorflowNode = TensorflowIRNode(nodeDef,opDef)
        val absMappingProcess = AbsMappingProcess()

        val input = absMappingProcess.applyProcess(tensorflowNode)
        println(input)

    }

    @Test
    fun testTensorflowConv2d() {
        val opDef = tensorflowOps.opList.filter { it.name == "Conv2D" }[0]
        val listValue = AttrValue.ListValue.newBuilder().addI(1).addI(1)
                .addI(1).addI(1)
                .build()
        val dilations = AttrValue.ListValue.newBuilder().addI(1).addI(1)
                .addI(1).addI(1)
                .build()

        val attrValue = AttrValue.newBuilder()
                .setList(listValue)
                .build()

        val dilationsAttr = AttrValue.newBuilder()
                .setList(dilations)
                .build()

        val dataFormatValue = AttrValue.newBuilder()
                .setS(ByteString.copyFrom("NCHW", Charset.defaultCharset()))
                .build()

        val paddingValue = AttrValue.newBuilder()
                .setS(ByteString.copyFrom("SAME", Charset.defaultCharset()))
                .build()

        val nodeDef = NodeDef.newBuilder()
                .addInput("input").addInput("filter")
                .setOp("Conv2D")
                .setName("input")
                .putAttr("strides", attrValue)
                .putAttr("data_format",dataFormatValue)
                .putAttr("padding",paddingValue)
                .putAttr("dilations",dilationsAttr)
                .build()

        val tensorflowIRNode = TensorflowIRNode(nodeDef,opDef)
        val conv2dMappingProcess = Conv2DMappingProcess()
        val processOutput = conv2dMappingProcess.applyProcess(tensorflowIRNode)
        println(processOutput)

    }


    /*   @Test
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


       }*/
}
package org.nd4j.codegen.ir

import org.junit.jupiter.api.Test
import org.tensorflow.framework.OpDef

class TestTensorflowIR {

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
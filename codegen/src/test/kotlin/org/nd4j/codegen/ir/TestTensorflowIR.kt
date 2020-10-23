package org.nd4j.codegen.ir

import org.apache.commons.io.IOUtils
import org.junit.jupiter.api.Test
import org.nd4j.common.io.ClassPathResource
import org.nd4j.ir.OpNamespace
import org.nd4j.shade.protobuf.ByteString
import org.nd4j.shade.protobuf.TextFormat
import org.tensorflow.framework.AttrValue
import org.tensorflow.framework.DataType
import org.tensorflow.framework.OpDef
import org.tensorflow.framework.OpList
import java.nio.charset.Charset

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
        val convertedRule = tensorflowStringEquals.toSaving()
        val convertedAttributes = tensorflowStringEquals.convertAttributes()


    }
}
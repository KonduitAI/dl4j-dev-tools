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
        //class TensorflowStringEqualsAdapterRule(opDescriptor: OpNamespace.OpDescriptor, mappingNamesToPerform: Map<String, String>, inputAttributeDef: AttrDef, inputAttributeValue: AttrValue, transformerArgs: Map<String, List<IRAttribute<AttrDef, AttrValue, TensorProto, org.tensorflow.framework.DataType>>>)
        val nd4jOpDescriptorResourceStream = ClassPathResource("op-ir.proto").inputStream
        val resourceString = IOUtils.toString(nd4jOpDescriptorResourceStream, Charset.defaultCharset())
        val descriptorListBuilder = OpNamespace.OpDescriptorList.newBuilder()
        TextFormat.merge(resourceString,descriptorListBuilder)
        val opDescriptorList = descriptorListBuilder.build()
        val targetOpDescriptor = opDescriptorList.opListList.filter { opDescriptor -> opDescriptor.name == "conv2d" }[0]
        val string = IOUtils.toString(ClassPathResource("ops.proto").inputStream, Charset.defaultCharset())
        val tfListBuilder = OpList.newBuilder()
        TextFormat.merge(string,tfListBuilder)
        val tensorflowOp = tfListBuilder.build().opList.filter { opDef -> opDef.name == "Conv2D" }[0]
        val attrDef = tensorflowOp.attrList.filter { attrDef: OpDef.AttrDef? -> attrDef!!.name == "data_format" }[0]
        val attrValueBuilder = AttrValue.newBuilder()
        attrValueBuilder.type = DataType.DT_STRING
        attrValueBuilder.s = ByteString.copyFrom("NCHW", Charset.defaultCharset())
        val attrValue = attrValueBuilder.build()

        val otherInputValue = AttrValue.newBuilder()
        otherInputValue.type = DataType.DT_STRING
        otherInputValue.s = ByteString.copyFrom("NCHW", Charset.defaultCharset())


        val irAttribute = listOf(TensorflowIRAttr(attrDef,attrValue), TensorflowIRAttr(attrDef,otherInputValue.build()))
        val tensorflowStringEquals = TensorflowStringEqualsAdapterRule(
                targetOpDescriptor,
                hashMapOf("dataFormat" to "data_format"),
                attrDef,
                attrValue,
                hashMapOf("dataFormat" to irAttribute))
        val convertedAttributes = tensorflowStringEquals.convertAttributes()


    }
}
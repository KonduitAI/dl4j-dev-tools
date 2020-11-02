package org.nd4j.codegen.ir

import org.apache.commons.io.IOUtils
import org.junit.jupiter.api.Test
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.codegen.ir.tensorflow.*
import org.nd4j.common.io.ClassPathResource
import org.nd4j.shade.protobuf.ByteString
import org.tensorflow.framework.*
import java.nio.charset.Charset

class TestTensorflowIR {

    @Test
    fun testTensorflowAbs() {
        val opDef = tensorflowOps.findOp("Abs")
        val nodeDef = NodeDef {
            op = "Abs"
            Input("x")
            Input("y")
            name = "test"
        }

        val graphDef = GraphDef {
            Node(nodeDef)
        }

        val tensorflowNode = TensorflowIRNode(nodeDef, opDef)
        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps)
        val absMappingProcess = OpRegistryHolder.lookupOpMappingProcess<NodeDef,OpDef,TensorProto,DataType, OpDef.AttrDef,AttrValue>(inputFrameworkName = "tensorflow",inputFrameworkOpName = "Abs")
        val input = absMappingProcess.applyProcess(tensorflowNode, tfGraph)
        println(input)

    }


    @Test
    fun loadModelTest() {
        val inputs = listOf("input_0", "input_1")
        val content = IOUtils.toByteArray(ClassPathResource("lenet_frozen.pb").inputStream)
        val graphDef = GraphDef.parseFrom(content)
        println(graphDef)
    }

    @Test
    fun testTensorflowConv2d() {
        val opDef = tensorflowOps.findOp("Conv2D")
        val attrValue = AttrValue {
            list = ListAttrValue(1,1,1,1)
        }

        val dilationsAttr = AttrValue {
            list = ListAttrValue(1,1,1,1)
        }

        val dataFormatValue = AttrValue {
            s = ByteString.copyFrom("NCHW", Charset.defaultCharset())
        }

        val paddingValue = AttrValue {
            s = ByteString.copyFrom("SAME", Charset.defaultCharset())
        }

        val nodeDef = NodeDef {
            Input("input")
            Input("filter")
            op = "Conv2D"
            name = "input"
            Attribute("strides",attrValue)
            Attribute("data_format",dataFormatValue)
            Attribute("padding",paddingValue)
            Attribute("dilations",dilationsAttr)
        }

        val graphDef = GraphDef {
            Node(nodeDef)
        }


        val tensorflowIRNode = TensorflowIRNode(nodeDef, opDef)
        val conv2dMappingProcess = OpRegistryHolder.lookupOpMappingProcess<NodeDef,OpDef,TensorProto,DataType, OpDef.AttrDef,AttrValue>(inputFrameworkName = "tensorflow",inputFrameworkOpName = "Conv2D")

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps)
        val processOutput = conv2dMappingProcess.applyProcess(tensorflowIRNode, tfGraph)
        println(processOutput)

    }
}
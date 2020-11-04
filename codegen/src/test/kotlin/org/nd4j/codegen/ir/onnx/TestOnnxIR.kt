package org.nd4j.codegen.ir.onnx

import org.junit.jupiter.api.Test
import org.nd4j.codegen.ir.registry.OpRegistryHolder

class TestOnnxIR {
    val declarations = OnnxOpDeclarations

    @Test
    fun testAbs() {
        val op = onnxops.first { it.name == "Abs" }
        val nodeProto = NodeProto {
            opType = "Abs"
            name = "input"
            Input("X")
            Input("Y")
        }

        val graphProto = GraphProto {
            Node(nodeProto)
        }

        val onnxIRGraph = OnnxIRGraph(graphDef = graphProto)
        val mappingProcess = OpRegistryHolder.onnx().lookupOpMappingProcess("Abs")
        val onnxMappingContext = OnnxMappingContext(opDef = op,node = nodeProto,graph = onnxIRGraph)
        val appliedProcess = mappingProcess.applyProcess(mappingCtx = onnxMappingContext)
        println(appliedProcess)
    }
}
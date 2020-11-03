package org.nd4j.codegen.ir.onnx

import org.junit.jupiter.api.Test

class TestOnnxIR {
    val declarations = OnnxOpDeclarations

    @Test
    fun testConv2d() {
        val op = onnxops.first { it.name == "Abs" }
    }
}
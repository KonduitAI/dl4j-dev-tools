package org.nd4j.codegen.util

import org.nd4j.codegen.impl.java.JavaPoetGenerator
import org.nd4j.codegen.ops.Bitwise
import org.nd4j.codegen.ops.Random
import java.io.File

fun main() {
    val outDir = File("F://test-output/")
    outDir.mkdirs()

    listOf(Bitwise(), Random()).forEach {
        val generator = JavaPoetGenerator()
        generator.generateNamespaceNd4j(it, null, outDir)
    }
}
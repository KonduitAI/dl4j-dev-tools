package org.nd4j.codegen.ops

import org.nd4j.codegen.api.DataType
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*

fun Linalg() =  Namespace("Linalg") {
    //val namespaceJavaPackage = "org.nd4j.linalg"

    Op("Cholesky") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms"
        javaOpClass = "Cholesky"
        Input(DataType.NUMERIC, "input") { description = "Input tensor with inner-most 2 dimensions forming square matrices" }
        Output(DataType.NUMERIC, "output"){ description = "Transformed tensor" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes the Cholesky decomposition of one or more square matrices.
            """.trimIndent()
        }
    }
}
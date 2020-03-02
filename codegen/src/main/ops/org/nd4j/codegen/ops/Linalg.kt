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
        Input(DataType.NDARRAY, "input") { description = "Input tensor with inner-most 2 dimensions forming square matrices" }
        Output(DataType.NDARRAY, "output"){ description = "Transformed tensor" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes the Cholesky decomposition of one or more square matrices.
            """.trimIndent()
        }
    }

    Op("Lstsq") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Lstsq"

        Input(DataType.NDARRAY, "matrix") {description = "input tensor"}
        Input(DataType.NDARRAY, "rhs") {description = "input tensor"}
        Arg(DataType.FLOATING_POINT, "l2_reguralizer") {description = "regularizer"}
        Arg(DataType.BOOL, "fast") {description = "fast mode, defaults to True"}
        Output(DataType.NDARRAY, "output"){ description = "Transformed tensor" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Solver for linear squares problems.
            """.trimIndent()
        }
    }

    Op("Solve") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "LinearSolve"

        Input(DataType.NDARRAY, "matrix") {description = "input tensor"}
        Input(DataType.NDARRAY, "rhs") {description = "input tensor"}
        Arg(DataType.BOOL, "adjoint") {description = "adjoint mode, defaults to False"}
        Output(DataType.NDARRAY, "output"){ description = "Transformed tensor" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Solver for systems of linear questions.
            """.trimIndent()
        }
    }

    Op("Lu") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Lu"

        Input(DataType.NDARRAY, "input") {description = "input tensor"}

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes LU decomposition.
            """.trimIndent()
        }
    }

    /*Op("Matmul") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Matmul"

        Input(DataType.NDARRAY, "a") {description = "input tensor"}
        Input(DataType.NDARRAY, "b") {description = "input tensor"}
        Input(DataType.NDARRAY, "output") {description = "output tensor"}

        Doc(Language.ANY, DocScope.ALL){
            """
             Performs matrix mutiplication on input tensors.
            """.trimIndent()
        }
    }

    Op("Qr") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Qr"

        Input(DataType.NDARRAY, "input") {description = "input tensor"}

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes QR decomposition.
            """.trimIndent()
        }
    }*/
}
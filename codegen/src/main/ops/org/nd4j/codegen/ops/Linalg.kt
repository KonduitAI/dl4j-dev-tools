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

    Op("Lstsq") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Lstsq"

        Input(DataType.NUMERIC, "matrix") {description = "input tensor"}
        Input(DataType.NUMERIC, "rhs") {description = "input tensor"}
        Arg(DataType.FLOATING_POINT, "l2_reguralizer") {description = "regularizer"}
        Arg(DataType.BOOL, "fast") {description = "fast mode, defaults to True"; defaultValue = true}
        Output(DataType.FLOATING_POINT, "output"){ description = "Transformed tensor" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Solver for linear squares problems.
            """.trimIndent()
        }
    }

    Op("Solve") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "LinearSolve"

        Input(DataType.NUMERIC, "matrix") {description = "input tensor"}
        Input(DataType.NUMERIC, "rhs") {description = "input tensor"}
        Arg(DataType.BOOL, "adjoint") {description = "adjoint mode, defaults to False"; defaultValue = false}
        Output(DataType.FLOATING_POINT, "output"){ description = "Output tensor" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Solver for systems of linear equations.
            """.trimIndent()
        }
    }

    Op("TriangularSolve") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "TriangularSolve"

        Input(DataType.NUMERIC, "matrix") {description = "input tensor"}
        Input(DataType.NUMERIC, "rhs") {description = "input tensor"}
        Arg(DataType.BOOL, "lower") {description = "defines whether innermost matrices in matrix are lower or upper triangular"}
        Arg(DataType.BOOL, "adjoint") {description = "adjoint mode"}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Solver for systems of linear questions.
            """.trimIndent()
        }
    }

    Op("Lu") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Lu"

        Input(DataType.NUMERIC, "input") {description = "input tensor"}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes LU decomposition.
            """.trimIndent()
        }
    }

    Op("Matmul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce"
        javaOpClass = "Mmul"

        Input(DataType.NUMERIC, "a") {description = "input tensor"}
        Input(DataType.NUMERIC, "b") {description = "input tensor"}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Performs matrix mutiplication on input tensors.
            """.trimIndent()
        }
    }

    Op("Qr") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "Qr"

        Input(DataType.NUMERIC, "input") {description = "input tensor"}
        Arg(DataType.BOOL, "full") {description = "full matrices mode"; defaultValue = false}
        Output(DataType.FLOATING_POINT, "outputQ")
        Output(DataType.FLOATING_POINT, "outputR")

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes the QR decompositions of input matrix.
            """.trimIndent()
        }
    }

    Op("MatrixBandPart") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "MatrixBandPart"

        Input(DataType.NUMERIC, "input") { description = "input tensor" }
        Arg(DataType.INT, "minLower") { description = "lower diagonal count" }
        Arg(DataType.INT, "maxUpper") { description = "upper diagonal count" }
        Output(DataType.FLOATING_POINT, "output1")
        Output(DataType.FLOATING_POINT, "output2")

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes QR decomposition of input matrix.
            """.trimIndent()
        }
    }

    Op("cross") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "Cross"

        Input(DataType.NUMERIC, "a") {"Input tensor a"}
        Input(DataType.NUMERIC, "b") {"Input tensor b"}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Computes pairwise cross product.
            """.trimIndent()
        }
    }

    Op("diag") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "Diag"

        Input(DataType.NUMERIC, "input") {"Input tensor"}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Calculates diagonal tensor.
            """.trimIndent()
        }
    }

    Op("diag_part") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "DiagPart"

        Input(DataType.NUMERIC, "input") {"Input tensor"}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Calculates diagonal tensor.
            """.trimIndent()
        }
    }

    Op("logdet") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "Logdet"

        Input(DataType.NUMERIC, "input") {"Input tensor"}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Calculates log of determinant.
            """.trimIndent()
        }
    }

    Op("svd") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "Svd"

        Input(DataType.NUMERIC, "input") {"Input tensor"}
        Arg(DataType.BOOL, "fullUV") {"Full matrices mode"}
        Arg(DataType.BOOL, "computeUV") {"Compute U and V"}
        Arg(DataType.INT, "switchNum") {"Switch number"; defaultValue = 16}
        Output(DataType.FLOATING_POINT, "output")

        Doc(Language.ANY, DocScope.ALL){
            """
             Calculates singular value decomposition.
            """.trimIndent()
        }
    }
}
package org.nd4j.codegen.ops

import org.nd4j.codegen.api.DataType.INT
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*


fun Bitwise() = Namespace("Bitwise"){
    val namespaceJavaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"

    Op("leftShift") {
        javaPackage = namespaceJavaPackage
        Input(INT, "x") { description = "Input to be bit shifted" }
        Input(INT, "y") { description = "Amount to shift elements of x array" }

        Output(INT, "output"){ description = "Bitwise shifted input x" }

        Doc(Language.ANY, DocScope.ALL){
            """
            Bitwise left shift operation. Supports broadcasting.
            """.trimIndent()
        }
    }

    Op("rightShift") {
        javaPackage = namespaceJavaPackage
        Input(INT, "x") { description = "Input to be bit shifted" }
        Input(INT, "y") { description = "Amount to shift elements of x array" }

        Output(INT, "output"){ description = "Bitwise shifted input x" }

        Doc(Language.ANY, DocScope.ALL){
            """
            Bitwise right shift operation. Supports broadcasting. 
            """.trimIndent()
        }
    }

    Op("leftShiftCyclic") {
        javaPackage = namespaceJavaPackage
        Input(INT, "x") { description = "Input to be bit shifted" }
        Input(INT, "y") { description = "Amount to shift elements of x array" }

        Output(INT, "output"){ description = "Bitwise cyclic shifted input x" }

        Doc(Language.ANY, DocScope.ALL){
            """
            Bitwise left cyclical shift operation. Supports broadcasting.
            Unlike {@link #leftShift(%INPUT_TYPE%, %INPUT_TYPE%)} the bits will "wrap around":
            {@code leftShiftCyclic(01110000, 2) -> 11000001}
            """.trimIndent()
        }
    }

    Op("rightShiftCyclic") {
        javaPackage = namespaceJavaPackage
        Input(INT, "x") { description = "Input to be bit shifted" }
        Input(INT, "y") { description = "Amount to shift elements of x array" }

        Output(INT, "output"){ description = "Bitwise cyclic shifted input x" }

        Doc(Language.ANY, DocScope.ALL){
            """
            Bitwise right cyclical shift operation. Supports broadcasting.
            Unlike {@link #rightShift(%INPUT_TYPE%, %INPUT_TYPE%)} the bits will "wrap around":
            {@code rightShiftCyclic(00001110, 2) -> 10000011}
            """.trimIndent()
        }
    }

    Op("bitsHammingDistance") {
        javaPackage = namespaceJavaPackage
        val x = Input(INT, "x") { description = "First input array." }
        val y = Input(INT, "y") { description = "Second input array." }
        Constraint("Must be same types"){ sameType(x, y) }

        Output(INT, "output"){ description = "bitwise Hamming distance" }

        Doc(Language.ANY, DocScope.ALL){
            """
            Bitwise Hamming distance reduction over all elements of both input arrays.<br>
            For example, if x=01100000 and y=1010000 then the bitwise Hamming distance is 2 (due to differences at positions 0 and 1)
            """.trimIndent()
        }
    }

    Op("and") {
        javaPackage = namespaceJavaPackage
        val x = Input(INT, "x") { description = "First input array" }
        val y = Input(INT, "y") { description = "Second input array" }
        Constraint("Must be same types"){ sameType(x, y) }
        BackendConstraint("Must have broadcastable shapes"){ broadcastableShapes(x, y) }

        Output(INT, "output"){ description = "Bitwise AND array" }

        Doc(Language.ANY, DocScope.ALL){
            """
            Bitwise AND operation. Supports broadcasting.
            """.trimIndent()
        }
    }

    Op("or") {
        javaPackage = namespaceJavaPackage
        val x = Input(INT, "x") { description = "First input array" }
        val y = Input(INT, "y") { description = "First input array" }
        Constraint("Must be same types"){ sameType(x, y) }
        BackendConstraint("Must have broadcastable shapes"){ broadcastableShapes(x, y) }

        Output(INT, "output"){ description = "Bitwise OR array" }

        Doc(Language.ANY, DocScope.ALL){
            """
            Bitwise OR operation. Supports broadcasting.
            """.trimIndent()
        }
    }

    Op("xor") {
        javaPackage = namespaceJavaPackage
        val x = Input(INT, "x") { description = "First input array" }
        val y = Input(INT, "y") { description = "First input array" }
        Constraint("Must be same types"){ sameType(x, y) }
        BackendConstraint("Must have broadcastable shapes"){ broadcastableShapes(x, y) }

        Output(INT, "output"){ description = "Bitwise XOR array" }

        Doc(Language.ANY, DocScope.ALL){
            """
            Bitwise XOR operation (exclusive OR). Supports broadcasting.
            """.trimIndent()
        }
    }
}
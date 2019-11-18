package org.nd4j.codegen.dsl

import junit.framework.TestCase
import org.apache.commons.io.FileUtils
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.DataType.INT
import org.nd4j.codegen.api.DataType.NUMERIC
import org.nd4j.codegen.api.Exactly
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.impl.java.JavaPoetGenerator
import java.io.File
import java.nio.charset.StandardCharsets

class OpBuilderTest {
    @get:Rule
    public var testDir = TemporaryFolder()

    @Test
    fun opBuilderTest() {
        val outDir = testDir.newFolder()

        val mathNs = Namespace("math") {
            Op("add") {
                javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic"

                Input(NUMERIC, "x") { optional = true; description = "First input to add" }
                Input(NUMERIC,"y") { count = AtLeast(1); description = "Second input to add" }
                Arg(INT,"shape") { count = AtLeast(1); description = "shape" }


                Output(NUMERIC, "z") { description = "Output (x+y)" }


                Doc(Language.ANY, DocScope.ALL) {
                    """
                    (From AddOp) Add op doc text that will appear everywhere - classes, constructors, op creators
                    """.trimIndent()
                }
                Doc(Language.ANY, DocScope.CLASS_DOC_ONLY) {
                    "Add op doc text that will appear in all class docs (javadoc etc)"
                }
                Doc(Language.ANY, DocScope.CONSTRUCTORS_ONLY) {
                    "Add op doc text for constructors only"
                }

            }

            val BaseArithmeticOp = Op("BaseArithmeticOp") {
                isAbstract = true
                javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic"

                Input(NUMERIC,"x") { count = Exactly(1); description = "First operand to %OPNAME%" }
                Input(NUMERIC,"y") { count = Exactly(1); description = "Second operand" }


                Output(NUMERIC,"z") { description = "Output" }

                Doc(Language.ANY, DocScope.ALL) {
                    "(From BaseArithmeticOp) op doc text that will appear everywhere - classes, constructors, op creators"
                }
                Doc(Language.ANY, DocScope.CLASS_DOC_ONLY) {
                    "op doc text that will appear in all class docs (javadoc etc)"
                }
                Doc(Language.ANY, DocScope.CONSTRUCTORS_ONLY) {
                    "op doc text for constructors only"
                }

            }

            Op("sub", extends = BaseArithmeticOp)

            Op("mul", extends = BaseArithmeticOp)

            Op("rsub", extends = BaseArithmeticOp) {
                isAbstract = false
                javaPackage = "org.nd4j.some.other.package"
                Doc(Language.ANY, DocScope.CREATORS_ONLY) {
                    "(From rsub) This doc section will appear only in creator method for %OPNAME%"
                }
            }

            Op("div"){
                javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic"

                val x = Input(NUMERIC,"x") { description = "First operand to div" }
                val y = Input(NUMERIC,"y") { description = "Second operand to div" }
                val idx = Arg(INT, "idx") { description = "Some kind of Index" }
                Constraint("Compatible Rank"){
                    x.rank() eq idx
                }

                Constraint("Compatible Shapes"){
                    sameShape(x,y)
                }


                // namespaces: sdbitwise, sdrandom


                Output(NUMERIC,"z") { description = "Output" }

                Doc(Language.ANY, DocScope.ALL) {
                    "op doc text that will appear everywhere - classes, constructors, op creators"
                }
            }

            Op("foo"){
                javaPackage = "bar"
                val x = Input(NUMERIC,"x") { description = "First operand to div" }
                val y = Input(NUMERIC,"y") { description = "Second operand to div" }

                Constraint("foo bar"){
                    x.sizeAt(7) eq 7 and y.isScalar()
                }
            }
        }

        val generator = JavaPoetGenerator()
        generator.generateNamespaceNd4j(mathNs, null, outDir)
        val exp = File(outDir, "org/nd4j/linalg/api/ops/Nd4jMath.java")
        TestCase.assertTrue(exp.isFile)

        println(FileUtils.readFileToString(exp, StandardCharsets.UTF_8))

    }
}
package org.nd4j.codegen.dsl

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.nd4j.codegen.api.*
import org.nd4j.codegen.api.doc.DocScope
import kotlin.test.assertEquals


class OpInvariantTest {

    @Test
    fun opMustBeDocumented() {
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {}
            }
        }
        assertEquals("foo: Ops must be documented!", thrown.message)
    }


    @Test
    fun opMustBeDocumentedAndNotEmpty() {
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "" }
                }
            }
        }
        assertEquals("foo: Ops must be documented!", thrown.message)
    }

    @Test
    fun opMustBeDocumentedWithDoc() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
            }
        }
    }

    @Test
    fun opSignatureMustCoverAllParameters() {
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Input(DataType.NUMERIC, "y")

                    Signature(x)
                }
            }
        }
        assertEquals("foo: Signature(x) does not cover all parameters! Missing: y", thrown.message)
    }

    @Test
    fun opSignatureMustCoverAllParameters2() {
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.NUMERIC, "y")

                    Signature(x)
                }
            }
        }

        assertEquals("foo: Signature(x) does not cover all parameters! Missing: y", thrown.message)
    }

    @Test
    fun opSignatureMustCoverAllParametersWithoutDefaults() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.NUMERIC, "y") {
                    defaultValue = 7
                }

                Signature(x)
            }
        }
    }

    @Test
    fun opSignatureMustTakeEachParameterOnlyOnce() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.NUMERIC, "y")

                    Signature(x, x, x)
                }
            }
        }

        assertEquals("A parameter may not be used twice in a signature!", thrown.message)
    }

    @Test
    fun opSignatureMustAllowOutputs() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.NUMERIC, "y") {
                    defaultValue = 7
                }
                val out = Output(DataType.NUMERIC, "out")

                Signature(out, x)
            }
        }
    }

    @Test
    fun opSignatureMustAllowOutputsOnlyOnce() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.NUMERIC, "y") {
                        defaultValue = 7
                    }
                    val out = Output(DataType.NUMERIC, "out")

                    Signature(out, x, out)
                }
            }
        }

        assertEquals("A parameter may not be used twice in a signature!", thrown.message)
    }

    @Test
    fun opSignatureDefaultValueMustHaveCorrectShape() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.INT, "y") {
                        defaultValue = x.shape()
                    }

                    Signature(x)
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y). Got x.shape() (org.nd4j.codegen.api.TensorShapeValue)", thrown.message)
    }

    @Test
    fun opSignatureDefaultValue() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") {
                    defaultValue = 2
                }

                Signature(x)
            }
        }
    }

    @Test
    fun opSignatureDefaultValueMustHaveCorrectDataType() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.INT, "y") {
                        defaultValue = 1.7
                    }

                    Signature(x)
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y). Got 1.7 (java.lang.Double)", thrown.message)
    }


    @Test
    fun opSignatureDefaultInputReference() {
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val z = Input(DataType.NUMERIC, "z")
                    val y = Arg(DataType.INT, "y") {
                        count = AtLeast(1)
                        defaultValue = z.shape()
                    }

                    Signature(x)
                }
            }
        }

        assertEquals("foo: Signature(x) does not cover all parameters! Missing: z, y", thrown.message)
    }

    @Test
    fun opSignatureDefaultOutputReference() {
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val y = Arg(DataType.INT, "y") {
                        count = AtLeast(1)
                        defaultValue = out.shape()
                    }

                    Signature(x)
                }
            }
        }

        assertEquals("foo: Signature(x) does not cover all parameters! Missing: y", thrown.message)
    }

    @Test
    fun opSignatureDefaultWithOutputReference() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") {
                    count = AtLeast(1)
                    defaultValue = out.shape()
                }

                Signature(out, x)
            }
        }
    }

    @Test
    fun opSignatureDefaultReferenceChain() {
        val thrown = assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    val out = Output(DataType.NUMERIC, "out")
                    val x = Input(DataType.NUMERIC, "x")
                    val z = Input(DataType.NUMERIC, "z")
                    val u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                    val v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                    val y = Arg(DataType.INT, "y") {
                        count = AtLeast(1)
                        defaultValue = v.shape()
                    }

                    Signature(x)
                }
            }
        }

        assertEquals("foo: Signature(x) does not cover all parameters! Missing: z, u, v, y", thrown.message)
    }

    @Test
    fun opSignatureDefaultReferenceChainWorking() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val z = Input(DataType.NUMERIC, "z") { defaultValue = x }
                val u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                val v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                val y = Arg(DataType.INT, "y") {
                    count = AtLeast(1)
                    defaultValue = v.shape()
                }

                Signature(x)
            }
        }
    }

    @Test
    fun opSignatureShorthandAllParams() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val z = Input(DataType.NUMERIC, "z") { defaultValue = x }
                val u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                val v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                val y = Arg(DataType.INT, "y") {
                    count = AtLeast(1)
                    defaultValue = v.shape()
                }

                AllParamSignature()
            }
        }

    }

    @Test
    fun opSignatureNullDefaults() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") {
                    count = AtLeast(1)
                    defaultValue = null
                }

                AllDefaultsSignature()
            }
        }
    }

    @Test
    fun opSignatureShorthandDefaultParams() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val z = Input(DataType.NUMERIC, "z") { defaultValue = x }
                val u = Input(DataType.NUMERIC, "u") { defaultValue = z }
                val v = Input(DataType.NUMERIC, "v") { defaultValue = u }
                val y = Arg(DataType.INT, "y") {
                    count = AtLeast(1)
                    defaultValue = v.shape()
                }

                AllDefaultsSignature()
            }
        }
    }

    @Test
    fun opSignatureSupportsArrayDefaults() {
        Namespace("math") {
            Op("foo") {
                Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                val out = Output(DataType.NUMERIC, "out")
                val x = Input(DataType.NUMERIC, "x")
                val y = Arg(DataType.INT, "y") { count = AtLeast(0); defaultValue = intArrayOf() }
                val z = Arg(DataType.FLOATING_POINT, "z") { count = Range(2, 5); defaultValue = doubleArrayOf(1.0, 2.0, 3.0) }
                val a = Arg(DataType.BOOL, "a") { count = AtLeast(1); defaultValue = booleanArrayOf(true) }

                AllDefaultsSignature()
            }
        }
    }


    @Test
    fun opSignatureSupportsArrayDefaultsAtLeast() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    Output(DataType.NUMERIC, "out")
                    Input(DataType.NUMERIC, "x")
                    Arg(DataType.INT, "y") { count = AtLeast(1); defaultValue = intArrayOf() }
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y){ count = AtLeast(min=1) }. Got [] ([I)", thrown.message)

    }

    @Test
    fun opSignatureSupportsArrayDefaultsAtMost() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    Output(DataType.NUMERIC, "out")
                    Input(DataType.NUMERIC, "x")
                    Arg(DataType.INT, "y") { count = AtMost(1); defaultValue = intArrayOf(1, 2) }
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y){ count = AtMost(max=1) }. Got [1, 2] ([I)", thrown.message)

    }

    @Test
    fun opSignatureSupportsArrayDefaultsRange() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    Output(DataType.NUMERIC, "out")
                    Input(DataType.NUMERIC, "x")
                    Arg(DataType.INT, "y") { count = Range(3, 7); defaultValue = intArrayOf() }
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y){ count = Range(from=3, to=7) }. Got [] ([I)", thrown.message)
    }

    @Test
    fun opSignatureSupportsArrayDefaultsExactly() {
        val thrown = assertThrows<java.lang.IllegalArgumentException> {
            Namespace("math") {
                Op("foo") {
                    Doc(Language.ANY, DocScope.ALL) { "Some Documentation" }
                    Output(DataType.NUMERIC, "out")
                    Input(DataType.NUMERIC, "x")
                    Arg(DataType.INT, "y") { count = Exactly(7); defaultValue = intArrayOf() }
                }
            }
        }

        assertEquals("Illegal default value for Arg(INT, y){ count = Exactly(count=7) }. Got [] ([I)", thrown.message)

    }
}

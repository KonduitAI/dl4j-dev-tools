package org.nd4j.codegen.dsl

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows


class OpInvariantTest {

    @Test
    fun opMustBeDocumented() {
        assertThrows<java.lang.IllegalStateException> {
            Namespace("math") {
                Op("foo"){}
            }
        }
    }
}
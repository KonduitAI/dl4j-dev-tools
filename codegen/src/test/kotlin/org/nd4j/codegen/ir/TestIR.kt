package org.nd4j.codegen.ir

import org.junit.jupiter.api.Test
import kotlin.test.assertTrue


class TestIR {
    @Test
    fun testOpList() {
        var foundAbs = false
        nd4jOpDescriptors.opListList.forEach {
            println(it.name)
            if(it.name == "abs")
                foundAbs = true
        }

        assertTrue { foundAbs }
    }
}


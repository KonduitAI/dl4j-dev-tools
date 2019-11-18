package org.nd4j.codegen.api.doc

import org.nd4j.codegen.api.Op

object DocTokens {
    private val OPNAME = "%OPNAME%".toRegex()
    private val LIBND4J_OPNAME = "%LIBND4J_OPNAME%".toRegex()

    @JvmStatic fun processDocText(doc: String?, op: Op): String? {
        return doc?.replace(OPNAME, op.opName!!)?.replace(LIBND4J_OPNAME, op.libnd4jOpName!!)
    }
}
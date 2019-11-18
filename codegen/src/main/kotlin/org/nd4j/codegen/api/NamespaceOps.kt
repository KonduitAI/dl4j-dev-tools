package org.nd4j.codegen.api

data class NamespaceOps @JvmOverloads constructor(
    var name: String? = null,
    var include: MutableList<String>? = null,
    var ops: MutableList<Op>? = null
)
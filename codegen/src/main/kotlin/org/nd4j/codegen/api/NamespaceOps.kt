package org.nd4j.codegen.api

data class NamespaceOps @JvmOverloads constructor(
    var name: String,
    var include: MutableList<String>? = null,
    var ops: MutableList<Op> = mutableListOf(),
    var configs: MutableList<Config> = mutableListOf()
) {
    fun addConfig(config: Config) {
        configs.add(config)
    }

    /**
     * Check that all required properties are set
     */
    fun checkInvariants() {
        val usedConfigs = mutableSetOf<Config>()
        ops.forEach { op ->
            usedConfigs.addAll(op.configs)
        }
        val unusedConfigs = configs.toSet() - usedConfigs
        if(unusedConfigs.size > 0){
            throw IllegalStateException("Found unused configs: ${unusedConfigs.joinToString(", ") { it.name }}")
        }
    }
}
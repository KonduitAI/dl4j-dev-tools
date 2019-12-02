package org.nd4j.codegen.api

import org.nd4j.codegen.api.doc.DocSection


data class Op (
        var opName: String? = null,
        var libnd4jOpName: String? = null,
        var javaOpClass: String? = null,
        var isAbstract: Boolean = false,
        var legacy: Boolean = false,
        var javaPackage: String? = null,
        val inputs: MutableList<Input> = mutableListOf(),
        val outputs: MutableList<Output> = mutableListOf(),
        val args: MutableList<Arg> = mutableListOf(),
        val constraints: MutableList<Constraint> = mutableListOf(),
        val signatures: MutableList<Signature> = mutableListOf(),
        val doc: MutableList<DocSection> = mutableListOf(),
        val configs: MutableList<Config> = mutableListOf()
) {

    override fun toString(): String {
        return "Op(opName=$opName, libnd4jOpName=$libnd4jOpName, isAbstract=$isAbstract)"
    }

    fun addInput(input: Input) { inputs.add(input) }
    fun addArgument(arg: Arg) { args.add(arg) }
    fun addOutput(output: Output) { outputs.add(output) }
    fun addDoc(docs: DocSection){ doc.add(docs) }
    fun addSignature(signature: Signature){ signatures.add(signature) }
    fun addConstraint(constraint: Constraint){ constraints.add(constraint) }
    fun addConfig(config: Config) { configs.add(config) }

    /**
     * Check that all required properties are set
     */
    fun checkInvariants() {
        if( !isAbstract && (doc?.size == 0 || doc?.all { it.text.isNullOrBlank() } != false )){
            throw IllegalStateException("$opName: Ops must be documented!")
        }

        signatures?.forEach {
            val opParameters = mutableListOf<Parameter>()
            opParameters.addAll(inputs!!)
            opParameters.addAll(args!!)

            val notCovered = opParameters.fold(mutableListOf<Parameter>()){acc, parameter ->
                if(!(it.parameters.contains(parameter) || parameter.defaultValueIsApplicable(it.parameters))){
                    acc.add(parameter)
                }
                acc
            }

            if(notCovered.size > 0){
                throw IllegalStateException("$opName: $it does not cover all parameters! Missing: ${notCovered.joinToString(", ") { it.name() }}")
            }
        }
    }

    fun Config.input(name: String) = inputs.find { it.name == name }
    fun Config.arg(name: String) = args.find { it.name == name }

}
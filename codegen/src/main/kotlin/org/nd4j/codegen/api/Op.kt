package org.nd4j.codegen.api

import org.nd4j.codegen.api.doc.DocSection


class Op @JvmOverloads constructor(
        var opName: String? = null,
        var libnd4jOpName: String? = null,
        var javaOpClass: String? = null,
        var isAbstract: Boolean = false,
        var extendsOp: Op? = null,
        legacy: Boolean? = null,
        javaPackage: String? = null,
        inputs: MutableList<Input>? = null,
        outputs: MutableList<Output>? = null,
        args: MutableList<Arg>? = null,
        constraints: MutableList<Constraint>? = null,
        signatures: MutableList<Signature>? = null,
        doc: MutableList<DocSection>? = null) {
    var javaPackage: String? = javaPackage
        get() = field ?: extendedOp { javaPackage }

    var legacy: Boolean? = legacy
        get() = field ?: extendedOp { legacy }

    fun addInput(input: Input) = addToList({inputs}, {this.inputs = it}, input)
    var inputs: MutableList<Input>? = inputs
        get() = field ?: extendedOp { inputs }


    fun addOutput(output: Output) = addToList({outputs}, {this.outputs = it}, output)
    var outputs: MutableList<Output>? = outputs
        get() = field ?: extendedOp { outputs }

    fun addArgument(arg: Arg) = addToList({args}, {this.args = it}, arg)
    var args: MutableList<Arg>? = args
        get() = field ?: extendedOp { args }

    fun addConstraint(constraint: Constraint) = addToList({constraints}, {this.constraints = it}, constraint)
    var constraints: MutableList<Constraint>? = constraints
        get() = field ?: extendedOp { constraints }


    fun addDoc(value: DocSection) = addToList({doc!!}, {this.doc = it}, value)
    var doc: MutableList<DocSection>? = doc
        get() = field ?: extendedOp { doc }

    fun addSignature(value: Signature) = addToList({signatures!!}, {this.signatures = it}, value)
    var signatures: MutableList<Signature>? = signatures
        get() = field ?: extendedOp { signatures }

    private fun extendedOp(): Op? {
        return extendsOp
    }

    private fun <T> extendedOp(block: Op.() -> T): T? {
        return extendedOp()?.block()
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Op

        if (opName != other.opName) return false
        if (extendsOp != other.extendsOp) return false

        return true
    }

    override fun hashCode(): Int {
        var result = opName?.hashCode() ?: 0
        result = 31 * result + (extendsOp?.hashCode() ?: 0)
        return result
    }

    override fun toString(): String {
        return "Op(opName=$opName, libnd4jOpName=$libnd4jOpName, isAbstract=$isAbstract, extendsOp=$extendsOp)"
    }

    private fun <T> addToList(field: Op.() -> MutableList<T>?, setter: (MutableList<T>) -> Unit, value: T){
        val list = if(extendsOp != null && extendedOp()?.field() === this.field()){
            mutableListOf()
        }else{
            this.field()
        }
        list?.add(value)
        setter(list!!)
    }

    private fun <T> addToMap(field: Op.() -> MutableMap<String, T>?, setter: (MutableMap<String, T>) -> Unit, key: String, value: T){
        val map = if(extendsOp != null && extendedOp()?.field() === this.field()){
            mutableMapOf()
        }else{
            this.field()
        }
        map?.put(key, value)
        setter(map!!)
    }

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
    }
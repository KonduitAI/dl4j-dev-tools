package org.nd4j.codegen.ir

import java.lang.StringBuilder
import java.util.regex.Pattern

class VariableStateTracker(nameScopes: List<String> = mutableListOf(),
                           ops: MutableSet<String> = mutableSetOf(),
                           variables: MutableSet<String> = mutableSetOf()) {

    private val nameScopes: List<String> = nameScopes
    private val ops: MutableSet<String> = ops
    private val variables: MutableSet<String> = variables



    fun hasOp(opName: String): Boolean {
        return ops.contains(opName)
    }

    fun hasVariable(varName: String): Boolean {
        return variables.contains(varName)
    }

    fun addOp(opName: String) {
        ops.add(opName)
    }

    fun addVariable(variableToAdd: String) {
        variables.add(variableToAdd)
    }

    fun numOps(): Int {
        return ops.size
    }

    fun numVariables(): Int {
        return variables.size
    }

    /**
     * @return The current name scope, if any (null otherwise). See [.withNameScope] for more details.
     */
    fun currentNameScope(): String {
        if (nameScopes.isEmpty()) return ""

        //Would use String.join but that is Java 8+
        val sb = StringBuilder()
        var first = true
        for (ns in nameScopes) {
            if (!first) {
                sb.append("/")
            }
            sb.append(ns)
            first = false
        }
        return sb.toString()
    }


    /**
     * @return The name with the current name scope (if any) appended. See [.withNameScope]
     */
    fun nameWithScope(name: String): String {
        val scope: String = currentNameScope() ?: return name
        return if (!name.startsWith("$scope/")) "$scope/$name" else name
    }

    /**
     * Generate a new, distinct op name of the form &lt;base&gt;_#.
     *
     *
     * Applies name scope if active.
     *
     * @param base  The base name to use
     * @param force Whether to force the result name to be the same as base.
     */
    fun getOpName(base: String, force: Boolean): String {
        var base = base
        base = nameWithScope(base)!!
        require(!(force && ops.contains(base))) { "Op with name \"$base\" already exists" }
        if (force) return base
        var start = 1

        // if we already have a name like "op_2", start from trying "op_3"
        if (base.contains("_") && base.matches(Regex(".*_\\d+"))) {
            // extract number used to generate base
            val num = Pattern.compile("(.*)_(\\d+)").matcher(base)
            // extract argIndex used to generate base
            if (num.find()) {
                start = num.group(2).toInt()
                base = num.group(1)
            }
        }
        var name = base
        var i = start
        while (true) {


            // ensure that there are no variables that look like they are outputs of this op
            var varWithName = false
            for (varName in variables) if (varName.startsWith("$name:") || varName == name) varWithName = true
            if (!ops.contains(name) && !varWithName) break
            name = base + "_" + i
            i++
        }
        return name
    }

    /**
     * See [.getOpName]
     * force is false
     */
    fun getOpName(base: String): String? {
        return getOpName(base = base, force = false)
    }



    /**
     * Generate a new, distinct variable name of the form &lt;base&gt;_#[:#].
     *
     *
     * Applies name scopes if active.
     *
     * @param base       The base of the name.
     * @param argIndex   The argument index, used in the ":#".  A value of 0 (or negative) does not include the ":#" part.
     * @param existingOp Whether to generate an distinct operation name from base (if false), or just use base (if true).
     */
    fun generateNewVarName(base: String, argIndex: Int, existingOp: Boolean): String {
        var base = base
        var argIndex = argIndex
        base = nameWithScope(base)!!
        if (argIndex > 0 && base.contains(":")) {
            val num = Pattern.compile("(.*):(\\d+)").matcher(base)
            // extract argIndex used to generate base
            if (num.find()) {
                argIndex = num.group(2).toInt() + 1
                base = num.group(1)
            }
        }
        if (!existingOp) base = getOpName(base)!!
        if (argIndex > 0) base += ":$argIndex"
        require(!variables.contains(base)) { "Variable with name \"$base\" already exists" }
        return base
    }

    /**
     * See [.generateNewVarName]
     * existingOp is true.
     */
    fun generateNewVarName(base: String, argIndex: Int): String {
        return generateNewVarName(base, argIndex, true)
    }


}
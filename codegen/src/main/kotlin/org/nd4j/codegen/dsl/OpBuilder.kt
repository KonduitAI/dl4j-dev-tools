package org.nd4j.codegen.dsl

import org.nd4j.codegen.api.*
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.api.doc.DocSection

fun Namespace(name: String, block: NamespaceOps.() -> Unit): NamespaceOps {
    val ns = NamespaceOps()
    ns.name = name
    ns.ops = mutableListOf()
    ns.block()

    return ns;
}

fun NamespaceOps.Op(name: String, block: Op.() -> Unit): Op {
    val op = Op()
    op.opName = name
    op.libnd4jOpName = name
    op.isAbstract = false
    op.inputs = mutableListOf()
    op.outputs = mutableListOf()
    op.doc = mutableListOf()
    op.constraints = mutableListOf()
    op.args = mutableListOf()
    op.signatures = mutableListOf()
    op.legacy = null

    op.block()

    if(!op.isAbstract && op.signatures!!.isEmpty()){
        op.AllParamSignature()
        op.AllDefaultsSignature()
    }

    op.checkInvariants()

    this.ops!!.add(op)
    return op
}

fun NamespaceOps.Op(name: String, extends: Op, block: (Op.() -> Unit)? = null): Op {
    return Op(name) {
        extendsOp = extends
        inputs = null
        outputs = null
        doc = null
        constraints = null
        args = null
        signatures = null
        if (block != null) block()
    }
}

fun Op.Input(dataType: DataType, name: String, block: (Input.() -> Unit)? = null): Input {
    val input = Input()
    input.name = name
    input.type = dataType
    if (block != null) input.block()

    this.addInput(input)
    return input
}

fun Op.Arg(dataType: DataType, name: String, block: (Arg.() -> Unit)? = null): Arg {
    val input = Arg()
    input.name = name
    input.type = dataType
    if (block != null) input.block()

    this.addArgument(input)
    return input
}

fun Op.Output(dataType: DataType, name: String, block: (Output.() -> Unit)? = null): Output {
    val output = Output()
    output.name = name
    output.type = dataType
    if (block != null) output.block()
    this.addOutput(output)
    return output
}

fun Op.Doc(language: Language, scope: DocScope, block: DocSection.() -> String): DocSection {
    val doc = DocSection().apply {
        this.language = language
        this.scope = scope
        text = this.block()
    }
    this.addDoc(doc)
    return doc
}

fun Op.AllParamSignature(withOutput: Boolean = true) {
    val allParameters = mutableListOf<Parameter>()
    allParameters.addAll(this.inputs!!)
    allParameters.addAll(this.args!!)
    this.addSignature(Signature(allParameters))
    if(withOutput){
        val withOutputParams = mutableListOf<Parameter>()
        withOutputParams.addAll(this.outputs!!)
        withOutputParams.addAll(allParameters)
        this.addSignature(Signature(withOutputParams))
    }
}

fun Op.AllDefaultsSignature(withOutput: Boolean = true) {
    val allParameters = mutableListOf<Parameter>().also{
        it.addAll(this.inputs!!)
        it.addAll(this.args!!)
    }

    if (!allParameters.all { it.defaultValue() == null }) {
        val params = allParameters.filter{ it.defaultValue() == null }
        this.addSignature(Signature(params))
        if(withOutput){
            val withOutputParams = mutableListOf<Parameter>()
            withOutputParams.addAll(this.outputs!!)
            withOutputParams.addAll(params)
            this.addSignature(Signature(withOutputParams))
        }
    }
}

fun Op.Signature(vararg params: Parameter, block: (Signature.() -> String)? = null): Signature {
    if(params.toSet().size < params.size){
        throw IllegalArgumentException("A parameter may not be used twice in a signature!")
    }
    val signature = Signature(params.toList())
    if(block != null){
        signature.block()
    }
    this.addSignature(signature)
    return signature
}

fun Op.Constraint(desc: String, block: ConstraintBuilder.() -> Expression): Constraint {
    val check = ConstraintBuilder().block()
    val constraint = Constraint()
    constraint.message = desc
    constraint.check = check
    this.addConstraint(constraint)
    return constraint
}

fun Op.BackendConstraint(desc: String, block: ConstraintBuilder.() -> Expression): Constraint {
    val check = ConstraintBuilder().block()
    val constraint = BackendConstraint()
    constraint.message = desc
    constraint.check = check
    this.addConstraint(constraint)
    return constraint
}

class ConstraintBuilder {
    fun broadcastableShapes(vararg inputs: Input) = BroadcastableShapesExpression(inputs.toList())
    fun sameShape(vararg inputs: Input) = SameShapeExpression(inputs.toList())
    fun sameType(vararg inputs: Input) = SameTypeExpression(inputs.toList())

    fun Input.sizeAt(i: Int) = InputShapeReference(this, i)
    fun Input.rank() = InputRankReference(this)
    fun Input.isScalar() = this.rank() eq 1

    fun some(expr: BooleanExpression, vararg exprs: BooleanExpression) = exprs.fold(expr, {acc, cur -> acc or cur} )
    fun all(expr: BooleanExpression, vararg exprs: BooleanExpression) = exprs.fold(expr, {acc, cur -> acc and cur} )
    fun not(expr: BooleanExpression) = expr eq false

    infix fun BooleanExpression.and(other: BooleanExpression) = BooleanExpression(this, other, BooleanOperation.AND)
    infix fun BooleanExpression.or(other: BooleanExpression) = BooleanExpression(this, other, BooleanOperation.OR)


    infix fun Reference.eq(other: Reference) = BooleanExpression(this, other, BooleanOperation.EQ)
    infix fun Reference.eq(other: Number) = this eq NumberReference(other)
    infix fun Reference.eq(other: Boolean) = this eq BooleanReference(other)


    infix fun Reference.neq(other: Reference) = BooleanExpression(this, other, BooleanOperation.NEQ)
    infix fun <T: Number> Reference.neq(other: T) = this neq NumberReference(other)
    infix fun Reference.neq(other: Boolean) = this neq BooleanReference(other)

    infix fun Reference.gt(other: Reference) = BooleanExpression(this, other, BooleanOperation.GT)
    infix fun <T: Number> Reference.gt(other: T) = this gt NumberReference(other)

    infix fun Reference.lt(other: Reference) = BooleanExpression(this, other, BooleanOperation.LT)
    infix fun <T: Number> Reference.lt(other: T) = this lt NumberReference(other)


    infix fun <T: Number> Reference.gte(other: T) = this gte NumberReference(other)
    infix fun Reference.gte(other: Reference) = BooleanExpression(this, other, BooleanOperation.GTE)

    infix fun <T: Number> Reference.lte(other: T) = this lte NumberReference(other)
    infix fun Reference.lte(other: Reference) = BooleanExpression(this, other, BooleanOperation.LTE)
}
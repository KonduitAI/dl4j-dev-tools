package org.nd4j.codegen.api


open class Constraint (
        var message: String? = null,
        var check: Expression? = null
)

class BackendConstraint(message: String? = null, check: Expression? = null): Constraint(message, check)

sealed class Reference
data class NumberReference<T: Number>(val value: T): Reference()
data class BooleanReference(val value: Boolean): Reference()
data class InputShapeReference(val input: Input, val idx: Int): Reference()
data class InputRankReference(val input: Input): Reference()
sealed class Expression: Reference()
data class BooleanExpression(val left: Reference, val right: Reference, val op: BooleanOperation): Expression()
data class SameTypeExpression(val inputs: List<Input>): Expression()
data class SameShapeExpression(val inputs: List<Input>): Expression()
data class BroadcastableShapesExpression(val inputs: List<Input>): Expression()
enum class BooleanOperation{EQ, NEQ, LT, LTE, GT, GTE, AND, OR}

sealed class Count
data class Range(val from: Int, val to: Int): Count()
data class AtLeast(val min: Int): Count()
data class AtMost(val max: Int): Count()
data class Exactly(val count: Int): Count()

data class Arg(
        var name: String? = null,
        var type: DataType? = null,
        var optional: Boolean = false,
        var description: String? = null,
        var count: Count? = null
) : Reference()

data class Input (
        var name: String? = null,
        var type: DataType? = null,
        var optional: Boolean = false,
        var description: String? = null,
        var count: Count? = null,
        var inPlace: Boolean = false,
        var supportsInPlaceInit: Boolean = false
)

data class Output(
        var name: String? = null,
        var type: DataType? = null,
        var description: String? = null
)
package org.nd4j.codegen.api


open class Constraint (
        var message: String? = null,
        var check: Expression? = null
)

class BackendConstraint(message: String? = null, check: Expression? = null): Constraint(message, check)

// Used in Constraint Expressions
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


// Used to define array sizes
sealed class Count
data class Range(val from: Int, val to: Int): Count()
data class AtLeast(val min: Int): Count()
data class AtMost(val max: Int): Count()
data class Exactly(val count: Int): Count()

// Actual parameters
interface Parameter {
    fun name(): String
    fun defaultValue() : Any?

    /**
     * A default value only is applicable if it is a literal value, or the referenced value is either directly a part of
     * the signature, or there is a reference chain that ends in something that is actually a part of the signature
     */
    fun defaultValueIsApplicable(otherParams: List<Parameter>): Boolean {
        val defaultValue = this.defaultValue()
        return when(defaultValue){
            null -> false
            is Number, is Boolean -> true
            is Parameter -> otherParams.contains(defaultValue) || defaultValue.defaultValueIsApplicable(otherParams)
            is TensorDataTypeValue -> otherParams.contains(defaultValue.tensor) || defaultValue.tensor.defaultValueIsApplicable(otherParams)
            is TensorShapeValue -> otherParams.contains(defaultValue.tensor) || defaultValue.tensor.defaultValueIsApplicable(otherParams)
            else -> false
        }
    }
}
interface Tensor: Parameter

data class Arg(
        var name: String? = null,
        var type: DataType? = null,
        var description: String? = null,
        var count: Count? = null
) : Reference(), Parameter {
    override fun name(): String = name!!
    override fun defaultValue(): Any? = defaultValue

    var defaultValue: Any? = null
        set(value) = if(isAssignableFrom(value) /*|| value == null*/) {
            field = value
        }else{
            throw IllegalArgumentException("Illegal default value for Arg($type, $name). Got $value (${value?.javaClass?.name})")
        }

    private fun matchesDataType(value: Any?) = when(type){
        DataType.FLOATING_POINT -> value is Double
        DataType.INT -> (value is Int) || (value is Long)
        DataType.NUMERIC -> value is Number
        DataType.BOOL -> value is Boolean
        else -> false
    }

    private fun isAssignableFrom(value: Any?) = when(value){
        is TensorShapeValue -> count != Exactly(1) && count != null && type == DataType.INT
        is TensorDataTypeValue -> type == DataType.DATA_TYPE
        is Number, is Boolean -> matchesDataType(value)
        is Arg -> value.count == count && value.type == type
        else -> false
    }

    fun Tensor.shape() = TensorShapeValue(this)
    fun Tensor.dataType() = TensorDataTypeValue(this)
}

data class Input (
        var name: String? = null,
        var type: DataType? = null,
        var description: String? = null,
        var count: Count? = null
) : Parameter, Tensor {
    override fun name(): String = name!!
    override fun defaultValue(): Any? = defaultValue

    var defaultValue: Input? = null
        set(value) = if(matchesDataType(value)){
            field = value
        }else{
            throw IllegalArgumentException("Illegal default value for Input($name). Allowed values have to match data type $type, but got $value (${value?.javaClass?.name})")
        }

    private fun matchesDataType(value: Input?) = value?.type == type
}

data class Output(
        var name: String? = null,
        var type: DataType? = null,
        var description: String? = null
) : Parameter, Tensor{
    override fun name(): String = name!!
    override fun defaultValue(): Any? = null
}

data class Signature(
        val parameters: List<Parameter>,
        val description: String? = null
){
    override fun toString(): String {
        return "Signature(${parameters.joinToString {it.name()}})"
    }
}

// Used in defining default values
data class TensorShapeValue(val tensor: Tensor) {
    override fun toString(): String = "${tensor.name()}.shape()"
}
data class TensorDataTypeValue(val tensor: Tensor){
    override fun toString(): String = "${tensor.name()}.dataType()"
}
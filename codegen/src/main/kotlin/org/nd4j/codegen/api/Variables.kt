package org.nd4j.codegen.api

import java.util.*


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

    fun hasDefaultValue(): Boolean

    /**
     * A default value only is applicable if it is a literal value, or the referenced value is either directly a part of
     * the signature, or there is a reference chain that ends in something that is actually a part of the signature
     */
    fun defaultValueIsApplicable(otherParams: List<Parameter>): Boolean = if(hasDefaultValue()){
        when(val defaultValue = this.defaultValue()){
            is Number, is Boolean, null -> true
            is IntArray, is BooleanArray, is DoubleArray -> true
            is Parameter -> otherParams.contains(defaultValue) || defaultValue.defaultValueIsApplicable(otherParams)
            is TensorDataTypeValue -> otherParams.contains(defaultValue.tensor) || defaultValue.tensor.defaultValueIsApplicable(otherParams)
            is TensorShapeValue -> otherParams.contains(defaultValue.tensor) || defaultValue.tensor.defaultValueIsApplicable(otherParams)
            else -> false
        }
    }else{
        false
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
    override fun hasDefaultValue(): Boolean = defaultValueIsSet

    private var defaultValueIsSet = false
    var defaultValue: Any? = null
        set(value) = if(isAssignableFrom(value)) {
            field = value
            defaultValueIsSet = true
        }else{
            throw IllegalArgumentException("Illegal default value for Arg($type, $name)${if(count != null) "{ count = $count }" else "" }. Got ${value.toDescriptiveString()} (${value?.javaClass?.name})")
        }

    private fun matchesDataType(value: Any?) = when(type){
        DataType.FLOATING_POINT -> value is Double
        DataType.INT -> (value is Int) || (value is Long)
        DataType.NUMERIC -> value is Number
        DataType.BOOL -> value is Boolean
        else -> false
    }

    private fun isAssignableFrom(value: Any?) = when(value){
        is TensorShapeValue -> isArray() && type == DataType.INT
        is TensorDataTypeValue -> type == DataType.DATA_TYPE
        is Number, is Boolean -> matchesDataType(value)
        is IntArray -> isArray() && (type == DataType.INT || type == DataType.NUMERIC) && countMatches(value.size)
        is DoubleArray -> isArray() && (type == DataType.FLOATING_POINT || type == DataType.NUMERIC) && countMatches(value.size)
        is BooleanArray -> isArray() && type == DataType.BOOL && countMatches(value.size)
        is Arg -> value.count == count && value.type == type
        null -> true
        else -> false
    }

    fun isArray() = count != Exactly(1) && count != null
    fun countMatches(size: Int) = when(val c = count!!){
        is Range -> c.from <= size && size <= c.to
        is AtLeast -> c.min <= size
        is AtMost -> size <= c.max
        is Exactly -> c.count == size
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
    override fun hasDefaultValue(): Boolean = defaultValueIsSet

    private var defaultValueIsSet = false
    var defaultValue: Input? = null
        set(value) = if(matchesDataType(value)){
            field = value
            defaultValueIsSet = true
        }else{
            throw IllegalArgumentException("Illegal default value for Input($name). Allowed values have to match data type $type, but got ${value.toDescriptiveString()} (${value?.javaClass?.name})")
        }

    private fun matchesDataType(value: Input?) = when(value){
        null -> true
        else -> value.type == type
    }
}

data class Output(
        var name: String? = null,
        var type: DataType? = null,
        var description: String? = null
) : Parameter, Tensor{
    override fun name(): String = name!!
    override fun defaultValue(): Any? = null
    override fun hasDefaultValue(): Boolean = false
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

fun Any?.toDescriptiveString() = when(this){
    null -> "null"
    is IntArray -> Arrays.toString(this)
    is LongArray -> Arrays.toString(this)
    is DoubleArray -> Arrays.toString(this)
    is FloatArray -> Arrays.toString(this)
    is BooleanArray -> Arrays.toString(this)
    is Array<*> -> Arrays.toString(this)
    else -> this.toString()
}
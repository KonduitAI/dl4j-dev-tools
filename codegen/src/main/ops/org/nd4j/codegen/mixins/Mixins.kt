package org.nd4j.codegen.mixins

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.DataType
import org.nd4j.codegen.dsl.*

val transform = Mixin("transform"){
    legacy = true
    Input(DataType.NUMERIC, "x") { description = "Input variable" }
    Output(DataType.NUMERIC, "output"){ description = "Output variable" }
}

val transformStrict = Mixin("transformStrict"){
    useMixin(transform)
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
}

val transformSame = Mixin("transformSame"){
    useMixin(transform)
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
}

val transformBool = Mixin("transformBool"){
    useMixin(transform)
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.bool"
}

val transformAny = Mixin("transformAny"){
    useMixin(transform)
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.any"
}

val transformFloating = Mixin("transformFloating"){
    useMixin(transform)
    javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.floating"
}

val scalar = Mixin("scalar"){
    legacy = true
    javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
    Input(DataType.NUMERIC, "x") { description = "Input variable" }
    Arg(DataType.NUMERIC, "value") { description = "Scalar value for op" }
    Output(DataType.NUMERIC, "output"){ description = "Output variable" }
}

val reduce = Mixin("reduce"){
    legacy = true
    Input(DataType.NUMERIC, "in") { description = "Input variable" }
    Arg(DataType.INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
    Output(DataType.NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
}

val reduceFloating = Mixin("reduceFloating"){
    useMixin(reduce)
    javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
}

val reduceSame = Mixin("reduceSame"){
    useMixin(reduce)
    javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.same"
}

val reduceLong = Mixin("reduceLong"){
    useMixin(reduce)
    javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.longer"
}

val reduce3 = Mixin("reduce3"){
    legacy = true
    javaPackage = "org.nd4j.linalg.api.ops.impl.reduce3"
    Input(DataType.NUMERIC, "x") { description = "Input variable x" }
    Input(DataType.NUMERIC, "y") { description = "Input variable y" }
    Arg(DataType.INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to calculate %OPNAME% over" }
    Output(DataType.NUMERIC, "output"){ description = "Output variable" }
}

val indexAccum = Mixin("indexAccum"){
    legacy = true
    javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
    val input = Input(DataType.NUMERIC, "in") { description = "Input variable" }
    val keepDims = Arg(DataType.BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions"; defaultValue = false }
    val dims = Arg(DataType.INT, "dimensions"){ count = AtLeast(1); isVargarg = true; description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
    Output(DataType.NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }

    Signature(input, dims)
    AllParamSignature(withOutput = false)
}
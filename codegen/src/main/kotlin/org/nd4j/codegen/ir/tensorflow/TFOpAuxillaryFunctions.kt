package org.nd4j.codegen.ir.tensorflow

import org.nd4j.codegen.ir.ArgDescriptor
import org.nd4j.codegen.ir.AttributeMappingRule
import org.nd4j.ir.OpNamespace
import org.tensorflow.framework.*

fun booleanConstant(inputName: String, constantValue: Boolean,argumentIndex: Int): List<TensorflowArgDescriptorConstant> {
        return listOf(argDescriptorConstant(listOf(
            ArgDescriptor {
                name = inputName
                boolValue = constantValue
                argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                argIndex = argumentIndex
            }
        )))
}

fun doubleConstant(inputName: String, constantValue: Double, argumentIndex: Int): List<TensorflowArgDescriptorConstant> {
        return listOf(argDescriptorConstant(listOf(
            ArgDescriptor {
                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                name = inputName
                doubleValue = constantValue
                argIndex = argumentIndex
            }
        )))
}

fun intConstant(inputName: String, constantValue: Integer, argumentIndex: Int): List<TensorflowArgDescriptorConstant> {
        return listOf(argDescriptorConstant(listOf(
            ArgDescriptor {
                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                name = inputName
                int64Value = constantValue.toLong()
                argIndex = argumentIndex
            }
        )))
}

fun mapSameName(names: List<String>): List<NDArrayMappingRule> {
        return listOf(mappingNDArrayInputs(names.map { name -> Pair(name, name) }.toMap().toMutableMap()))
}

fun mapTensorNamesWithOp(inputFrameworkOpName: String,
                         opName: String,
                         tensorNames: MutableMap<String,String>,
                         attributeMappingRules: List<AttributeMappingRule<GraphDef, OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>> = emptyList()): TensorflowMappingProcess {
        return TensorflowMappingProcess(
            opName = opName,
            inputFrameworkOpName = inputFrameworkOpName,
            opMappingRegistry = tensorflowOpRegistry,
            tensorMappingRules = listOf(mappingNDArrayInputs(tensorNames)),
            attributeMappingRules = attributeMappingRules
        )

}

fun multipleNameMapping(inputFrameworkOpNames: List<String>,
                        opName: String, tensorNames: MutableMap<String, String>,
                        attributeMappingRules: List<AttributeMappingRule<GraphDef, OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>> = emptyList()):
        List<TensorflowMappingProcess> {
        return inputFrameworkOpNames.map {
            mapTensorNamesWithOp(
                inputFrameworkOpName = it,
                opName = opName,
                tensorNames = tensorNames,
                attributeMappingRules = attributeMappingRules
            )
        }
}

fun defineBiasAdd(names :List<String> =  listOf("BiasAdd","BiasAddV1")) {
        names.forEach {
            TensorflowMappingProcess(
                opName = "biasadd",
                inputFrameworkOpName = it,
                opMappingRegistry = tensorflowOpRegistry,
                tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "value", "bias" to "bias"))),
                attributeMappingRules = booleanConstant(inputName = "nchw", constantValue = false, argumentIndex = 0)

            )
        }
}

fun defineTensorflowSingleTransform(inputOpName: String, inputFrameworkOpName: String): TensorflowMappingProcess {
        return TensorflowMappingProcess(
            opName = inputOpName,
            inputFrameworkOpName = inputFrameworkOpName, tensorMappingRules = listOf(
                NDArrayMappingRule(
                    mappingNamesToPerform = mutableMapOf("input" to "x")
                )
            ),
            attributeMappingRules = listOf(argDescriptorConstant(
                listOf(
                    ArgDescriptor {
                        name = "inPlace"
                        boolValue = false
                        argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                        argIndex = 0
                    }
                )
            )),
            opMappingRegistry = tensorflowOpRegistry)

}

fun defineSingularReduce(inputFrameworkOpName: String, inputOpName: String): TensorflowMappingProcess {
        return mapTensorNamesWithOp(
            inputFrameworkOpName = inputFrameworkOpName,
            opName = inputOpName,
            attributeMappingRules = listOf(
                valueMapping(mutableMapOf("keepDims" to "keep_dims")),
                ndarrayToIntList(mutableMapOf("dimensions" to "reduction_indices"))
            ),
            tensorNames = mutableMapOf("input" to "input")
        )
}

fun defineTensorflowPairwiseTransforms(opName: String, inputFrameworkOpName: String,
                                       firstOutputName: String = "input",
                                       secondOutputName: String = "y",
                                       firstInput: String = "x", secondInput: String = "y") : TensorflowMappingProcess {
        return TensorflowMappingProcess(
            opName = opName,
            tensorMappingRules = listOf(
                NDArrayMappingRule(
                    mappingNamesToPerform = mutableMapOf(
                        firstOutputName to firstInput,
                        secondOutputName to secondInput
                    )
                )
            ),
            inputFrameworkOpName = inputFrameworkOpName,
            inputFramework = "tensorflow",
            attributeMappingRules = booleanConstant(inputName = "inPlace", constantValue = false, argumentIndex = 0),
            opMappingRegistry = tensorflowOpRegistry
        )
}

fun defineBoundingBoxes(listOfNames: List<String> = listOf("DrawBoundingBoxes")) {
        listOfNames.forEach {
                val drawBoundingBoxes = TensorflowMappingProcess(
                    inputFrameworkOpName = it,
                    opName = "draw_bounding_boxes",
                    tensorMappingRules = listOf(
                        mappingNDArrayInputs(
                            mutableMapOf(
                                "images" to "images",
                                "boxes" to "boxes"
                            )
                        )
                    ),
                    opMappingRegistry = tensorflowOpRegistry
                )
        }
}
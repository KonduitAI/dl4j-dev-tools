package org.nd4j.codegen.ir

import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

abstract class BaseAttributeExtractionRule<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(
        name: String,
        mappingNamesToPerform: Map<String, String>,
        transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        AttributeMappingRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    protected var opDescriptor: OpNamespace.OpDescriptor? = null
    protected val mappingNamesToPerform = mappingNamesToPerform
    protected val transformerArgs = transformerArgs
    protected val name = name


    override fun initWithMappingProcess(mappingProcess: MappingProcess<OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>) {

    }

    override fun mappingNamesToPerform(): Map<String, String> {
        return mappingNamesToPerform
    }

    override fun name(): String {
        return name
    }

    override fun mappingTransformerArgs(): Map<String, List<OpNamespace.ArgDescriptor>> {
        return transformerArgs
    }



    abstract fun createIRAttribute(name: String, attrDef: ATTR_DEF, attributeValueType: ATTR_VALUE_TYPE): IRAttribute<ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>



    override fun serialize(): MapperNamespace.MappingRule {
        val builder = MapperNamespace.MappingRule.newBuilder()
        builder.ruleName = name()
        builder.functionName = name()
        val descriptorList = opDescriptor!!.argDescriptorList
        for((k,v) in transformerArgs) {
            val filteredList = descriptorList.filter { input -> input.name == k }
            require(filteredList.isNotEmpty()) {"Output attribute " + k + " was not found in op descriptor " + name() + " list of attribtues was " + descriptorList.map { input -> input.name }}

            val descriptor = filteredList[0]
            when(descriptor.argType) {
                OpNamespace.ArgDescriptor.ArgType.BOOL -> builder.addOutputBooleanName(k)
                OpNamespace.ArgDescriptor.ArgType.INT64 -> builder.addOutputIntName(k)
                OpNamespace.ArgDescriptor.ArgType.FLOAT -> builder.addOutputFloatName(k)
                OpNamespace.ArgDescriptor.ArgType.DOUBLE -> builder.addOutputDoubleName(k)
                OpNamespace.ArgDescriptor.ArgType.INT64 -> builder.addOutputIntName(k)
                OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR -> builder.addInputTensorName(k)
                OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR -> builder.addOutputTensorName(k)
            }

            for(associatedInput in v) {
                when(associatedInput.argType) {
                    AttributeValueType.STRING -> builder.addInputStringAttrName(associatedInput.name)
                    AttributeValueType.BOOL -> builder.addInputBooleanName(associatedInput.name)
                    AttributeValueType.FLOAT -> builder.addInputFloatName(associatedInput.name)
                    AttributeValueType.INT -> builder.addInputIntName(associatedInput.name)
                    AttributeValueType.TENSOR -> builder.addInputTensorName(associatedInput.name)
                }
            }



        }


        return builder.build()
    }
}

abstract class StringEqualsAdapterRule<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(
        mappingNamesToPerform: Map<String, String> = emptyMap(),
        transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "stringequals",
                mappingNamesToPerform =  mappingNamesToPerform,
                transformerArgs = transformerArgs)
        where DATA_TYPE: ProtocolMessageEnum {

    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE,DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = transformerArgs[k]
            val compString = descriptorForName!![0].stringValue
            val testValue = mappingCtx.irAttributeValueForNode(v).stringValue()
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = v
            descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.BOOL
            descriptorBuilder.boolValue = testValue == compString
            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}


abstract class StringContainsAdapterRule<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(
        mappingNamesToPerform: Map<String, String> = emptyMap(),
        transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "stringcontains",
                mappingNamesToPerform =  mappingNamesToPerform,
                transformerArgs = transformerArgs)
        where DATA_TYPE: ProtocolMessageEnum {

    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE,DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = transformerArgs[k]
            val compString = descriptorForName!![0].stringValue
            val testValue = mappingCtx.irAttributeValueForNode(v).stringValue()
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = v
            descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.BOOL
            descriptorBuilder.boolValue = compString.contains(testValue)
            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}

abstract class StringNotEqualsAdapterRule<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(
        mappingNamesToPerform: Map<String, String> = emptyMap(),
        transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "sizethresholdarrayint",
                mappingNamesToPerform =  mappingNamesToPerform,
                transformerArgs = transformerArgs)
        where DATA_TYPE: ProtocolMessageEnum {

    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE,DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = transformerArgs[k]
            val compString = descriptorForName!![0].stringValue
            val testValue = mappingCtx.irAttributeValueForNode(v).stringValue()
            val argDescriptor = ArgDescriptor {
                name = v
                argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                boolValue = testValue != compString
            }

            ret.add(argDescriptor)

        }
        return ret
    }
}

abstract class SizeThresholdIntArrayIntIndexRule<OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(mappingNamesToPerform: Map<String, String>,
                                                     transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "sizethresholdarrayint", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) where DATA_TYPE: ProtocolMessageEnum {



    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE,DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = transformerArgs[k]
            val inputArr = mappingCtx.irAttributeValueForNode(v).listIntValue()
            val index = descriptorForName!![0].int32Value
            val sizeThreshold = descriptorForName!![1].int64Value
            val fallbackIndex = descriptorForName!![2].stringValue
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = v
            descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.INT64
            if(inputArr.size < sizeThreshold) {
                descriptorBuilder.int64Value = inputArr[fallbackIndex.toInt()]
            } else {
                descriptorBuilder.int64Value = inputArr[index.toInt()]
            }

            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}

abstract class ConditionalFieldValueIntIndexArrayRule<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(mappingNamesToPerform: Map<String, String>,
                                                     transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "conditionalfieldvalueintindex", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs)
        where  DATA_TYPE: ProtocolMessageEnum {


    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE,DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val listOfArgs  = transformerArgs[k]
            val inputArr = mappingCtx.irAttributeValueForNode(listOfArgs!![3].stringValue).listIntValue()
            val trueIndex = listOfArgs!![1].int32Value
            val falseIndex = listOfArgs!![2].int32Value
            val targetValueToTest = listOfArgs!![0].stringValue
            val testValue = mappingCtx.irAttributeValueForNode(v).stringValue()
            val intValueToSet = if (testValue == targetValueToTest)  inputArr[trueIndex] else inputArr[falseIndex]
            ret.add(ArgDescriptor {
                name  = v
                int64Value = intValueToSet
                argType = OpNamespace.ArgDescriptor.ArgType.INT64
            })

        }
        return ret
    }
}


/**
 * Need to implement tensor size extraction value at index
 */


abstract class NDArraySizeAtRule<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(mappingNamesToPerform: Map<String, String>,
                                                     transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "ndarraysizeat", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs)
        where  DATA_TYPE: ProtocolMessageEnum {


    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE,DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        mappingNamesToPerform().forEach { (k, v) ->
            val transformArgsForAttribute = transformerArgs[k]
            //note that this finds a value for a named tensor within either the graph or the node
            //some frameworks may have a value node with a value attribute
            //others may have the actual tensor value
            val inputArr = mappingCtx.tensorInputFor(v)
            val sizeIndex = transformArgsForAttribute!![0].int32Value
            val sizeAt = inputArr.shape()[sizeIndex]
            val argDescriptor = ArgDescriptor {
                name = v
                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                int64Value = sizeAt
            }
            ret.add(argDescriptor)
        }

        return ret
    }
}


/**
 * Need to implement tensor size extraction value at index
 */


abstract class ValueMapping<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(mappingNamesToPerform: Map<String, String>,
                                                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "valuemapping", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {


    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE,DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = k
            val irAttribute = mappingCtx.irAttributeValueForNode(v)
            when(irAttribute.attributeValueType()) {
                AttributeValueType.INT -> {
                    descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.INT64
                    descriptorBuilder.int64Value = irAttribute.intValue()
                }

                AttributeValueType.FLOAT -> {
                    descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.FLOAT
                    descriptorBuilder.floatValue = irAttribute.floatValue()
                }

                AttributeValueType.BOOL -> {
                    descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                    descriptorBuilder.boolValue = irAttribute.boolValue()
                }

                AttributeValueType.STRING -> {
                    descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.STRING
                    descriptorBuilder.stringValue = irAttribute.stringValue()

                }

                else -> {
                    throw IllegalArgumentException("Unable to map value $k. Please use different rule for list values and tensors.")
                }
            }


            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}


abstract class BooleanToInt<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(mappingNamesToPerform: Map<String, String>,
                                                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "booleantoint", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {


    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE,DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = k
            val irAttribute = mappingCtx.irAttributeValueForNode(v)
            when(irAttribute.attributeValueType()) {
                AttributeValueType.INT -> {
                    descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.INT64
                    descriptorBuilder.int64Value = irAttribute.intValue()
                }


                AttributeValueType.BOOL -> {
                    descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.INT64
                    descriptorBuilder.int64Value = if (irAttribute.boolValue()) 1 else 0
                }


                else -> {
                    throw IllegalArgumentException("Unable to map value $k. Please use different rule for list values and tensors.")
                }
            }


            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}


abstract class StringToIndex<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(mappingNamesToPerform: Map<String, String>,
                                                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "stringtoindex", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {
    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val listOfValues = (transformerArgs[v] ?: error("")).map { argDescriptor -> argDescriptor.stringValue }
            val stringValIndex = mappingCtx.irAttributeValueForNode(v).stringValue()
            val argDescriptor = ArgDescriptor {
                name = k
                int64Value = listOfValues.indexOf(stringValIndex).toLong()
            }

            ret.add(argDescriptor)

        }

        return ret
    }
}


abstract class ListAttributeValueLookupToIndex<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(mappingNamesToPerform: Map<String, String>,
                                                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "listattributevaluelookuptoindex", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {
    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val index = (transformerArgs[v] ?: error(""))[0]!!.int64Value
            val listOfValues = mappingCtx.irAttributeValueForNode(k)
            when(listOfValues.attributeValueType()) {
                AttributeValueType.LIST_FLOAT -> {
                    val listFloat = listOfValues.listFloatValue()
                    val argDescriptor = ArgDescriptor {
                        name = k
                        floatValue = listFloat[index.toInt()]
                    }

                    ret.add(argDescriptor)
                }
                AttributeValueType.LIST_INT -> {
                    val listInt = listOfValues.listIntValue()
                    val argDescriptor = ArgDescriptor {
                        name = k
                        int64Value = listInt[index.toInt()]
                    }

                    ret.add(argDescriptor)
                }

                AttributeValueType.LIST_STRING -> {
                    val listString = listOfValues.listStringValue()
                    val argDescriptor = ArgDescriptor {
                        name = k
                        stringValue = listString[index.toInt()]
                    }

                    ret.add(argDescriptor)
                }

                AttributeValueType.LIST_TENSOR -> {
                    val listTensor = listOfValues.listTensorValue()
                    val argDescriptor = ArgDescriptor {
                        name = k
                        inputValue = listTensor[index.toInt()].toArgTensor()
                    }

                    ret.add(argDescriptor)
                }

                AttributeValueType.LIST_BOOL -> {
                    val listBool = listOfValues.listBoolValue()
                    val argDescriptor = ArgDescriptor {
                        name = k
                        boolValue = listBool[index.toInt()]
                    }

                    ret.add(argDescriptor)
                }

            }


        }

        return ret
    }
}


abstract class AttributeNumberListNDArray<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(mappingNamesToPerform: Map<String, String>,
                                                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "attributenumberlistndarrayinput", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {
    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val irAttribute = mappingCtx.irAttributeValueForNode(v)
            when(irAttribute.attributeValueType()) {
                AttributeValueType.LIST_FLOAT -> {
                    ret.add(ArgDescriptor {
                        argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                        name = k
                        inputValue = nameSpaceTensorFromNDarray(Nd4j.create(irAttribute.listFloatValue()))
                    })
                }

                AttributeValueType.LIST_INT -> {
                    ret.add(ArgDescriptor {
                        argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                        name = k
                        inputValue = nameSpaceTensorFromNDarray(Nd4j.create(irAttribute.listIntValue()))
                    })
                }

            }

        }

        return ret
    }
}

abstract class ListNumberToListNumber<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(mappingNamesToPerform: Map<String, String>,
                                                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "listnumbertolistnumber", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {
    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val listOfValues = (transformerArgs[v] ?: error(""))
            when(listOfValues[0].argType) {
                OpNamespace.ArgDescriptor.ArgType.INT64, OpNamespace.ArgDescriptor.ArgType.INT32 -> {
                    val listOfValuesNumbers = listOfValues.map { argDescriptor -> argDescriptor.int64Value }
                    listOfValuesNumbers.forEachIndexed { index, element ->
                        val argDescriptor = ArgDescriptor {
                            name = k + "$index"
                            int64Value = element
                        }

                        ret.add(argDescriptor)
                    }
                }

                OpNamespace.ArgDescriptor.ArgType.FLOAT, OpNamespace.ArgDescriptor.ArgType.DOUBLE -> {
                    val listOfValuesNumbers = listOfValues.map { argDescriptor -> argDescriptor.doubleValue }
                    listOfValuesNumbers.forEachIndexed { index, element ->
                        val argDescriptor = ArgDescriptor {
                            name = k + "$index"
                            doubleValue = element
                        }

                        ret.add(argDescriptor)
                    }

                }

            }


        }

        return ret
    }
}


abstract class NDArrayInputToScalarAttribute<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(mappingNamesToPerform: Map<String, String>,
                                                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "ndarrayinputtoscalarattribute", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {
    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val inputTensor = mappingCtx.tensorInputFor(v).toNd4jNDArray()
            when(inputTensor.dataType()) {
                DataType.FLOAT,DataType.DOUBLE -> {
                    val floatVal = inputTensor.getDouble(0)
                    ret.add(ArgDescriptor {
                        name = k
                        argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                        doubleValue = floatVal
                    })
                }

                DataType.UINT64,DataType.INT32,DataType.UINT32,DataType.INT64 -> {
                    val intVal = inputTensor.getInt(0)
                    ret.add(ArgDescriptor {
                        name = k
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        int64Value = intVal.toLong()
                    })
                }
                else -> {
                    throw java.lang.IllegalArgumentException("Attribute $k is invalid type ${inputTensor.dataType()}")
                }
            }

        }

        return ret
    }
}


abstract class AttributeScalarNDArrayAttribute<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(mappingNamesToPerform: Map<String, String>,
                                                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "attributescalarndarrayattribute", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {
    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val irAttribute = mappingCtx.irAttributeValueForNode(v)
            when(irAttribute.attributeValueType()) {
                AttributeValueType.FLOAT -> {
                    ret.add(ArgDescriptor {
                        argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                        name = k
                        inputValue = nameSpaceTensorFromNDarray(Nd4j.scalar(irAttribute.floatValue()).reshape(1,1))
                    })
                }

                AttributeValueType.INT -> {
                    ret.add(ArgDescriptor {
                        argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                        name = k
                        inputValue = nameSpaceTensorFromNDarray(Nd4j.scalar(irAttribute.intValue()).reshape(1,1))
                    })
                }
                else -> {
                    throw IllegalArgumentException("Attribute $v is not a valid type. Type was ${irAttribute.attributeValueType()}")
                }

            }

        }

        return ret
    }
}




abstract class NDArrayToIntAttributeValue<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(mappingNamesToPerform: Map<String, String>,
                                                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "ndarraytointattributevalue", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {

    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val ndarray = mappingCtx.tensorInputFor(v).toNd4jNDArray()
            val arrInts = ndarray.toIntVector()
            for(i in 0 .. ndarray.length()) {
                val argDescriptor = ArgDescriptor {
                    name = k
                    int64Value = arrInts[i.toInt()].toLong()
                }

                ret.add(argDescriptor)
            }
        }

        return ret
    }
}



abstract class BaseNDArrayMappingRule<OP_DEF_TYPE: GeneratedMessageV3
        ,NODE_DEF_TYPE: GeneratedMessageV3,ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3,
        DATA_TYPE>(mappingNamesToPerform: MutableMap<String, String> = mutableMapOf(),
                   transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()):
        TensorMappingRule<OP_DEF_TYPE,NODE_DEF_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    protected var opDescriptor: OpNamespace.OpDescriptor? = null
    protected  val mappingNamesToPerform = mappingNamesToPerform
    protected val transformerArgs = transformerArgs
    protected var mappingProcess: MappingProcess<OP_DEF_TYPE,NODE_DEF_TYPE,TENSOR_TYPE,ATTR_DEF,ATTR_VALUE_TYPE,DATA_TYPE>? = null


    override fun initWithMappingProcess(mappingProcess: MappingProcess<OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>) {
        val opDescriptorList = nd4jOpDescriptors
        if(!opDescriptorList.opListList.map { it -> it.name }.contains(mappingProcess.opName())) {
            throw java.lang.IllegalArgumentException("Op name ${mappingProcess.opName()} not found!")
        }
        opDescriptor = opDescriptorList.opListList.first {
            input -> input.name ==  mappingProcess.opName()
        } ?: error("")
        this.mappingProcess = mappingProcess
    }


    operator  fun set(outputAttribute: String,inputAttribute: String) {
        mappingNamesToPerform[outputAttribute] = inputAttribute
    }

    override fun name(): String {
        return "ndarraymapping"
    }


    override fun mappingNamesToPerform(): Map<String, String> {
        return mappingNamesToPerform
    }


    override fun convertInput(): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()

        val mappingsToPerform = inputArgumentMappings()
        for(i in 0 until opDescriptor!!.argDescriptorCount) {
            if(mappingsToPerform.containsKey(opDescriptor!!.getArgDescriptor(i).name)) {
                val outputName = mappingsToPerform[mappingsToPerform[opDescriptor!!.getArgDescriptor(i).name]]
                val builder = OpNamespace.ArgDescriptor.newBuilder()
                builder.argType = opDescriptor!!.argDescriptorList[i].argType
                builder.name = outputName
                require(opDescriptor!!.argDescriptorList[i].argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR) {"Input type must be INPUT_TENSOR"}
                builder.argIndex = opDescriptor!!.argDescriptorList[i].argIndex
                ret.add(builder.build())
            }

        }

        return ret
    }

    abstract fun createTensorProto(input: TENSOR_TYPE): TensorNamespace.TensorProto


    override fun convertInputsReverse(toReverse: List<OpNamespace.ArgDescriptor>): List<TENSOR_TYPE> {
        for(argument in toReverse) {
            require(argument.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR) {"Type to reverse must be an input tensor."}
        }
        TODO("Not yet implemented")
    }

    override fun inputArgumentMappings(): Map<String, String> {
        return mappingNamesToPerform
    }

    override fun serialize(): MapperNamespace.MappingRule {
        val builder = MapperNamespace.MappingRule.newBuilder()
        builder.ruleName = name()
        builder.functionName = name()
        for((k,v) in transformerArgs) {
            val descriptor = opDescriptor!!.argDescriptorList.filter { input -> input.name == k }[0]
            when(descriptor.argType) {
                OpNamespace.ArgDescriptor.ArgType.BOOL -> builder.addOutputBooleanName(k)
                OpNamespace.ArgDescriptor.ArgType.INT64 -> builder.addOutputIntName(k)
                OpNamespace.ArgDescriptor.ArgType.FLOAT -> builder.addOutputFloatName(k)
                OpNamespace.ArgDescriptor.ArgType.DOUBLE -> builder.addOutputDoubleName(k)
                OpNamespace.ArgDescriptor.ArgType.INT64 -> builder.addOutputIntName(k)
                OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR -> builder.addInputTensorName(k)
                OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR -> builder.addOutputTensorName(k)
            }

            for(associatedInput in v) {
                when(associatedInput.argType) {
                    AttributeValueType.STRING -> builder.addInputStringAttrName(associatedInput.name)
                    AttributeValueType.BOOL -> builder.addInputBooleanName(associatedInput.name)
                    AttributeValueType.FLOAT -> builder.addInputFloatName(associatedInput.name)
                    AttributeValueType.INT -> builder.addInputIntName(associatedInput.name)
                    AttributeValueType.TENSOR -> builder.addInputTensorName(associatedInput.name)
                }
            }



        }


        return builder.build()
    }

}


abstract class ArgDescriptorConstant<
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
        BaseAttributeExtractionRule<OP_DEF_TYPE,NODE_TYPE,ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "argdescriptorconstant", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {
    override fun convertAttributes(mappingCtx: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        return transformerArgs.flatMap { it.value }
    }
}
package org.nd4j.codegen.ir.onnx

import junit.framework.Assert
import junit.framework.Assert.*
import onnx.Onnx
import onnx.OnnxMl
import org.junit.jupiter.api.Test
import org.nd4j.codegen.ir.importGraph
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.codegen.ir.tensorflow.*
import org.nd4j.ir.OpNamespace
import org.nd4j.shade.protobuf.ByteString
import org.tensorflow.framework.DataType
import java.nio.charset.Charset
import kotlin.math.exp
import kotlin.math.max
import kotlin.test.assertTrue

class TestOnnxIR {
    val declarations = OnnxOpDeclarations



    @Test
    fun testInputOutputNames() {
        val onnxOpNames = onnxOpRegistry.inputFrameworkOpNames()
        val nd4jOpNames = onnxOpRegistry.nd4jOpNames()
        onnxOpRegistry.mappingProcessNames().map {
            onnxOpRegistry.lookupOpMappingProcess(it)
        }.forEach {
            println("Beginning processing of op ${it.inputFrameworkOpName()} and nd4j op ${it.opName()}")
            assertTrue(onnxOpNames.contains(it.inputFrameworkOpName()))
            assertTrue(nd4jOpNames.contains(it.opName()))
            val nd4jOpDef = onnxOpRegistry.lookupNd4jOpDef(it.opName())
            val onnxOpDef = onnxOpRegistry.lookupInputFrameworkOpDef(it.inputFrameworkOpName())
            val inputNameArgDefs = nd4jOpDef.argDescriptorList.filter {
                    argDef -> argDef.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
            }.map { argDef -> argDef.name }

            val inputFrameworkOpDefNames = onnxOpDef.inputList

            val nd4jArgDefNames = nd4jOpDef.argDescriptorList.map { nd4jArgDef -> nd4jArgDef.name }
            val onnxAttrNames = onnxOpDef.attributeList.map { onnxAttr -> onnxAttr.name }
            it.tensorMappingRules().forEach { tensorRules ->
                println("Running tensor mapping rule ${tensorRules.name()} for op ${it.inputFrameworkOpName()} and nd4j op name ${it.opName()}")
                run {
                    tensorRules.mappingNamesToPerform().forEach { tensorRule ->
                        run {
                            println("Testing assertion for nd4j name ${tensorRule.key} and input name ${tensorRule.value}")
                            assertTrue(inputNameArgDefs.contains(tensorRule.key)) ?: error("Failed on inputArgName ${tensorRule.key}")
                            assertTrue(inputFrameworkOpDefNames.contains(tensorRule.value)) ?: error("Failed on inputArgName ${tensorRule.value}")
                        }

                    }
                }

            }

            println("Running attribute mapping rules for ${it.opName()} and input op name ${it.inputFrameworkOpName()}")
            it.attributeMappingRules().forEach { attrRule ->
                run {
                    attrRule.mappingNamesToPerform().forEach { attrMapping ->
                        run {
                            println("Testing nd4j name  ${attrMapping.key} and input framework name ${attrMapping.value}")
                            assertTrue(nd4jArgDefNames.contains(attrMapping.key) || inputNameArgDefs.contains(attrMapping.key))
                            assertTrue(onnxAttrNames.contains(attrMapping.value) || inputFrameworkOpDefNames.contains(attrMapping.value))

                        }

                    }
                }
            }

        }
    }


    @Test
    fun testOpOrdering() {
        val onnxOpNames = onnxOpRegistry.inputFrameworkOpNames()
        //TODO: list ops need to work and TopK has a data type conversion issue with the k ndarray input
        val bannedOps = setOf("Constant","Squeeze","ArgMax","Split",
            "ReduceLogSumExp","AveragePool","TopK","RandomUniform")

        onnxOpNames.forEach { opName ->
            if(onnxOpRegistry.hasMappingOpProcess(opName)) {
                val opDef = onnxOpRegistry.lookupInputFrameworkOpDef(opName)
                println("Processing op name $opName")

                val nodeBuilder = Onnx.NodeProto.newBuilder()
                nodeBuilder.name = opName
                val graphBuilder = Onnx.GraphProto.newBuilder()
                nodeBuilder.opType = opName
                val attrNames = opDef.attributeList.map {attrDef -> attrDef.name }

                //convert to a default case + return graph in new method
                opDef.inputList.forEach { inputArgDef ->
                    //val inputNumberAttr = inputArgDef.numberAttr
                    val numAttributeValue = 1
                    val typeAttrName = "$inputArgDef-types"
                    val typeAttrValue = opDef.attributeList.filter { attributeProto -> attributeProto.name == typeAttrName }
                    for(i in 0 until numAttributeValue) {
                        val listOfFloats = mutableListOf<Float>()
                        val listOfInts = mutableListOf<Int>()
                        val listOfDoubles = mutableListOf<Double>()
                        val listOfBools = mutableListOf<Boolean>()
                        val listOfLongs = mutableListOf<Long>()
                        val listOfStrings = mutableListOf<String>()
                        //the largest tensors we're likely to touch are 5d
                        for(i in 0 until (1 * 2 * 3 * 4 * 5 * 6)) {
                            listOfFloats.add(i.toFloat())
                            listOfInts.add(i)
                            listOfDoubles.add(i.toDouble())
                            listOfBools.add(true)
                            listOfLongs.add(i.toLong())
                            listOfStrings.add("$i")
                        }

                        val nodeName = if(i <= 0) inputArgDef else inputArgDef + "$i"
                        nodeBuilder.addInput(nodeName)

                        when(typeAttrValue[0].stringsList[0].toStringUtf8()) {
                            "double" -> {
                                val onnxTensorProto = Onnx.TensorProto.newBuilder()
                                onnxTensorProto.name = nodeName
                                onnxTensorProto.dataType = Onnx.TensorProto.DataType.DOUBLE_VALUE
                                onnxTensorProto.addAllDoubleData(listOfDoubles)
                                onnxTensorProto.addAllDims(listOf(1,2,3,4,5,6))
                                graphBuilder.addInitializer(onnxTensorProto.build())
                                val onnxNodeToAdd = Onnx.NodeProto.newBuilder()
                                onnxNodeToAdd.name = nodeName
                                onnxNodeToAdd.opType = "Constant"
                                val attrValue = Onnx.AttributeProto.newBuilder()
                                attrValue.name = "value"
                                attrValue.addTensors(onnxTensorProto.build())
                                onnxNodeToAdd.addAttribute(attrValue.build())
                                graphBuilder.addNode(onnxNodeToAdd)
                            }

                            "bool" -> {
                                val onnxTensorProto = Onnx.TensorProto.newBuilder()
                                onnxTensorProto.name = nodeName
                                onnxTensorProto.dataType = Onnx.TensorProto.DataType.BOOL_VALUE
                                onnxTensorProto.addAllInt32Data(listOfInts)
                                onnxTensorProto.addAllDims(listOf(1,2,3,4,5,6))
                                graphBuilder.addInitializer(onnxTensorProto.build())

                                val onnxNodeToAdd = Onnx.NodeProto.newBuilder()
                                onnxNodeToAdd.name = nodeName
                                onnxNodeToAdd.opType = "Constant"
                                val attrValue = Onnx.AttributeProto.newBuilder()
                                attrValue.name = "value"
                                attrValue.addTensors(onnxTensorProto.build())
                                onnxNodeToAdd.addAttribute(attrValue.build())
                                graphBuilder.addNode(onnxNodeToAdd)
                            }

                            "float" -> {
                                val onnxTensorProto = Onnx.TensorProto.newBuilder()
                                onnxTensorProto.name = nodeName
                                onnxTensorProto.dataType = Onnx.TensorProto.DataType.FLOAT_VALUE
                                onnxTensorProto.addAllFloatData(listOfFloats)
                                onnxTensorProto.addAllDims(listOf(1,2,3,4,5,6))
                                graphBuilder.addInitializer(onnxTensorProto.build())

                                val onnxNodeToAdd = Onnx.NodeProto.newBuilder()
                                onnxNodeToAdd.name = nodeName
                                onnxNodeToAdd.opType = "Constant"
                                val attrValue = Onnx.AttributeProto.newBuilder()
                                attrValue.name = "value"
                                attrValue.addTensors(onnxTensorProto.build())
                                onnxNodeToAdd.addAttribute(attrValue.build())
                                graphBuilder.addNode(onnxNodeToAdd)
                            }


                            "int16","uint16" -> {
                                val onnxTensorProto = Onnx.TensorProto.newBuilder()
                                onnxTensorProto.name = nodeName
                                onnxTensorProto.dataType = Onnx.TensorProto.DataType.INT16_VALUE
                                onnxTensorProto.addAllInt32Data(listOfInts)
                                onnxTensorProto.addAllDims(listOf(1,2,3,4,5,6))
                                graphBuilder.addInitializer(onnxTensorProto.build())
                                val onnxNodeToAdd = Onnx.NodeProto.newBuilder()
                                onnxNodeToAdd.name = nodeName
                                onnxNodeToAdd.opType = "Constant"
                                val attrValue = Onnx.AttributeProto.newBuilder()
                                attrValue.name = "value"
                                attrValue.addTensors(onnxTensorProto.build())
                                onnxNodeToAdd.addAttribute(attrValue.build())
                                graphBuilder.addNode(onnxNodeToAdd)
                            }

                            "int32","uint32" -> {
                                val onnxTensorProto = Onnx.TensorProto.newBuilder()
                                onnxTensorProto.name = nodeName
                                onnxTensorProto.dataType = Onnx.TensorProto.DataType.INT32_VALUE
                                onnxTensorProto.addAllDims(listOf(1,2,3,4,5,6))
                                onnxTensorProto.addAllInt32Data(listOfInts)
                                graphBuilder.addInitializer(onnxTensorProto.build())
                                val onnxNodeToAdd = Onnx.NodeProto.newBuilder()
                                onnxNodeToAdd.name = nodeName
                                onnxNodeToAdd.opType = "Constant"
                                val attrValue = Onnx.AttributeProto.newBuilder()
                                attrValue.name = "value"
                                attrValue.addTensors(onnxTensorProto.build())
                                onnxNodeToAdd.addAttribute(attrValue.build())
                                graphBuilder.addNode(onnxNodeToAdd)
                            }

                            "int64","uint64" -> {
                                val onnxTensorProto = Onnx.TensorProto.newBuilder()
                                onnxTensorProto.name = nodeName
                                onnxTensorProto.addAllDims(listOf(1,2,3,4,5,6))
                                onnxTensorProto.dataType = Onnx.TensorProto.DataType.INT64_VALUE
                                onnxTensorProto.addAllInt64Data(listOfLongs)
                                graphBuilder.addInitializer(onnxTensorProto.build())
                                val onnxNodeToAdd = Onnx.NodeProto.newBuilder()
                                onnxNodeToAdd.name = nodeName
                                onnxNodeToAdd.opType = "Constant"
                                val attrValue = Onnx.AttributeProto.newBuilder()
                                attrValue.name = "value"
                                attrValue.addTensors(onnxTensorProto.build())
                                onnxNodeToAdd.addAttribute(attrValue.build())
                                graphBuilder.addNode(onnxNodeToAdd)
                            }

                            "string" -> {
                                val onnxTensorProto = Onnx.TensorProto.newBuilder()
                                onnxTensorProto.name = nodeName
                                onnxTensorProto.dataType = Onnx.TensorProto.DataType.STRING_VALUE
                                onnxTensorProto.addAllDims(listOf(1,2,3,4,5,6))
                                onnxTensorProto.addAllStringData(listOfStrings.map { input -> ByteString.copyFrom(input.toByteArray(
                                    Charset.defaultCharset())) })
                                graphBuilder.addInitializer(onnxTensorProto.build())
                                val onnxNodeToAdd = Onnx.NodeProto.newBuilder()
                                onnxNodeToAdd.name = nodeName
                                onnxNodeToAdd.opType = "Constant"
                                val attrValue = Onnx.AttributeProto.newBuilder()
                                attrValue.name = "value"
                                attrValue.addTensors(onnxTensorProto.build())
                                onnxNodeToAdd.addAttribute(attrValue.build())
                                graphBuilder.addNode(onnxNodeToAdd)
                            }
                        }
                    }

                }


                opDef.attributeList.forEach { attr ->
                    when(attr.type) {
                        Onnx.AttributeProto.AttributeType.INTS -> {
                            //replace empty value with default ints for convolutions
                            val attrBuilder = Onnx.AttributeProto.newBuilder()
                            attrBuilder.addAllInts(listOf(1,1,1,1))
                            attrBuilder.name = attr.name
                            nodeBuilder.addAttribute(attrBuilder.build())
                        }

                        Onnx.AttributeProto.AttributeType.FLOATS -> {
                            //replace empty value with default ints for convolutions
                            val attrBuilder = Onnx.AttributeProto.newBuilder()
                            attrBuilder.addAllFloats(listOf(1.0f,1.0f,1.0f,1.0f))
                            attrBuilder.name = attr.name
                            nodeBuilder.addAttribute(attrBuilder.build())
                        }


                        Onnx.AttributeProto.AttributeType.STRINGS -> {
                            //replace empty value with default ints for convolutions
                            val attrBuilder = Onnx.AttributeProto.newBuilder()
                            if(opName != "LSTM")
                                attrBuilder.addAllStrings(listOf("1","2","3","4").map { input -> ByteString.copyFrom(input.toByteArray(
                                    Charset.defaultCharset()))
                                })
                            else {
                                attrBuilder.addAllStrings(listOf("Relu","Tanh","Sigmoid","Relu").map { input -> ByteString.copyFrom(input.toByteArray(
                                    Charset.defaultCharset()))
                                })
                            }
                            attrBuilder.name = attr.name
                            nodeBuilder.addAttribute(attrBuilder.build())
                        }

                        Onnx.AttributeProto.AttributeType.TENSOR -> {
                            val attrBuilder = Onnx.AttributeProto.newBuilder()
                            attrBuilder.t = Onnx.TensorProto.newBuilder()
                                .addAllDims(listOf(1,1)).setDataType(Onnx.TensorProto.DataType.DOUBLE_VALUE)
                                .addAllDoubleData(listOf(1.0))
                                .build()
                            attrBuilder.name = attr.name
                            nodeBuilder.addAttribute(attrBuilder.build())
                        }



                        else -> {
                            nodeBuilder.addAttribute(attr)
                        }
                    }

                }


                graphBuilder.addNode(nodeBuilder.build())
                val graph = graphBuilder.build()




                if(!bannedOps.contains(opName)) {
                    val mappingProcess = onnxOpRegistry.lookupOpMappingProcess(opName)
                    val irGraph = OnnxIRGraph(graphDef = graph)
                    val mappingContext = OnnxMappingContext(opDef = opDef,node = nodeBuilder.build(),graph = irGraph)
                    val mapResult = mappingProcess.applyProcess(mappingContext)
                    val groupedByArgType = mapResult.second.argDescriptorList.groupBy { keySelector -> keySelector.argType }
                    val sortedGroups = HashMap<OpNamespace.ArgDescriptor.ArgType,List<OpNamespace.ArgDescriptor>>()
                    groupedByArgType.forEach { (argType, argDescriptors) ->
                        sortedGroups[argType] = argDescriptors.sortedBy { argDescriptor -> argDescriptor.argIndex }
                    }

                    //NOTE: Bitcast is in this list for examination outside of list offsets for assertions. We don't currently support data types for the test nodes.
                    sortedGroups.values.forEach { list ->   run {
                        val namesEncountered = HashSet<String>()
                        list.forEachIndexed { index, argDescriptor ->
                            //don't validate a name encountered more than once, this is probably an array
                            //note that we skip some ops here due to this assumption breaking for list types, we will test list types separately
                            if(!namesEncountered.contains(argDescriptor.name)
                                && !bannedOps.contains(opName)) {
                                assertEquals(
                                    "Arg index $index for arg descriptor name ${argDescriptor.name} for nd4j op ${mappingContext.nd4jOpName()} when arg index was actually ${argDescriptor.argIndex}. Full arg descriptor was ${argDescriptor}.",
                                    argDescriptor.argIndex, index
                                )
                                namesEncountered.add(argDescriptor.name)
                            }
                        }
                    }
                        //SameDiff.importFrozenTF(irGraph.graphDef)
                        val sameDiffResult = importGraph(irGraph = irGraph,importOverride = null,opFilter = null)
                        println("Processed op name $opName")

                    }
                }


            }
        }
    }



    @Test
    fun testOpsMapped() {
        val onnxOpNames = onnxOpRegistry.inputFrameworkOpNames().filter { onnxOpRegistry.registeredOps.containsKey(it) }
        val nd4jOpNames = onnxOpRegistry.nd4jOpNames()
        /**
         * TODO: Assert each op is mapped.
         *
         * Assert all attributes in nd4j are mapped.
         * If not, let's document what isn't and why for each op.
         *
         * Create an op generation tool that allows random generation of test cases
         * based on existing mapped ops between nd4j and tensorflow.
         */
        onnxOpNames.map { onnxOpName -> onnxOpRegistry.lookupOpMappingProcess(onnxOpName)}
            .forEach {
                val onnxNamesMapped = HashSet<String>()
                val nd4jNamesMapped = HashSet<String>()
                //we can ignore dtype for now
                nd4jNamesMapped.add("dtype")
                val opDef = onnxOpRegistry.lookupNd4jOpDef(it.opName())
                val onnxOpDef = onnxOpRegistry.lookupInputFrameworkOpDef(it.inputFrameworkOpName())
                val onnxAssertionNames = HashSet<String>()
                onnxAssertionNames.addAll(onnxOpDef.inputList.map { arg -> arg.toString() })
                onnxAssertionNames.addAll(onnxOpDef.attributeList.map { attr -> attr.name })
                val nd4jOpDefAssertions = HashSet<String>()
                nd4jOpDefAssertions.addAll(opDef.argDescriptorList.map { argDescriptor -> argDescriptor.name })
                val numRequiredInputs = onnxOpDef.inputCount
                val nd4jInputs = opDef.argDescriptorList.filter { arg -> arg.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }.count()
                /**
                 * TODO: Grab total collection of mapped nd4j names
                 * as outputs and mapped tensorflow names as inputs.
                 * Compare the mapped names to the op definitions
                 * in nd4j and tensorflow respectively.
                 */
                it.tensorMappingRules().forEach { mappingRule ->
                    mappingRule.mappingNamesToPerform().forEach {  mappingName ->
                        onnxNamesMapped.add(mappingName.value)
                        nd4jNamesMapped.add(mappingName.key)
                    }
                }

                it.attributeMappingRules().forEach { mappingRule ->
                    mappingRule.mappingNamesToPerform().forEach { mappingName ->
                        onnxNamesMapped.add(mappingName.value)
                        nd4jNamesMapped.add(mappingName.key)
                    }

                    mappingRule.mappingTransformerArgs().forEach {transformerArg ->
                        run {
                            transformerArg.value.forEach { argValue ->
                                nd4jNamesMapped.add(argValue.name)

                            }
                        }
                    }

                }


                onnxOpDef.inputList.forEach { inputName ->
                    Assert.assertTrue(onnxAssertionNames.contains(inputName))
                }

                onnxOpDef.attributeList.map { attrDef -> attrDef.name }.forEach { attrName ->
                    Assert.assertTrue(onnxAssertionNames.contains(attrName))
                }



                opDef.argDescriptorList.forEach {  argDef ->
                    //only require it when the

                    when(argDef.argType) {
                        OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR -> {
                            /**
                             * Nd4j typically has many optional inputs that can also double as attributes
                             * We need to allow for a bit of flexibility in how we handle op definitions. If they're not mapped 1 to 1,
                             * we just log a warning for unmapped inputs. Otherwise we can do an assertion.
                             */
                            if(numRequiredInputs == nd4jInputs)
                                assertTrue("Nd4j op name ${opDef.name} with onnx mapping ${onnxOpDef.name} has missing mapping ${argDef.name}", nd4jNamesMapped.contains(argDef.name))
                            else if(!nd4jNamesMapped.contains(argDef.name)) {
                                println("Warning: Nd4j op name ${opDef.name} with onnx mapping ${onnxOpDef.name} has missing mapping ${argDef.name}")
                            }
                        }
                        OpNamespace.ArgDescriptor.ArgType.INT32,OpNamespace.ArgDescriptor.ArgType.INT64 -> {
                            assertTrue("Nd4j op name ${opDef.name} with onnx mapping ${onnxOpDef.name}  has missing mapping ${argDef.name}", nd4jNamesMapped.contains(argDef.name))
                        }
                        OpNamespace.ArgDescriptor.ArgType.DOUBLE, OpNamespace.ArgDescriptor.ArgType.FLOAT -> {
                            assertTrue("Nd4j op name ${opDef.name} with onnx mapping ${onnxOpDef.name}  has missing mapping ${argDef.name}", nd4jNamesMapped.contains(argDef.name))
                        }
                        OpNamespace.ArgDescriptor.ArgType.BOOL -> {
                            assertTrue("Nd4j op name ${opDef.name} with onnx mapping ${onnxOpDef.name}  has missing mapping ${argDef.name}", nd4jNamesMapped.contains(argDef.name))
                        }
                    }

                }

            }
    }

    @Test
    fun testOpExecution() {

    }


}
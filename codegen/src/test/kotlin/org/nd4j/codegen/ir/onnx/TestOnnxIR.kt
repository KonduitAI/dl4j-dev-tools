package org.nd4j.codegen.ir.onnx

import junit.framework.Assert
import org.junit.jupiter.api.Test
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.ir.OpNamespace
import kotlin.test.assertTrue

class TestOnnxIR {
    val declarations = OnnxOpDeclarations

    @Test
    fun testAbs() {
        val op = onnxops.first { it.name == "Abs" }
        val nodeProto = NodeProto {
            opType = "Abs"
            name = "input"
            Input("X")
            Input("Y")
        }

        val graphProto = GraphProto {
            Node(nodeProto)
        }

        val onnxIRGraph = OnnxIRGraph(graphDef = graphProto)
        val mappingProcess = OpRegistryHolder.onnx().lookupOpMappingProcess("Abs")
        val onnxMappingContext = OnnxMappingContext(opDef = op,node = nodeProto,graph = onnxIRGraph)
        val appliedProcess = mappingProcess.applyProcess(mappingCtx = onnxMappingContext)
        println(appliedProcess)
    }

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
                                    Assert.assertTrue("Nd4j op name ${opDef.name} with onnx mapping ${onnxOpDef.name} has missing mapping ${argDef.name}", nd4jNamesMapped.contains(argDef.name))
                                else if(!nd4jNamesMapped.contains(argDef.name)) {
                                    println("Warning: Nd4j op name ${opDef.name} with onnx mapping ${onnxOpDef.name} has missing mapping ${argDef.name}")
                                }
                            }
                            OpNamespace.ArgDescriptor.ArgType.INT32,OpNamespace.ArgDescriptor.ArgType.INT64 -> {
                                Assert.assertTrue("Nd4j op name ${opDef.name} with onnx mapping ${onnxOpDef.name}  has missing mapping ${argDef.name}", nd4jNamesMapped.contains(argDef.name))
                            }
                            OpNamespace.ArgDescriptor.ArgType.DOUBLE, OpNamespace.ArgDescriptor.ArgType.FLOAT -> {
                                Assert.assertTrue("Nd4j op name ${opDef.name} with onnx mapping ${onnxOpDef.name}  has missing mapping ${argDef.name}", nd4jNamesMapped.contains(argDef.name))
                            }
                            OpNamespace.ArgDescriptor.ArgType.BOOL -> {
                                Assert.assertTrue("Nd4j op name ${opDef.name} with onnx mapping ${onnxOpDef.name}  has missing mapping ${argDef.name}", nd4jNamesMapped.contains(argDef.name))
                            }
                        }

                    }

                }
    }

}
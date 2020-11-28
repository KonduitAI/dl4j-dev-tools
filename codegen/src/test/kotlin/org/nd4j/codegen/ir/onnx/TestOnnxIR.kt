package org.nd4j.codegen.ir.onnx

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
}
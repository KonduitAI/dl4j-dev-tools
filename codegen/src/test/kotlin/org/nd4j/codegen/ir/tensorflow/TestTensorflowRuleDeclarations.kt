package org.nd4j.codegen.ir.tensorflow

import org.junit.jupiter.api.Test
import org.nd4j.codegen.ir.nd4jOpDescriptors

class TestTensorflowRuleDeclarations {
    @Test
    fun testConditionalIndex() {
        val conditionalIndex = conditionalFieldValueIntIndexArrayRule(outputAttribute = "type",inputFrameworkAttributeName = "N",targetValue = "test1",
                trueIndex = 0,falseIndex = 0)
        val opDef = tensorflowOps.findOp("AddN")
        val nodeDef = NodeDef {
            op = "AddN"
            Input("inputs")
            Input("y")
            name = "test"
            Attribute(name = "N",value = AttrValue {
                name = "N"
                i = 1
            })
        }

        val graphDef = GraphDef {
            Node(nodeDef)
        }

        val tensorflowNode = TensorflowIRNode(nodeDef, opDef)
        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps)


        val mappingContext = TensorflowMappingContext(opDef = opDef,node = nodeDef,graph = tfGraph)
        conditionalIndex.convertAttributes(mappingContext)
    }
}




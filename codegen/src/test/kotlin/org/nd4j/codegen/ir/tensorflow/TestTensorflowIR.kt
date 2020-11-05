package org.nd4j.codegen.ir.tensorflow

import org.apache.commons.io.IOUtils
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.common.io.ClassPathResource
import org.nd4j.shade.protobuf.ByteString
import org.tensorflow.framework.*
import java.nio.charset.Charset

class TestTensorflowIR {
    val declarations = TensorflowOpDeclarations

    @Test
    fun testTensorflowAbs() {
        val opDef = tensorflowOps.findOp("Abs")
        val nodeDef = NodeDef {
            op = "Abs"
            Input("x")
            Input("y")
            name = "test"
        }

        val graphDef = GraphDef {
            Node(nodeDef)
        }

        val tensorflowNode = TensorflowIRNode(nodeDef, opDef)
        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps)
        val absMappingProcess = OpRegistryHolder.tensorflow().lookupOpMappingProcess(inputFrameworkOpName = "Abs")

        val mappingContext = TensorflowMappingContext(opDef = opDef,node = nodeDef,graph = tfGraph)
        val input = absMappingProcess.applyProcess(mappingContext)
        println(input)

    }


    @Test
    fun loadModelTest() {
        val inputs = listOf("input_0", "input_1")
        val content = IOUtils.toByteArray(ClassPathResource("lenet_frozen.pb").inputStream)
        val graphDef = GraphDef.parseFrom(content)
        println(graphDef)
    }

    @Test
    fun runTfImportProcess() {
        val importProcess = TensorflowImportProcess()
        val opDef = tensorflowOps.findOp("Abs")
        val nodeDef = NodeDef {
            op = "Abs"
            Input("x")
            Input("y")
            name = "test"
        }

        val x = NodeDef {
            op = "Const"
            name = "x"
            Attribute(name = "value",value = AttrValue {
                tensor = TensorProto {
                    name = "x"
                    FloatData(listOf(1f))
                    shape = TensorShapeProto {
                        Dim(name = "0",size = 1)
                        Dim(name = "1", size = 1)
                    }

                    DataType(DataType.DT_FLOAT)

                }
            })
        }

        val graphDef = GraphDef {
            Node(nodeDef)
            Node(x)
        }

        val tensorflowNode = TensorflowIRNode(nodeDef, opDef)
        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps)
        val mappingProcesses = importProcess.createMappingProcesses(graph = tfGraph).map {
            process -> TensorflowImportContext(process = process,mappingContext = importProcess.createMappingContext(graph = tfGraph,node = nodeDef))
        }

        val samediff = importProcess.runImportProcess(mappingProcesses)

        println(samediff)

    }

    @Test
    fun testRegistry() {
        val registry = OpRegistryHolder.tensorflow()
        val mappingProcess = registry.lookupOpMappingProcess("Conv2D")
        println(mappingProcess)
    }

    @Test
    fun testTensorflowConv2d() {
        val opDef = tensorflowOps.findOp("Conv2D")
        val attrValue = AttrValue {
            list = ListAttrValue(1,1,1,1)
        }

        val dilationsAttr = AttrValue {
            list = ListAttrValue(1,1,1,1)
        }

        val dataFormatValue = AttrValue {
            s = ByteString.copyFrom("NCHW", Charset.defaultCharset())
        }

        val paddingValue = AttrValue {
            s = ByteString.copyFrom("SAME", Charset.defaultCharset())
        }

        val nodeDef = NodeDef {
            Input("input")
            Input("filter")
            op = "Conv2D"
            name = "input"
            Attribute("strides",attrValue)
            Attribute("data_format",dataFormatValue)
            Attribute("padding",paddingValue)
            Attribute("dilations",dilationsAttr)
        }

        val tensorValue2 = TensorProto {
            tensorShape = TensorShapeProto {
                Dim(name = "0", size = 1)
                Dim(name = "1", size = 5)
                Dim(name = "2", size = 5)
                Dim(name = "3", size = 6)
            }
        }

        val weightsNode = NodeDef {
            op = "Const"
            name = "filter"
            Attribute("value", AttrValue {
                tensor = tensorValue2
            })
        }

        val graphDef = GraphDef {
            Node(nodeDef)
            Node(weightsNode)
        }


        val tensorflowIRNode = TensorflowIRNode(nodeDef, opDef)
        val conv2dMappingProcess = OpRegistryHolder.lookupOpMappingProcess<NodeDef,OpDef,TensorProto,DataType, OpDef.AttrDef,AttrValue>(inputFrameworkName = "tensorflow",inputFrameworkOpName = "Conv2D")

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = nodeDef,graph = tfGraph)
        """
            node {
              name: "Lenet/conv1_1/Conv2D"
              op: "Conv2D"
              input: "Reshape"
              input: "Lenet/conv1/weights"
              attr {
                key: "use_cudnn_on_gpu"
                value {
                  b: true
                }
              }
              attr {
                key: "padding"
                value {
                  s: "SAME"
                }
              }
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
              attr {
                key: "strides"
                value {
                  list {
                    i: 1
                    i: 1
                    i: 1
                    i: 1
                  }
                }
              }
              attr {
                key: "data_format"
                value {
                  s: "NHWC"
                }
              }
            }
        """.trimIndent()

        """
            node {
              name: "Lenet/conv1/weights"
              op: "Const"
              attr {
                key: "value"
                value {
                  tensor {
                    dtype: DT_FLOAT
                    tensor_shape {
                      dim {
                        size: 5
                      }
                      dim {
                        size: 5
                      }
                      dim {
                        size: 1
                      }
                      dim {
                        size: 6
                      }
                    }
                    tensor_content: "\265`\023=\207\274\277<\345j\234<\2515\266<\217\001Y<\375\223\346<k\3273<\"\202\367<\341\201\303<\276\262\343<\232!\220<\210\301\343<\215\363G\273\3131\265<\344\324\341<\351\251\315<\000]X<\200\251\223<_\334\022<=\344\311:W\"m<\224\300\256<\022\274\177;\323\005%<R\2518<\301\343\264<-\366\023<0`\230<\310\223\022=\376\374\300<b\343\305\272\003\355v<\2220\273<\310R\231;uv\241<\326\350\261<\326KL<P\320J<^\257\370;@\005\323<\246\237\226<&\370m<\231V\237<S)\r<Z5\016=\033\364\020<\200X\016=\341Ji<\251\232\204\273\254\303\255<L\2121\270\3057\003=W\034\327<W\326n<a\v\317;\302r\270<_\353R<C/\303<\333&{<\275]\315<\267\270\310<\310\270\245<\313\260,<bH\004=\027\215\244<\322\301\207<\351\242\246<${'$'}\247_\273\025\327n<O\205\243<\222RI<&m\001<D\360\263<\333\316\214<\226\035\232<?\277*<\256E\356<!\203x<\343\034J<\032#\330;ip\257<\017\215\226<=\273\a=g_\201<\363\261z<\005\2620<\210q/<mw~<\352\245\347;\374!,\272\006\235^<l\314\241<\252P\304<{\267\326<\244\023><\375\236Q<\213\016D<\223s\376\272\313\2561<\352\374\303<\357\036{<\r1\272<\271\020\252<u\027\236\274\"\301\334<\362\274Q<\vej<d\022\314<\207\302\226<TO\206;\373\327\304;\312\373\3139%\355\325<_m\313<\002y\257<\251>\265;\232=\b<{a\236;\242j\243<\212\353\330<\227J\023=\026\210\252\271\b\035\263<\264;N<\372\215\227;\351g\226<p=\231<l\310\253<\323\201\316<\377\207\222<\332\262D<\024\264\022=K\374\245<\023\361?\273\223\243\254<*Az<\3654\377<<\205m;^6R<]\274\326\274\240\263h;@\305\313;|V\343;N\333\346<\225\270\211<\016\230\355\273\021\2415;8\265\215\274@o\246;\243\205\260<"
                  }
                }
              }
              attr {
                key: "dtype"
                value {
                  type: DT_FLOAT
                }
              }
            } 
        """.trimIndent()

        val content = IOUtils.toByteArray(ClassPathResource("lenet_frozen.pb").inputStream)
        val graphDef2 = GraphDef.parseFrom(content)

        val ret = SameDiff.importFrozenTF(ClassPathResource("lenet_frozen.pb").file)
        val processOutput = conv2dMappingProcess.applyProcess(mappingContext)
        println(processOutput)

    }
}
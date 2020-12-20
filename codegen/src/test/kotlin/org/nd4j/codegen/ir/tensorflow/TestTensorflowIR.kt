package org.nd4j.codegen.ir.tensorflow

import junit.framework.Assert.assertEquals
import junit.framework.Assert.assertTrue
import org.apache.commons.io.IOUtils
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.codegen.ir.importGraph
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.common.io.ClassPathResource
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.shade.protobuf.ByteString
import org.tensorflow.framework.*
import java.lang.IllegalArgumentException
import java.lang.IllegalStateException
import java.nio.charset.Charset
import kotlin.math.max

data class GraphInput(val graphDef: GraphDef,val inputNames: List<String>,val outputNames: List<String>,
                      val inputArrays: Map<String,INDArray>)

class TestTensorflowIR {
    val declarations = TensorflowOpDeclarations

    @Test
    fun testTensorflowAbs() {
        val opDef = tensorflowOps.findOp("Abs")
        val nodeDef = NodeDef {
            op = "Abs"
            name = "test"
            Input("x")
        }




        val x = NodeDef {
            op = "Const"
            name = "x"
            Attribute(name = "value",value = AttrValue {
                tensor = TensorProto.getDefaultInstance()
            })
        }


        val graphDef = GraphDef {
            Node(nodeDef)
            Node(x)
        }

        val tensorflowNode = TensorflowIRNode(nodeDef, opDef)
        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps)
        val absMappingProcess = OpRegistryHolder.tensorflow().lookupOpMappingProcess(inputFrameworkOpName = "Abs")

        val mappingContext = TensorflowMappingContext(opDef = opDef,node = nodeDef,graph = tfGraph,dynamicVariables = emptyMap())
        val input = absMappingProcess.applyProcess(mappingContext)
        println(input)

    }


    @Test
    fun loadModelTest() {
        val inputs = listOf("input_0", "input_1")
        val content = IOUtils.toByteArray(ClassPathResource("lenet_frozen.pb").inputStream)
        val graphDef = GraphDef.parseFrom(content)
        val irGraph = TensorflowIRGraph(graphDef, tensorflowOps)
        val importedModel = importGraph(irGraph = irGraph,importOverride = null,opFilter = null)
        println(importedModel)
    }


    @Test
    fun testRegistry() {
        val registry = OpRegistryHolder.tensorflow()
        val mappingProcess = registry.lookupOpMappingProcess("Conv2D")
        println(mappingProcess)
    }


    @Test
    fun testOpOrdering() {
        val tensorflowOpNames = tensorflowOpRegistry.inputFrameworkOpNames()
        val bannedOps = setOf("RandomUniformInt","ResizeArea","UnsortedSegmentProd","UnsortedSegmentMin","SpaceToBatch",
            "ResizeNearestNeighbor","Dilation2D","Bitcast","LinSpace","UnsortedSegmentSum",
            "TensorArrayScatter","OneHot","UnsortedSegmentMax","TopKV2","TopK","Range","HistogramFixedWidth","ClipByValue","ResizeBilinear","Bincount","SplitV")

        tensorflowOpNames.forEach { opName ->
            if(tensorflowOpRegistry.hasMappingOpProcess(opName)) {
                val opDef = tensorflowOps.findOp(opName)
                println("Processing op name $opName")

                val nodeBuilder = NodeDef.newBuilder()
                nodeBuilder.name = opName
                val graphBuilder = GraphDef.newBuilder()
                nodeBuilder.op = opName
                val attrNames = opDef.attrList.map {attrDef -> attrDef.name }

                //convert to a default case + return graph in new method
                opDef.inputArgList.forEach { inputArgDef ->
                    val inputNumberAttr = inputArgDef.numberAttr
                    val numAttributeValue = if(!inputNumberAttr.isEmpty()) max(opDef.attrList.find { attrDef -> attrDef.name == inputNumberAttr }!!.minimum,2) else 1
                    val typeAttrName = if(inputArgDef.typeAttr.isNotEmpty()) inputArgDef.typeAttr else "T"
                    val typeAttrValue = if(inputArgDef.typeAttr.isNotEmpty() && attrNames.contains(inputArgDef.typeAttr))
                        opDef.attrList.first { attrDef -> attrDef.name == inputArgDef.typeAttr }.defaultValue.type else inputArgDef.type
                    for(i in 0 until numAttributeValue) {
                        val listOfFloats = mutableListOf<Float>()
                        val listOfInts = mutableListOf<Int>()
                        val listOfDoubles = mutableListOf<Double>()
                        val listOfBools = mutableListOf<Boolean>()
                        val listOfLongs = mutableListOf<Long>()
                        //the largest tensors we're likely to touch are 5d
                        for(i in 0 until (1 * 2 * 3 * 4 * 5 * 6)) {
                            listOfFloats.add(i.toFloat())
                            listOfInts.add(i)
                            listOfDoubles.add(i.toDouble())
                            listOfBools.add(true)
                            listOfLongs.add(i.toLong())
                        }

                        val nodeName = if(i <= 0) inputArgDef.name else inputArgDef.name + "$i"
                        nodeBuilder.addInput(nodeName)

                        when(typeAttrValue) {
                            DataType.DT_DOUBLE -> {
                                //add placeholders for all parameters
                                val placeHolder = NodeDef {
                                    name = nodeName
                                    op = "Const"
                                    Attribute(name = typeAttrName,value = AttrValue {
                                        type = DataType.DT_DOUBLE
                                    })

                                    Attribute(name = "value",value = AttrValue {
                                        tensor = TensorProto {
                                            Shape(listOf(1,2,3,4,5,6))
                                            DoubleData(listOfDoubles)
                                            DataType(DataType.DT_DOUBLE)
                                        }
                                    })

                                }

                                graphBuilder.addNode(placeHolder)
                            }

                            DataType.DT_FLOAT -> {
                                //add placeholders for all parameters
                                val placeHolder = NodeDef {
                                    name = nodeName
                                    op = "Const"
                                    Attribute(name = typeAttrName,value = AttrValue {
                                        type = DataType.DT_FLOAT
                                    })

                                    Attribute(name = "value",value = AttrValue {
                                        tensor = TensorProto {
                                            Shape(listOf(1,2,3,4,5,6))
                                            FloatData(listOfFloats)
                                            DataType(DataType.DT_FLOAT)
                                        }
                                    })

                                }

                                graphBuilder.addNode(placeHolder)
                            }

                            DataType.DT_BOOL -> {
                                //add placeholders for all parameters
                                val placeHolder = NodeDef {
                                    name = nodeName
                                    op = "Const"
                                    Attribute(name = typeAttrName,value = AttrValue {
                                        type = DataType.DT_BOOL
                                    })

                                    Attribute(name = "value",value = AttrValue {
                                        tensor = TensorProto {
                                            Shape(listOf(1,2,3,4,5,6))
                                            BooleanData(listOfBools)
                                            DataType(DataType.DT_BOOL)
                                        }
                                    })

                                }

                                graphBuilder.addNode(placeHolder)
                            }

                            DataType.DT_INT16 -> {
                                //add placeholders for all parameters
                                val placeHolder = NodeDef {
                                    name = nodeName
                                    op = "Const"
                                    Attribute(name = typeAttrName,value = AttrValue {
                                        type = DataType.DT_INT16
                                    })

                                    Attribute(name = "value",value = AttrValue {
                                        tensor = TensorProto {
                                            Shape(listOf(1,2,3,4,5,6))
                                            Int32Data(listOfInts)
                                            DataType(DataType.DT_INT16)
                                        }
                                    })

                                }

                                graphBuilder.addNode(placeHolder)
                            }

                            DataType.DT_INT32 -> {
                                //add placeholders for all parameters
                                val placeHolder = NodeDef {
                                    name = nodeName
                                    op = "Const"
                                    Attribute(name = typeAttrName,value = AttrValue {
                                        type = DataType.DT_INT32
                                    })

                                    Attribute(name = "value",value = AttrValue {
                                        tensor = TensorProto {
                                            Shape(listOf(1,2,3,4,5,6))
                                            Int32Data(listOfInts)
                                            DataType(DataType.DT_INT32)
                                        }
                                    })

                                }

                                graphBuilder.addNode(placeHolder)
                            }

                            DataType.DT_INT64 -> {
                                //add placeholders for all parameters
                                val placeHolder = NodeDef {
                                    name = nodeName
                                    op = "Const"
                                    Attribute(name = typeAttrName,value = AttrValue {
                                        type = DataType.DT_INT64
                                    })

                                    Attribute(name = "value",value = AttrValue {
                                        tensor = TensorProto {
                                            Shape(listOf(1,2,3,4,5,6))
                                            Int64Data(listOfLongs)
                                            DataType(DataType.DT_INT64)
                                        }
                                    })

                                }

                                graphBuilder.addNode(placeHolder)
                            }

                            DataType.DT_STRING -> {
                                //add placeholders for all parameters
                                val placeHolder = NodeDef {
                                    name = nodeName
                                    op = "Const"
                                    Attribute(name = typeAttrName,value = AttrValue {
                                        type = DataType.DT_DOUBLE
                                    })

                                    Attribute(name = "value",value = AttrValue {
                                        tensor = TensorProto {
                                            Shape(listOf(1,2,3,4,5,6))
                                            FloatData(listOfFloats)
                                            DataType(DataType.DT_DOUBLE)
                                        }
                                    })

                                }

                                graphBuilder.addNode(placeHolder)
                            }
                            else -> {

                                //add placeholders for all parameters
                                val placeHolder = NodeDef {
                                    name = nodeName
                                    op = "Const"
                                    Attribute(name = typeAttrName,value = AttrValue {
                                        type = DataType.DT_DOUBLE
                                    })

                                    Attribute(name = "value",value = AttrValue {
                                        tensor = TensorProto {
                                            Shape(listOf(1,2,3,4,5,6))
                                            DoubleData(listOfDoubles)
                                            DataType(DataType.DT_DOUBLE)
                                        }
                                    })

                                }

                                graphBuilder.addNode(placeHolder)

                            }


                        }
                    }

                }


                opDef.attrList.forEach { attr ->
                    if(attr.hasMinimum or attr.type.contains("list")) {
                        //it varies whether lists have minimums or not (some should)
                        //defaulting to size 4 or the minimum will hit most use cases
                        val listSize = max(attr.minimum,5)
                        when(attr.type) {
                            "list(int)" -> {
                                val attrList = ArrayList<Long>()
                                for(i in 0 until listSize) {
                                    attrList.add(i)
                                }

                                nodeBuilder.putAttr(attr.name, AttrValue {
                                    ListInts(attrList)
                                })
                            }
                            "list(float)" -> {
                                val attrList = ArrayList<Float>()
                                for(i in 0 until listSize) {
                                    attrList.add(i.toFloat())
                                }
                                nodeBuilder.putAttr(attr.name, AttrValue {
                                    ListFloats(attrList)
                                })
                            }
                            else -> {
                                if(attr.hasMinimum) {
                                    when(attr.type) {
                                        "float" -> {
                                            nodeBuilder.putAttr(attr.name, org.nd4j.codegen.ir.tensorflow.AttrValue {
                                                f = attr.minimum.toFloat()
                                            })

                                        }
                                        "int" -> {
                                            nodeBuilder.putAttr(attr.name, org.nd4j.codegen.ir.tensorflow.AttrValue {
                                                i = attr.minimum.toLong()
                                            })
                                        }
                                    }
                                }
                                else
                                    nodeBuilder.putAttr(attr.name,attr.defaultValue)

                            }
                        }
                    }
                    else
                        nodeBuilder.putAttr(attr.name,attr.defaultValue)
                }


                graphBuilder.addNode(nodeBuilder.build())
                val graph = graphBuilder.build()




                if(!bannedOps.contains(opName)) {
                    val mappingProcess = tensorflowOpRegistry.lookupOpMappingProcess(opName)
                    val irGraph = TensorflowIRGraph(graphDef = graph,opDef = tensorflowOps)
                    val mappingContext = TensorflowMappingContext(opDef = opDef,node = nodeBuilder.build(),graph = irGraph,dynamicVariables = emptyMap())
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
                            if(!namesEncountered.contains(argDescriptor.name) && opName != "BatchToSpace" && !opName.contains("NonMaxSuppression")
                                && !bannedOps.contains(opName)) {
                                assertEquals("Arg index $index for arg descriptor name ${argDescriptor.name} for nd4j op ${mappingContext.nd4jOpName()} when arg index was actually ${argDescriptor.argIndex}. Full arg descriptor was ${argDescriptor}. Graph was ${graph}",
                                    argDescriptor.argIndex, index)
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
    fun testTensorflowConv2dOld() {
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
        val conv2dMappingProcess = OpRegistryHolder.lookupOpMappingProcess<GraphDef,NodeDef,OpDef,TensorProto,DataType, OpDef.AttrDef,AttrValue>(inputFrameworkName = "tensorflow",inputFrameworkOpName = "Conv2D")

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps)
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
        val irGraph = TensorflowIRGraph(graphDef = graphDef2,opDef = tensorflowOps)
        //val importedGraph = org.nd4j.codegen.ir.importGraph(irGraph = irGraph,importOverride =  null,opFilter =  null)
        val ret = SameDiff.importFrozenTF(ClassPathResource("lenet_frozen.pb").file)
        //val processOutput = conv2dMappingProcess.applyProcess(mappingContext)
        println(ret)

    }

    @Test
    fun testTensorflowMappingContext() {
        val absOpDef = tensorflowOpRegistry.lookupOpMappingProcess("Abs")
        val opDef = tensorflowOps.findOp("Abs")
        val absNodeDef = NodeDef {
            name = "input"
            Input("input1")
            op = "Abs"
        }

        val graph = GraphDef {
            Node(absNodeDef)
        }

        val tfIRGraph = TensorflowIRGraph(graphDef = graph,opDef = tensorflowOps)

        val tfMappingCtx = TensorflowMappingContext(
            opDef =opDef,
            node = absNodeDef,
            graph = tfIRGraph,dynamicVariables = emptyMap())

        assertEquals(opDef,tfMappingCtx.opDef)

    }

    @Test
    fun testOpsMapped() {
        val tensorflowOpNames = tensorflowOpRegistry.inputFrameworkOpNames().filter { tensorflowOpRegistry.registeredOps.containsKey(it) }
        val nd4jOpNames = tensorflowOpRegistry.nd4jOpNames()

        tensorflowOpNames.map {tensorflowOpName -> tensorflowOpRegistry.lookupOpMappingProcess(tensorflowOpName)}
            .forEach {
                val tensorflowNamesMapped = HashSet<String>()
                val nd4jNamesMapped = HashSet<String>()
                //we can ignore dtype for now
                nd4jNamesMapped.add("dtype")
                val opDef = tensorflowOpRegistry.lookupNd4jOpDef(it.opName())
                val tensorflowOpDef = tensorflowOpRegistry.lookupInputFrameworkOpDef(it.inputFrameworkOpName())
                val tensorflowAssertionNames = HashSet<String>()
                tensorflowAssertionNames.addAll(tensorflowOpDef.inputArgList.map { arg -> arg.name })
                tensorflowAssertionNames.addAll(tensorflowOpDef.attrList.map { attr -> attr.name })
                val nd4jOpDefAssertions = HashSet<String>()
                nd4jOpDefAssertions.addAll(opDef.argDescriptorList.map { argDescriptor -> argDescriptor.name })
                val numRequiredInputsTf = tensorflowOpDef.inputArgCount
                val nd4jInputs = opDef.argDescriptorList.filter { arg -> arg.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }.count()
                /**
                 * TODO: Grab total collection of mapped nd4j names
                 * as outputs and mapped tensorflow names as inputs.
                 * Compare the mapped names to the op definitions
                 * in nd4j and tensorflow respectively.
                 */
                it.tensorMappingRules().forEach { mappingRule ->
                    mappingRule.mappingNamesToPerform().forEach {  mappingName ->
                        tensorflowNamesMapped.add(mappingName.value)
                        nd4jNamesMapped.add(mappingName.key)
                    }
                }

                it.attributeMappingRules().forEach { mappingRule ->
                    mappingRule.mappingNamesToPerform().forEach { mappingName ->
                        tensorflowNamesMapped.add(mappingName.value)
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


                tensorflowOpDef.inputArgList.map {input -> input.name}.forEach { inputName ->
                    assertTrue(tensorflowAssertionNames.contains(inputName))
                }

                tensorflowOpDef.attrList.filter { attrDef -> attrDef.type != "type" }.map {attrDef -> attrDef.name }.forEach { attrName ->
                    assertTrue(tensorflowAssertionNames.contains(attrName))
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
                            if(numRequiredInputsTf == nd4jInputs)
                                assertTrue("Nd4j op name ${opDef.name} with tensorflow mapping ${tensorflowOpDef.name} has missing mapping ${argDef.name}",nd4jNamesMapped.contains(argDef.name))
                            else if(!nd4jNamesMapped.contains(argDef.name)) {
                                println("Warning: Nd4j op name ${opDef.name} with tensorflow mapping ${tensorflowOpDef.name} has missing mapping ${argDef.name}")
                            }
                        }
                        OpNamespace.ArgDescriptor.ArgType.INT32,OpNamespace.ArgDescriptor.ArgType.INT64 -> {
                            assertTrue("Nd4j op name ${opDef.name} with tensorflow mapping ${tensorflowOpDef.name}  has missing mapping ${argDef.name}",nd4jNamesMapped.contains(argDef.name))
                        }
                        OpNamespace.ArgDescriptor.ArgType.DOUBLE, OpNamespace.ArgDescriptor.ArgType.FLOAT -> {
                            assertTrue("Nd4j op name ${opDef.name} with tensorflow mapping ${tensorflowOpDef.name}  has missing mapping ${argDef.name}",nd4jNamesMapped.contains(argDef.name))
                        }
                        OpNamespace.ArgDescriptor.ArgType.BOOL -> {
                            assertTrue("Nd4j op name ${opDef.name} with tensorflow mapping ${tensorflowOpDef.name}  has missing mapping ${argDef.name}",nd4jNamesMapped.contains(argDef.name))
                        }
                    }

                }

            }
    }

    @Test
    fun testInputOutputNames() {
        val tensorflowOpNames = tensorflowOpRegistry.inputFrameworkOpNames()
        val nd4jOpNames = tensorflowOpRegistry.nd4jOpNames()
        tensorflowOpRegistry.mappingProcessNames().map {
            tensorflowOpRegistry.lookupOpMappingProcess(it)
        }.forEach {
            println("Beginning processing of op ${it.inputFrameworkOpName()} and nd4j op ${it.opName()}")
            assertTrue(tensorflowOpNames.contains(it.inputFrameworkOpName()))
            assertTrue(nd4jOpNames.contains(it.opName()))
            val nd4jOpDef = tensorflowOpRegistry.lookupNd4jOpDef(it.opName())
            val tensorflowOpDef = tensorflowOpRegistry.lookupInputFrameworkOpDef(it.inputFrameworkOpName())
            val inputNameArgDefs = nd4jOpDef.argDescriptorList.filter {
                    argDef -> argDef.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
            }.map { argDef -> argDef.name }

            val inputFrameworkOpDefNames = tensorflowOpDef.inputArgList.map { tfOpDef -> tfOpDef.name}

            val nd4jArgDefNames = nd4jOpDef.argDescriptorList.map { nd4jArgDef -> nd4jArgDef.name }
            val tfAttrNames = tensorflowOpDef.attrList.map { tfAttr -> tfAttr.name }
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
                            assertTrue(tfAttrNames.contains(attrMapping.value)  || inputFrameworkOpDefNames.contains(attrMapping.value))
                        }

                    }
                }
            }

        }
    }

    @Test
    fun testOpExecution() {
        Nd4j.getRandom().setSeed(12345)
        val tensorflowOpNames = tensorflowOpRegistry.inputFrameworkOpNames()
        val nd4jOpNames = tensorflowOpRegistry.nd4jOpNames()
        val dynamicOps = mapOf(
            "pow" to mapOf("y" to TensorProto {
                DoubleData(listOf(1.0))
                dtype = DataType.DT_DOUBLE
                tensorShape = TensorShapeProto {
                    Dims(listOf(1,1))
                }
            })
        )
        val scalarInputs = mapOf(
            "abs" to -1.0,
            "acos" to 1.0,
            "acosh" to 1.0,
            "asin" to 1.0,
            "asinh" to 1.0,
            "atan" to 1.0,
            "atanh" to 0.5,
            "ceil" to 1.0,
            "cos" to 1.0,
            "cosh" to 1.0,
            "erf" to 1.0,
            "elu" to 1.0,
            "erfc" to 1.0,
            "exp" to 1.0,
            "expm1" to 1.0,
            "floor" to 1.0,
            "identity" to 1.0,
            "isfinite" to 1.0,
            "isinf" to 1.0,
            "isnan" to 1.0,
            //"leakyrelu" to 1.5,
            //"identity_n" to 1.0,
            "log" to 1.0,
            "log1p" to 1.0,
            "neg" to 1.0,
            "ones_as" to 1.0,
            "Reciprocal" to 1.0,
            "rank" to 1.0,
            "relu6" to 1.0,
            "rint" to 1.0,
            "round" to 1.0,
            "rsqrt" to 1.0,
            "sigmoid" to 1.0,
            "sign" to 1.0,
            "size" to 1.0,
            "sin" to 1.0,
            "sinh" to 1.0,
            "square" to 1.0,
            "sqrt" to 1.0,
            "tan" to 1.0,
            "tanh" to 1.0,
            "selu" to 1.0,
            "softsign" to 1.0,
            "softplus" to 1.0,
            "zeroslike" to 1.0)

        val singleInputOps = scalarInputs.keys

        val pairWiseInputs = mapOf(
            "add" to listOf(1.0,1.0),
            "divide" to listOf(1.0,1.0),
            "greater" to listOf(1.0,1.0),
            "less" to listOf(1.0,1.0),
            "less_equal" to listOf(1.0,1.0),
            "multiply" to listOf(1.0,1.0),
            "floordiv" to listOf(1.0,1.0),
            "mod" to listOf(1.0,1.0),
            "squaredsubtract" to listOf(1.0,1.0),
            "not_equals" to listOf(1.0,1.0),
            "realdiv" to listOf(1.0,1.0),
            "tf_atan2" to listOf(1.0,1.0),
            "min_pairwise" to listOf(1.0,1.0),
            "equals" to listOf(1.0,1.0),
            "min_pairwise" to listOf(1.0,1.0),
            "divide_no_nan" to listOf(1.0,1.0),
            "merge_sum" to listOf(2.0,3.0),
            "zeta" to listOf(2.0,3.0)


        )




        /**
         * Conv and Pooling2d ops
         */


        /**
         * Conv3d and Pooling3d ops
         */


        /**
         * Control flow ops
         */

        /**
         * Random distribution ops
         */


        /**
         * Creation ops
         * Empty
         * CopyHost
         * Linspace
         * OnesLike
         */

        /**
         * Scatter ops:
         * scatter_div
         * scatter_add
         * scatter_sub
         * scatter_min
         * scatter_mul
         * scatter_update
         * scatter_nd
         * scatter_nd_add
         * scatter_nd_sub
         * scatter_nd_update
         */




        val pairWiseIntOps = mapOf(
            "fmod" to listOf(1,1),
            "rshift_bits" to listOf(1,1),
            "truncatediv" to listOf(1,1),
            "bitwise_and" to listOf(1,1),
            "bitwise_or" to listOf(1,1),
            "bitwise_xor" to listOf(1,1),
            "shift_bits" to listOf(1,1)
        )

        val pairWiseNames = pairWiseInputs.keys


        val singularReduceOps = mapOf(
            "reduce_mean" to Nd4j.linspace(1,4,4).reshape(2,2),
            "reduce_prod" to Nd4j.linspace(1,4,4).reshape(2,2),
            "reduce_min" to Nd4j.linspace(1,4,4).reshape(2,2),
            "reduce_sum" to Nd4j.linspace(1,4,4).reshape(2,2),
            "reduce_max" to Nd4j.linspace(1,4,4).reshape(2,2)
        )



        val mappedOps = setOf(
            //"Assert", //TODO: may not need to test
            "softmax",
            "relu",
            "relu6",
            "argmin",
            "argmax",
            "cross",
            "cumsum",
            "cumprod",
            "diag",
            "diag_part",
            "digamma",
            "depth_to_space",
            "in_top_k",
            "lu",
            "matrix_inverse",
            "matrix_determinant",
            "reshape",
            "noop",
            "nth_element",
            "non_max_suppression_overlaps",
            "non_max_suppression",
            "non_max_suppression_v3",
            "onehot",
            "pad",
            "pow",
            "transpose",
            "space_to_depth",
            "Where",
            "unsorted_segment_max",
            "unsorted_segment_min",
            "unsorted_segment_prod",
            "unsorted_segment_sum",
            "unique_with_counts",
            "unique",
            "boolean_and",
            "boolean_not",
            "segment_mean",
            "segment_min",
            "segment_max",
            "segment_prod",
            "segment_sum"

            //"scatter_add", Skipping due to different op validation
            //"scatter_sub", Skipping due to different op validation
            //"scatter_update", Skipping due to different op validation
            //"scatter_nd" Skipping due to different op validation
        )




        //Skipping due to using references rather than tensors
        //"scatter_nd_add",
        //"scatter_nd_sub",
        // "scatter_nd_update"
        // //"scatter_min",
        //            //"scatter_mul",)

        val singularReduceNames = singularReduceOps.keys
        val testedOps = HashSet<String>()
        tensorflowOpRegistry.mappingProcessNames().map {
            tensorflowOpRegistry.lookupOpMappingProcess(it)
        }.forEach {
            val nd4jOpDef = tensorflowOpRegistry.lookupNd4jOpDef(it.opName())
            val tensorflowOpDef = tensorflowOpRegistry.lookupInputFrameworkOpDef(it.inputFrameworkOpName())
            val mappingProcess = tensorflowOpRegistry.lookupOpMappingProcess(tensorflowOpDef.name)

            if(singleInputOps.contains(nd4jOpDef.name) && tensorflowOpDef.name != "Variable" && tensorflowOpDef.name != "VariableV2" && tensorflowOpDef.name != "Const") {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                println("Running test import process for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    Input("x")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }


                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }
                val tensorflowGraph = TensorflowIRGraph(graphDef, tensorflowOps)
                val mappedGraph = importGraph(tensorflowGraph,null,null)!!
                val xVal =  Nd4j.scalar(scalarInputs[mappingProcess.opName()]).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
                val tensorflowRunner = TensorflowIRGraphRunner(irGraph =   tensorflowGraph,inputNames = listOf("x"),outputNames = listOf("output"))
                val inputs = mapOf("x" to xVal)
                if(!mappedGraph.hasVariable("output"))
                    throw IllegalStateException("No output variable found. Variables include ${mappedGraph.variables}")
                val tfResults = tensorflowRunner.run(inputs)
                val results = mappedGraph.output(inputs,"output")
                val tfOutput = tfResults["output"]!!
                assertTrue(tfOutput.isScalar)
                val nd4jOutput = results["output"]!!
                assertTrue(nd4jOutput.isScalar)
                assertEquals("Function ${nd4jOpDef.name} failed with input $xVal",nd4jOutput.getDouble(0), tfOutput.getDouble(0),1e-3)
                testedOps.add(nd4jOpDef.name)
            }

            else if(singularReduceNames.contains(nd4jOpDef.name)) {
                listOf(listOf(0),listOf(-1),listOf(0,1)).forEach { dimensions ->
                    listOf(true,false).forEach { keepDim ->
                        val tensorNode = NodeDef {
                            name = "x"
                            op = "Placeholder"
                            Attribute("dtype",AttrValue {
                                type = DataType.DT_DOUBLE
                            })
                        }

                        val opNode = NodeDef {
                            Input("x")
                            Input("dimensions")
                            op = tensorflowOpDef.name
                            name = "output"
                            Attribute("T",AttrValue {
                                type = DataType.DT_DOUBLE
                            })
                            Attribute("Tidx",AttrValue {
                                type = DataType.DT_INT32
                            })
                            Attribute("keep_dims",AttrValue {
                                b = keepDim
                            })
                        }

                        val tensorNode2 = NodeDef {
                            op = "Const"
                            name = "dimensions"
                            Attribute("value",AttrValue {
                                tensor = TensorProto {
                                    Int32Data(dimensions)
                                    dtype = DataType.DT_INT32
                                    tensorShape = TensorShapeProto {
                                        Dims(listOf(1,dimensions.size.toLong()))
                                    }
                                }
                            })
                            Attribute("dtype",AttrValue {
                                type = DataType.DT_INT32
                            })
                        }

                        val graphDef = GraphDef {
                            Node(tensorNode)
                            Node(tensorNode2)
                            Node(opNode)
                        }

                        val mappingProcess = tensorflowOpRegistry.lookupOpMappingProcess(tensorflowOpDef.name)
                        val tensorflowGraph = TensorflowIRGraph(graphDef, tensorflowOps)
                        val mappedGraph = importGraph(tensorflowGraph,null,null)!!
                        val xVal =  singularReduceOps[mappingProcess.opName()]!!.castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
                        val tensorflowRunner = TensorflowIRGraphRunner(irGraph =   tensorflowGraph,inputNames = listOf("x"),outputNames = listOf("output"))
                        val inputs = mapOf("x" to xVal)
                        val results = mappedGraph.output(inputs,"output")
                        val tfResults = tensorflowRunner.run(inputs)
                        //2 dimensions means sum the whole array, sometimes there are subtle differences in the shape like 1,1 vs a zero length array which is effectively the same thing
                        if(dimensions.size < 2)
                            assertEquals("Function ${nd4jOpDef.name} failed with input $xVal and dimension ${dimensions}",tfResults["output"]!!, results["output"]!!)
                        else
                            assertEquals("Function ${nd4jOpDef.name} failed with input $xVal and dimension ${dimensions}",tfResults["output"]!!.reshape(1,1), results["output"]!!.reshape(1,1))

                    }

                }

                testedOps.add(nd4jOpDef.name)

            }


            else if(pairWiseNames.contains(nd4jOpDef.name)) {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val tensorNode2 = NodeDef {
                    op = "Placeholder"
                    name = "y"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("y")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                    Node(tensorNode2)
                }

                val mappingProcess = tensorflowOpRegistry.lookupOpMappingProcess(tensorflowOpDef.name)
                val tensorflowGraph = TensorflowIRGraph(graphDef, tensorflowOps)
                val mappedGraph = importGraph(tensorflowGraph,null,null,dynamicVariables = mapOf("y" to TensorProto {
                    dtype = DataType.DT_DOUBLE
                    DoubleData(listOf(1.0))
                    Shape(listOf(1,1))
                }))!!

                val xVal =  Nd4j.scalar(pairWiseInputs[mappingProcess.opName()]!![0])
                    .reshape(1,1)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
                val yVal =  Nd4j.scalar(pairWiseInputs[mappingProcess.opName()]!![1])
                    .reshape(1,1)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val tensorflowRunner = TensorflowIRGraphRunner(irGraph =   tensorflowGraph,inputNames = listOf("x","y"),outputNames = listOf("output"))
                val inputs = mapOf("x" to xVal,"y" to yVal)
                val results = mappedGraph.output(inputs,"output")
                val tfResults = tensorflowRunner.run(inputs)
                assertEquals("Function ${nd4jOpDef.name} failed with input $xVal",tfResults["output"]!!.reshape(1,1), results["output"]!!.reshape(1,1))
                testedOps.add(nd4jOpDef.name)

            }

            else if(pairWiseIntOps.contains(nd4jOpDef.name)) {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val tensorNode2 = NodeDef {
                    op = "Placeholder"
                    name = "y"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("y")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_INT32
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                    Node(tensorNode2)
                }

                val mappingProcess = tensorflowOpRegistry.lookupOpMappingProcess(tensorflowOpDef.name)
                val tensorflowGraph = TensorflowIRGraph(graphDef, tensorflowOps)
                val mappedGraph = importGraph(tensorflowGraph,null,null)!!
                val xVal =  Nd4j.scalar(pairWiseIntOps[mappingProcess.opName()]!![0])
                    .reshape(1,1)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

                val yVal =  Nd4j.scalar(pairWiseIntOps[mappingProcess.opName()]!![1])
                    .reshape(1,1)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

                val tensorflowRunner = TensorflowIRGraphRunner(irGraph =   tensorflowGraph,inputNames = listOf("x","y"),outputNames = listOf("output"))
                val inputs = mapOf("x" to xVal,"y" to yVal)
                val results = mappedGraph.output(inputs,"output")
                val tfResults = tensorflowRunner.run(inputs)
                assertEquals("Function ${nd4jOpDef.name} failed with input $xVal",tfResults["output"]!!.reshape(1,1), results["output"]!!.reshape(1,1))
                testedOps.add(nd4jOpDef.name)

            } else if(mappedOps.contains(mappingProcess.opName())) {
                val graphInputList = graphForOp(nd4jOpName = mappingProcess.opName(),inputFrameworkOpName = mappingProcess.inputFrameworkOpName())
                graphInputList.forEach { graphInput ->
                    val tensorflowGraph = TensorflowIRGraph(graphInput.graphDef, tensorflowOps)
                    val dynamicOpsMap = if(dynamicOps.containsKey(mappingProcess.opName())) dynamicOps[mappingProcess.opName()]!! else emptyMap()
                    val mappedGraph = importGraph(tensorflowGraph,null,null,dynamicOpsMap)!!
                    val tensorflowRunner = TensorflowIRGraphRunner(irGraph =  tensorflowGraph,inputNames = graphInput.inputNames,outputNames = graphInput.outputNames)
                    val bannedOps = setOf("noop","unique","unique_with_counts","matrix_determinant")
                    if(!bannedOps.contains(mappingProcess.opName())) {
                        val tfResults = tensorflowRunner.run(graphInput.inputArrays)
                        val results = mappedGraph.output(graphInput.inputArrays,graphInput.outputNames)
                        assertEquals("Function ${nd4jOpDef.name} failed with input ${graphInput.inputNames}",tfResults, results)
                    } else if(mappingProcess.opName() == "unique_with_counts" || mappingProcess.opName() == "unique") {
                        //note: this is a separate case since the results are equal, minus dimensions
                        val results = mappedGraph.output(graphInput.inputArrays,graphInput.outputNames)
                        val tfResults = tensorflowRunner.run(graphInput.inputArrays)
                        assertEquals("Function ${nd4jOpDef.name} failed with input ${graphInput.inputNames}",tfResults["output"]!!.ravel(), results["output"]!!.ravel())
                    }//slight difference in scalar result, doesn't matter in practice
                    else if(mappingProcess.opName() == "matrix_determinant" ) {
                        //note: this is a separate case since the results are equal, minus dimensions
                        val results = mappedGraph.output(graphInput.inputArrays,graphInput.outputNames)
                        val tfResults = tensorflowRunner.run(graphInput.inputArrays)
                        assertEquals("Function ${nd4jOpDef.name} failed with input ${graphInput.inputNames}",tfResults["output"]!!.ravel().getDouble(0), results["output"]!!.ravel().getDouble(0),1e-3)
                    }

                }

                testedOps.add(nd4jOpDef.name)

            }
        }

        val differenceOfSet = tensorflowOpRegistry.mappedNd4jOpNames() - testedOps
        print("Ops left to test is ${differenceOfSet.size} and ops are $differenceOfSet")

    }





    fun graphForOp(nd4jOpName: String,inputFrameworkOpName: String): List<GraphInput> {
        val tensorflowOpDef = tensorflowOpRegistry.lookupInputFrameworkOpDef(inputFrameworkOpName)
        when (nd4jOpName) {

            "non_max_suppression","non_max_suppression_v3" -> {
                if(inputFrameworkOpName == "NonMaxSuppression") {
                    val overlaps = NodeDef {
                        name = "overlaps"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_FLOAT
                        })
                    }

                    val scores = NodeDef {
                        name = "scores"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_FLOAT
                        })
                    }

                    val maxOutputSize = NodeDef {
                        name = "maxOutputSize"
                        op = "Const"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                        Attribute("value",AttrValue {
                            tensor = TensorProto {
                                Int32Data(listOf(1))
                                dtype = DataType.DT_INT32

                            }
                        })
                    }



                    val opNode = NodeDef {
                        Input("overlaps")
                        Input("scores")
                        Input("maxOutputSize")
                        op = tensorflowOpDef.name
                        name = "output"
                        Attribute("iou_threshold",AttrValue {
                            f = 0.5f
                        })
                    }

                    val graphDef = GraphDef {
                        Node(overlaps)
                        Node(scores)
                        Node(maxOutputSize)
                        Node(opNode)
                    }



                    val overlapsVal = Nd4j.create(arrayOf(
                        floatArrayOf(0f,0f,1f,1f),
                        floatArrayOf(0f,0.1f,1f,1.1f),
                        floatArrayOf(0f,-0.1f,1f,0.9f),
                        floatArrayOf(0f,10f,1f,11f)
                    )).castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                    val scoresVal = Nd4j.create(listOf(0.9f,0.75f,0.6f,0.95f).toFloatArray())
                        .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                    val inputs = mapOf("overlaps" to overlapsVal,"scores" to scoresVal)

                    return listOf(GraphInput(
                        graphDef = graphDef,
                        inputNames = listOf("overlaps","scores"),
                        outputNames = listOf("output"),
                        inputArrays = inputs
                    ))
                }
                else if(inputFrameworkOpName == "NonMaxSuppressionV2") {
                    val overlaps = NodeDef {
                        name = "overlaps"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_FLOAT
                        })
                    }

                    val scores = NodeDef {
                        name = "scores"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_FLOAT
                        })
                    }

                    val maxOutputSize = NodeDef {
                        name = "maxOutputSize"
                        op = "Const"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                        Attribute("value",AttrValue {
                            tensor = TensorProto {
                                Int32Data(listOf(1))
                                dtype = DataType.DT_INT32

                            }
                        })
                    }

                    val iouThreshold = NodeDef {
                        name = "iouThreshold"
                        op = "Const"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_FLOAT
                        })
                        Attribute("value",AttrValue {
                            tensor = TensorProto {
                                FloatData(listOf(0.5f))
                                dtype = DataType.DT_FLOAT

                            }
                        })
                    }



                    val opNode = NodeDef {
                        Input("overlaps")
                        Input("scores")
                        Input("maxOutputSize")
                        Input("iouThreshold")
                        op = tensorflowOpDef.name
                        name = "output"

                    }

                    val graphDef = GraphDef {
                        Node(overlaps)
                        Node(scores)
                        Node(iouThreshold)
                        Node(maxOutputSize)
                        Node(opNode)
                    }



                    val overlapsVal = Nd4j.create(arrayOf(
                        floatArrayOf(0f,0f,1f,1f),
                        floatArrayOf(0f,0.1f,1f,1.1f),
                        floatArrayOf(0f,-0.1f,1f,0.9f),
                        floatArrayOf(0f,10f,1f,11f)
                    )).castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                    val scoresVal = Nd4j.create(listOf(0.9f,0.75f,0.6f,0.95f).toFloatArray())
                        .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                    val inputs = mapOf("overlaps" to overlapsVal,"scores" to scoresVal)

                    return listOf(GraphInput(
                        graphDef = graphDef,
                        inputNames = listOf("overlaps","scores"),
                        outputNames = listOf("output"),
                        inputArrays = inputs
                    ))
                } else {
                    //V3 and later
                    val overlaps = NodeDef {
                        name = "overlaps"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_FLOAT
                        })
                    }

                    val scores = NodeDef {
                        name = "scores"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_FLOAT
                        })
                    }

                    val maxOutputSize = NodeDef {
                        name = "maxOutputSize"
                        op = "Const"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                        Attribute("value",AttrValue {
                            tensor = TensorProto {
                                Int32Data(listOf(1))
                                dtype = DataType.DT_INT32

                            }
                        })
                    }

                    val overlapThreshold = NodeDef {
                        name = "iouThreshold"
                        op = "Const"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_FLOAT
                        })
                        Attribute("value",AttrValue {
                            tensor = TensorProto {
                                FloatData(listOf(0.5f))
                                dtype = DataType.DT_FLOAT

                            }
                        })
                    }

                    val scoreThreshold = NodeDef {
                        name = "scoreThreshold"
                        op = "Const"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_FLOAT
                        })
                        Attribute("value",AttrValue {
                            tensor = TensorProto {
                                FloatData(listOf(0.5f))
                                dtype = DataType.DT_FLOAT

                            }
                        })
                    }

                    val opNode = NodeDef {
                        Input("overlaps")
                        Input("scores")
                        Input("maxOutputSize")
                        Input("iouThreshold")
                        Input("scoreThreshold")
                        op = tensorflowOpDef.name
                        name = "output"

                    }

                    val graphDef = GraphDef {
                        Node(overlaps)
                        Node(scores)
                        Node(scoreThreshold)
                        Node(overlapThreshold)
                        Node(maxOutputSize)
                        Node(opNode)
                    }



                    val overlapsVal = Nd4j.create(arrayOf(
                        floatArrayOf(0f,0f,1f,1f),
                        floatArrayOf(0f,0.1f,1f,1.1f),
                        floatArrayOf(0f,-0.1f,1f,0.9f),
                        floatArrayOf(0f,10f,1f,11f)
                    )).castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                    val scoresVal = Nd4j.create(listOf(0.9f,0.75f,0.6f,0.95f).toFloatArray())
                        .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                    val inputs = mapOf("overlaps" to overlapsVal,"scores" to scoresVal)

                    return listOf(GraphInput(
                        graphDef = graphDef,
                        inputNames = listOf("overlaps","scores"),
                        outputNames = listOf("output"),
                        inputArrays = inputs
                    ))
                }
            }

            "non_max_suppression_overlaps" -> {
                val overlaps = NodeDef {
                    name = "overlaps"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val scores = NodeDef {
                    name = "scores"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val maxOutputSize = NodeDef {
                    name = "maxOutputSize"
                    op = "Const"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            Int32Data(listOf(1))
                            dtype = DataType.DT_INT32

                        }
                    })
                }

                val overlapThreshold = NodeDef {
                    name = "overlapThreshold"
                    op = "Const"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_FLOAT
                    })
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            FloatData(listOf(2.0f))
                            dtype = DataType.DT_FLOAT

                        }
                    })
                }

                val scoreThreshold = NodeDef {
                    name = "scoreThreshold"
                    op = "Const"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_FLOAT
                    })
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            FloatData(listOf(0.5f))
                            dtype = DataType.DT_FLOAT

                        }
                    })
                }

                val opNode = NodeDef {
                    Input("overlaps")
                    Input("scores")
                    Input("maxOutputSize")
                    Input("overlapThreshold")
                    Input("scoreThreshold")
                    op = tensorflowOpDef.name
                    name = "output"

                }

                val graphDef = GraphDef {
                    Node(overlaps)
                    Node(scores)
                    Node(scoreThreshold)
                    Node(overlapThreshold)
                    Node(maxOutputSize)
                    Node(opNode)
                }



                val overlapsVal = Nd4j.create(arrayOf(
                    floatArrayOf(0f,0f,1f,1f),
                    floatArrayOf(0f,0.1f,1f,1.1f),
                    floatArrayOf(0f,-0.1f,1f,0.9f),
                    floatArrayOf(0f,10f,1f,11f)
                )).castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val scoresVal = Nd4j.create(listOf(0.9f,0.75f,0.6f,0.95f).toFloatArray())
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val inputs = mapOf("overlaps" to overlapsVal,"scores" to scoresVal)

                return listOf(GraphInput(
                    graphDef = graphDef,
                    inputNames = listOf("overlaps","scores"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))
            }

            "nth_element" -> {
                val ret = ArrayList<GraphInput>()
                listOf(true,false).forEach { reverse ->
                    val input = NodeDef {
                        name = "input"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }

                    val n = NodeDef {
                        name = "n"
                        op = "Const"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                        Attribute("value",AttrValue {
                            tensor = TensorProto {
                                Int32Data(listOf(2))
                                dtype = DataType.DT_INT32

                            }
                        })
                    }

                    val opNode = NodeDef {
                        Input("input")
                        Input("n")
                        op = tensorflowOpDef.name
                        name = "output"
                        Attribute("T",AttrValue {
                            type = DataType.DT_INT32
                        })

                        Attribute("reverse",AttrValue {
                            type = DataType.DT_BOOL
                            b = reverse
                        })

                    }

                    val graphDef = GraphDef {
                        Node(input)
                        Node(n)
                        Node(opNode)
                    }


                    val xVal = Nd4j.linspace(1, 6, 6)
                        .reshape(2, 3)
                        .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

                    val inputs = mapOf("input" to xVal)

                    ret.add(GraphInput(
                        graphDef =graphDef, inputNames = listOf("input"),
                        outputNames = listOf("output"),
                        inputArrays = inputs
                    ))
                }

                return ret
            }

            "matrix_determinant" -> {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }


                val opNode = NodeDef {
                    Input("x")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })

                }

                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val inputs = mapOf("x" to xVal)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("x"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))
            }


            "lu" -> {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }


                val opNode = NodeDef {
                    Input("x")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })

                }

                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val inputs = mapOf("x" to xVal)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("x"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))
            }

            "matrix_inverse" -> {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }


                val opNode = NodeDef {
                    Input("x")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })

                }

                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val inputs = mapOf("x" to xVal)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("x"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))
            }

            "in_top_k" -> {
                if(tensorflowOpDef.name == "InTopK") {
                    val tensorNode = NodeDef {
                        name = "x"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_FLOAT
                        })
                    }

                    val predictions = NodeDef {
                        name = "predictions"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }

                    val opNode = NodeDef {
                        Input("x")
                        Input("predictions")
                        op = tensorflowOpDef.name
                        name = "output"
                        Attribute("T",AttrValue {
                            type = DataType.DT_INT32
                        })
                        Attribute("k",AttrValue {
                            i = 2
                        })
                    }



                    val graphDef = GraphDef {
                        Node(tensorNode)
                        Node(predictions)
                        Node(opNode)
                    }


                    val xVal = Nd4j.linspace(1, 4, 4)
                        .reshape(2, 2)
                        .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                    val predictionsArr = Nd4j.linspace(1, 2, 2)
                        .reshape(2)
                        .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


                    val inputs = mapOf("x" to xVal,"predictions" to predictionsArr)


                    return listOf(GraphInput(
                        graphDef =graphDef, inputNames = listOf("x","predictions"),
                        outputNames = listOf("output"),
                        inputArrays = inputs
                    ))
                } else {
                    val tensorNode = NodeDef {
                        name = "x"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_FLOAT
                        })
                    }

                    val predictions = NodeDef {
                        name = "predictions"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }

                    val k = NodeDef {
                        name = "k"
                        op = "Const"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                        Attribute("value",AttrValue {
                            tensor = TensorProto {
                                Int32Data(listOf(2))
                                dtype = DataType.DT_INT32

                            }
                        })
                    }

                    val opNode = NodeDef {
                        Input("x")
                        Input("predictions")
                        Input("k")
                        op = tensorflowOpDef.name
                        name = "output"
                        Attribute("T",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }



                    val graphDef = GraphDef {
                        Node(tensorNode)
                        Node(predictions)
                        Node(k)
                        Node(opNode)
                    }


                    val xVal = Nd4j.linspace(1, 4, 4)
                        .reshape(2, 2)
                        .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                    val predictionsArr = Nd4j.linspace(1, 2, 2)
                        .reshape(2)
                        .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


                    val inputs = mapOf("x" to xVal,"predictions" to predictionsArr)


                    return listOf(GraphInput(
                        graphDef =graphDef, inputNames = listOf("x","predictions"),
                        outputNames = listOf("output"),
                        inputArrays = inputs
                    ))
                }



            }


            "onehot" -> {
                val indices = NodeDef {
                    name = "indices"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT64
                    })
                }

                val depth = NodeDef {
                    name = "depth"
                    op = "Const"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            dtype = DataType.DT_INT32
                            Int32Data(listOf(1))

                        }
                    })
                }

                val onValue = NodeDef {
                    name = "on"
                    op = "Const"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT64
                    })
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            dtype = DataType.DT_INT64
                            Int64Data(listOf(1))

                        }
                    })
                }


                val offValue = NodeDef {
                    name = "off"
                    op = "Const"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT64
                    })
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            dtype = DataType.DT_INT64
                            Int64Data(listOf(0))

                        }
                    })
                }


                val opNode = NodeDef {
                    Input("indices")
                    Input("depth")
                    Input("on")
                    Input("off")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("TI",AttrValue {
                        type = DataType.DT_INT64
                    })
                    Attribute("T",AttrValue {
                        type = DataType.DT_INT64
                    })

                    Attribute("axis",AttrValue {
                        i = 0
                    })
                }



                val graphDef = GraphDef {
                    Node(indices)
                    Node(depth)
                    Node(onValue)
                    Node(offValue)
                    Node(opNode)
                }


                val indicesVal = Nd4j.linspace(1, 4, 4)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT64)
                val inputs = mapOf("indices" to indicesVal)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("indices"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))
            }

            "cross" -> {
                val a = NodeDef {
                    name = "a"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val b = NodeDef {
                    name = "b"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val opNode = NodeDef {
                    Input("a")
                    Input("b")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_FLOAT
                    })

                }



                val graphDef = GraphDef {
                    Node(a)
                    Node(b)
                    Node(opNode)
                }


                val aVal = Nd4j.linspace(1, 27, 27)
                    .reshape(3,3,3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val bVal = Nd4j.linspace(1, 27, 27)
                    .reshape(3,3,3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


                val inputs = mapOf("a" to aVal,"b" to bVal)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("a","b"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))
            }

            "transpose" ->  {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val tensorNode2 = NodeDef {
                    op = "Const"
                    name = "perm"
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            Int32Data(listOf(0,1))
                            Shape(listOf(2))
                            dtype = DataType.DT_INT32
                        }
                    })
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("perm")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                    Attribute("Tperm",AttrValue {
                        type = DataType.DT_INT32
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(tensorNode2)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)



                val inputs = mapOf("x" to xVal)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("x"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))

            }
            "relu", "relu6" -> {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }


                val opNode = NodeDef {
                    Input("x")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val inputs = mapOf("x" to xVal)

                return listOf(GraphInput(
                    graphDef = GraphDef {
                        Node(tensorNode)
                        Node(opNode)
                    }, inputNames = listOf("x"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))
            }

            "depth_to_space","space_to_depth" -> {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }


                val opNode = NodeDef {
                    Input("x")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                    Attribute("data_format", AttrValue {
                        s = ByteString.copyFrom("NHWC".toByteArray(Charset.defaultCharset()))
                    })
                    Attribute("block_size", AttrValue {
                        i = 2
                    })
                }

                val xVal = Nd4j.linspace(1, 256, 256)
                    .reshape(4, 4,4,4)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val inputs = mapOf("x" to xVal)

                return listOf(GraphInput(
                    graphDef = GraphDef {
                        Node(tensorNode)
                        Node(opNode)
                    }, inputNames = listOf("x"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))
            }

            "softmax","digamma","diag","diag_part" -> {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }


                val opNode = NodeDef {
                    Input("x")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val inputs = mapOf("x" to xVal)

                return listOf(GraphInput(
                    graphDef = GraphDef {
                        Node(tensorNode)
                        Node(opNode)
                    }, inputNames = listOf("x"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))
            }

            "cumsum","cumprod" -> {
                val ret = ArrayList<GraphInput>()
                listOf(false,true).forEach { reverse ->
                    listOf(false,true).forEach { exclusive ->
                        val inputNames = listOf("x")
                        val tensorNode = NodeDef {
                            name = "x"
                            op = "Placeholder"
                            Attribute("dtype",AttrValue {
                                type = DataType.DT_DOUBLE
                            })
                        }

                        val dimensions = listOf(1)
                        val tensorNode2 = NodeDef {
                            op = "Const"
                            name = "dimensions"
                            Attribute("value",AttrValue {
                                tensor = TensorProto {
                                    Int32Data(dimensions)
                                    dtype = DataType.DT_INT32
                                    tensorShape = TensorShapeProto {
                                        Dims(listOf())
                                    }
                                }
                            })
                            Attribute("dtype",AttrValue {
                                type = DataType.DT_INT32
                            })
                        }

                        val opNode = NodeDef {
                            Input("x")
                            Input("dimensions")
                            op = tensorflowOpDef.name
                            name = "output"
                            Attribute("T",AttrValue {
                                type = DataType.DT_DOUBLE
                            })
                            Attribute("exclusive",AttrValue {
                                b = exclusive
                            })

                            Attribute("reverse",AttrValue {
                                b = reverse
                            })
                        }



                        val graphDef = GraphDef {
                            Node(tensorNode)
                            Node(tensorNode2)
                            Node(opNode)
                        }

                        val xVal = Nd4j.linspace(1, 4, 4)
                            .reshape(2, 2)
                            .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)


                        val inputs = mapOf("x" to xVal)
                        ret.add(GraphInput(
                            graphDef =graphDef, inputNames = inputNames,
                            outputNames = listOf("output"),
                            inputArrays = inputs
                        ))
                    }
                }

                return ret

            }

            "Assert" -> {
                val tensorNode = NodeDef {
                    name = "condition"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_BOOL
                    })
                }

                val tensorNode2 = NodeDef {
                    op = "Placeholder"
                    name = "data"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                println("Running op def for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    Input("condition")
                    Input("data")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        ListDataType(listOf(DataType.DT_DOUBLE))
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                    Node(tensorNode2)
                }

                val inputs = mapOf("data" to Nd4j.linspace(1,4,4).castTo(
                    org.nd4j.linalg.api.buffer.DataType.DOUBLE
                ),"condition" to Nd4j.ones(2).addi(1).castTo(org.nd4j.linalg.api.buffer.DataType.BOOL))
                return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("condition","data"),outputNames = listOf("output"),inputArrays = inputs))
            }


            "Where" -> {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }


                println("Running op def for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    Input("x")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }

                val inputs = mapOf("x" to Nd4j.linspace(1,4,4).castTo(
                    org.nd4j.linalg.api.buffer.DataType.DOUBLE
                ))
                return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x"),outputNames = listOf("output"),inputArrays = inputs))
            }


            "boolean_and" -> {
                println("Running op def for op ${tensorflowOpDef.name}")
                val inputNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_BOOL
                    })
                }


                val secondNode = NodeDef {
                    name = "y"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_BOOL
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("y")
                    name = "and"
                    op = tensorflowOpDef.name
                }


                val inputs = mapOf("x" to Nd4j.ones(2,2).castTo(
                    org.nd4j.linalg.api.buffer.DataType.BOOL
                ), "y" to Nd4j.zeros(2,2).castTo(
                    org.nd4j.linalg.api.buffer.DataType.BOOL
                ))


                val graphDef = GraphDef {
                    Node(inputNode)
                    Node(secondNode)
                    Node(opNode)
                }

                return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x","y"),outputNames = listOf("and"),inputArrays = inputs))
            }


            "boolean_not" -> {
                println("Running op def for op ${tensorflowOpDef.name}")
                val inputNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_BOOL
                    })
                }




                val opNode = NodeDef {
                    Input("x")
                    name = "not"
                    op = tensorflowOpDef.name
                }


                val inputs = mapOf("x" to Nd4j.ones(2,2).castTo(
                    org.nd4j.linalg.api.buffer.DataType.BOOL
                ))

                val graphDef = GraphDef {
                    Node(inputNode)
                    Node(opNode)
                }

                return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x"),outputNames = listOf("not"),inputArrays = inputs))
            }


            "noop" -> {
                println("Running op def for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    name = "noop"
                    op = tensorflowOpDef.name
                }



                val graphDef = GraphDef {
                    Node(opNode)
                }

                return listOf(GraphInput(graphDef = graphDef,inputNames = listOf(),outputNames = listOf(),inputArrays = emptyMap()))
            }

            "While" -> {
                println("Running op def for op ${tensorflowOpDef.name}")
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        ListDataType(listOf(DataType.DT_DOUBLE))
                    })
                }


                val opNode = NodeDef {
                    Input("x")
                    name = "while"
                    op = tensorflowOpDef.name
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }

                val inputs = mapOf("x" to Nd4j.scalar(1.0))

                return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x"),outputNames = listOf("output"),inputArrays = inputs))
            }

            "unique_with_counts","unique" -> {
                println("Running op def for op ${tensorflowOpDef.name}")
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                if(tensorflowOpDef.name == "UniqueWithCountsV2" || tensorflowOpDef.name == "UniqueV2") {
                    val axis = NodeDef {
                        name = "axis"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT64
                        })
                    }


                    val opNode = NodeDef {
                        Input("x")
                        Input("axis")
                        name = "output"
                        op = tensorflowOpDef.name
                        Attribute("T",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                    }



                    val graphDef = GraphDef {
                        Node(tensorNode)
                        Node(axis)
                        Node(opNode)
                    }

                    val inputs = mapOf("x" to Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE),
                        "axis" to Nd4j.scalar(1).reshape(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT64))

                    return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x","axis"),outputNames = listOf("output"),inputArrays = inputs))
                }
                else {
                    val opNode = NodeDef {
                        Input("x")
                        name = "output"
                        op = tensorflowOpDef.name
                        Attribute("T",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                    }



                    val graphDef = GraphDef {
                        Node(tensorNode)
                        Node(opNode)
                    }

                    val inputs = mapOf("x" to Nd4j.linspace(1,4,4).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE))

                    return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x"),outputNames = listOf("output"),inputArrays = inputs))
                }

            }


            "pad" -> {
                if(tensorflowOpDef.name == "Pad") {
                    val tensorNode = NodeDef {
                        name = "x"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                    }

                    val tensorNode2 = NodeDef {
                        op = "Placeholder"
                        name = "paddings"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }

                    val opNode = NodeDef {
                        Input("x")
                        Input("paddings")
                        op = tensorflowOpDef.name
                        name = "output"
                        Attribute("T",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                        Attribute("Tpaddings",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }



                    val graphDef = GraphDef {
                        Node(tensorNode)
                        Node(opNode)
                        Node(tensorNode2)
                    }

                    val inputs = mapOf("x" to Nd4j.linspace(1,4,4).castTo(
                        org.nd4j.linalg.api.buffer.DataType.DOUBLE
                    ),"paddings" to Nd4j.ones(1,2).addi(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT32))
                    return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x","paddings"),outputNames = listOf("output"),inputArrays = inputs))
                } else if(tensorflowOpDef.name == "PadV2"){
                    val tensorNode = NodeDef {
                        name = "x"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                    }

                    val tensorNode2 = NodeDef {
                        op = "Placeholder"
                        name = "paddings"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }

                    val constantValues = NodeDef {
                        op = "Const"
                        name = "constant_values"
                        Attribute("value",AttrValue {
                            tensor = TensorProto {
                                DoubleData(listOf(1.0))
                                dtype = DataType.DT_DOUBLE
                                tensorShape = TensorShapeProto {
                                    Dims(listOf())
                                }
                            }
                        })
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                    }

                    val opNode = NodeDef {
                        Input("x")
                        Input("paddings")
                        Input("constant_values")
                        op = tensorflowOpDef.name
                        name = "output"
                        Attribute("T",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                        Attribute("Tpaddings",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }



                    val graphDef = GraphDef {
                        Node(tensorNode)
                        Node(opNode)
                        Node(constantValues)
                        Node(tensorNode2)

                    }

                    val inputs = mapOf("x" to Nd4j.linspace(1,4,4).castTo(
                        org.nd4j.linalg.api.buffer.DataType.DOUBLE
                    ),"paddings" to Nd4j.ones(1,2).addi(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT32))
                    return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x","paddings"),outputNames = listOf("output"),inputArrays = inputs))
                } else {
                    throw IllegalArgumentException("Illegal mapping for padding op $tensorflowOpDef.name")
                }

            }


            "reshape" -> {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val tensorNode2 = NodeDef {
                    op = "Placeholder"
                    name = "shape"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("shape")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                    Node(tensorNode2)
                }

                val inputs = mapOf("x" to Nd4j.linspace(1,4,4).castTo(
                    org.nd4j.linalg.api.buffer.DataType.DOUBLE
                ),"shape" to Nd4j.ones(2).addi(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT32))
                return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x","shape"),outputNames = listOf("output"),inputArrays = inputs))
            }

            "argmin", "argmax" -> {
                val ret = ArrayList<GraphInput>()
                listOf(true, false).forEach { keepDim ->
                    val tensorNode = NodeDef {
                        name = "x"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                    }

                    val opNode = NodeDef {
                        Input("x")
                        Input("dimensions")
                        op = tensorflowOpDef.name
                        name = "output"
                        Attribute("T",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                        Attribute("Tidx",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }

                    val dimensions = listOf(0)
                    val tensorNode2 = NodeDef {
                        op = "Const"
                        name = "dimensions"
                        Attribute("value",AttrValue {
                            tensor = TensorProto {
                                Int32Data(dimensions)
                                dtype = DataType.DT_INT32
                                tensorShape = TensorShapeProto {
                                    Dims(listOf())
                                }
                            }
                        })
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }

                    val graphDef = GraphDef {
                        Node(tensorNode)
                        Node(tensorNode2)
                        Node(opNode)
                    }


                    val xVal = Nd4j.linspace(1, 4, 4)
                        .reshape(2, 2)
                        .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                    val inputs = mapOf("x" to xVal)

                    ret.add(GraphInput(
                        graphDef =graphDef, inputNames = listOf("x"),
                        outputNames = listOf("output"),
                        inputArrays = inputs
                    ))
                }

                return ret
            }

            "pow" -> {
                val ret = ArrayList<GraphInput>()
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val tensorNode2 = NodeDef {
                    op = "Const"
                    name = "y"
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            DoubleData(listOf(1.0))
                            dtype = DataType.DT_DOUBLE
                            tensorShape = TensorShapeProto {
                                Dims(listOf())
                            }
                        }
                    })
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("y")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(tensorNode2)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val inputs = mapOf("x" to xVal)

                ret.add(GraphInput(
                    graphDef =graphDef, inputNames = listOf("x"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))

                return ret
            }



            //scatter_div
            //TODO: Revisit. TF op validation seems to be different than ours.
            "scatter_add","scatter_sub","scatter_min","scatter_sub","scatter_min","scatter_mul","scatter_update","scatter_nd","scatter_nd_add","scatter_nd_sub","scatter_nd_update" -> {
                val ret = ArrayList<GraphInput>()
                listOf(true,false).forEach { lock ->
                    val xRef = NodeDef {
                        name = "x"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                    }


                    val tensorNode2 = NodeDef {
                        op = "Placeholder"
                        name = "indices"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }


                    val updates2 = NodeDef {
                        op = "Placeholder"
                        name = "updates"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                    }

                    val opNode = NodeDef {
                        Input("x")
                        Input("indices")
                        Input("updates")
                        op = tensorflowOpDef.name
                        name = "output"
                        Attribute("T",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                        Attribute("Tindices",AttrValue {
                            type = DataType.DT_INT32
                        })
                    }


                    val graphDef = GraphDef {
                        Node(xRef)
                        Node(tensorNode2)
                        Node(updates2)
                        Node(opNode)
                    }


                    //from testScatterOpGradients.
                    val xVal = Nd4j.ones(org.nd4j.linalg.api.buffer.DataType.DOUBLE, 20, 10)
                    val indices = Nd4j.createFromArray(3, 4, 5, 10, 18)
                    val updates = Nd4j.ones(org.nd4j.linalg.api.buffer.DataType.DOUBLE, 5, 10)

                    val inputs = mapOf("x" to xVal,"indices" to indices,"updates" to updates)

                    ret.add(GraphInput(
                        graphDef =graphDef, inputNames = listOf("x","indices","updates"),
                        outputNames = listOf("output"),
                        inputArrays = inputs
                    ))
                }






                return ret
            }




            "segment_mean", "segment_min","segment_max","segment_prod","segment_sum" -> {
                val ret = ArrayList<GraphInput>()
                val tensorNode = NodeDef {
                    name = "data"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val segmentIds = NodeDef {
                    op = "Placeholder"
                    name = "segment_ids"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }


                val opNode = NodeDef {
                    Input("data")
                    Input("segment_ids")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                    Attribute("Tindices",AttrValue {
                        type = DataType.DT_INT32
                    })

                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(segmentIds)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 12, 12)
                    .reshape(3, 4)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)


                val indices = Nd4j.create(floatArrayOf(1.0f,2.0f,3.0f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
                val inputs = mapOf("data" to xVal,"segment_ids" to indices)

                ret.add(GraphInput(
                    graphDef =graphDef, inputNames = listOf("data","segment_ids"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))


                return ret
            }


            "unsorted_segment_sum", "unsorted_segment_prod","unsorted_segment_min","unsorted_segment_max" -> {
                val ret = ArrayList<GraphInput>()
                val tensorNode = NodeDef {
                    name = "data"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val segmentIds = NodeDef {
                    op = "Placeholder"
                    name = "segment_ids"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val numSegmentsNode = NodeDef {
                    op = "Const"
                    name = "num_segments"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })

                    Attribute(name = "value",value = AttrValue {
                        tensor = TensorProto {
                            Shape(listOf())
                            Int32Data(listOf(2))
                            DataType(DataType.DT_INT32)
                        }
                    })
                }

                val opNode = NodeDef {
                    Input("data")
                    Input("segment_ids")
                    Input("num_segments")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                    Attribute("Tindices",AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("Tnumsegments",AttrValue {
                        type = DataType.DT_INT32
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(segmentIds)
                    Node(numSegmentsNode)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 12, 12)
                    .reshape(3, 4)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)


                val indices = Nd4j.create(floatArrayOf(0.0f,1.0f,0.0f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
                val numSegments = Nd4j.scalar(2).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
                val inputs = mapOf("data" to xVal,"segment_ids" to indices,"num_segments" to numSegments)

                ret.add(GraphInput(
                    graphDef =graphDef, inputNames = listOf("data","segment_ids","num_segments"),
                    outputNames = listOf("output"),
                    inputArrays = inputs
                ))


                return ret
            }


            else -> {
                throw IllegalArgumentException("Illegal op name $inputFrameworkOpName")
            }
        }
    }

}


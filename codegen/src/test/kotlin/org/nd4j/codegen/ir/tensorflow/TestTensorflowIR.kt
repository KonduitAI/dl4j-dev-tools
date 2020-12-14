package org.nd4j.codegen.ir.tensorflow

import com.sun.org.apache.xpath.internal.operations.Bool
import com.sun.xml.internal.bind.v2.runtime.unmarshaller.IntData
import junit.framework.Assert.assertEquals
import junit.framework.Assert.assertTrue
import org.apache.commons.io.IOUtils
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.fail
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.codegen.ir.importGraph
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.common.io.ClassPathResource
import org.nd4j.ir.OpNamespace
import org.nd4j.shade.protobuf.ByteString
import org.tensorflow.framework.*
import java.nio.charset.Charset
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlin.test.assertTrue

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

        val mappingContext = TensorflowMappingContext(opDef = opDef,node = nodeDef,graph = tfGraph)
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
                    val mappingContext = TensorflowMappingContext(opDef = opDef,node = nodeBuilder.build(),graph = irGraph)
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
            graph = tfIRGraph
        )

        assertEquals(opDef,tfMappingCtx.opDef)

    }

    @Test
    fun testOpsMapped() {
        val tensorflowOpNames = tensorflowOpRegistry.inputFrameworkOpNames().filter { tensorflowOpRegistry.registeredOps.containsKey(it) }
        val nd4jOpNames = tensorflowOpRegistry.nd4jOpNames()
        /**
         * TODO: Assert each op is mapped.
         *
         * Assert all attributes in nd4j are mapped.
         * If not, let's document what isn't and why for each op.
         *
         * Create an op generation tool that allows random generation of test cases
         * based on existing mapped ops between nd4j and tensorflow.
         */
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

}
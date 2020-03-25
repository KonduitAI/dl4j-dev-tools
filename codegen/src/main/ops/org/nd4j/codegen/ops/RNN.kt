package org.nd4j.codegen.ops

import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*

fun SDRNN() = Namespace("SDRNN") {


    val LSTMConfiguration = Config("LSTMConfiguration") {

        Arg(ENUM, "RnnDataFormat") {
            possibleValues = listOf("TNS", "NST", "NTS"); description = " The data format of the input. Input shape depends on data format (in config):<br>\n" +
                " TNS -> [timeSteps, batchSize, inSize]<br>\n" +
                " NST -> [batchSize, inSize, timeSteps]<br>\n" +
                " NTS -> [batchSize, timeSteps, inSize]<br>"
        }


        Arg(BOOL, "peepHole") { description = "Whether to provide peephole connections"; }
        Arg(NUMERIC, "forgetBias") { description = "The bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training."; }
        Arg(NUMERIC, "clippingCellValue") { description = "The bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training."; }

        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration"
    }


    val LSTMLayerConfig = Config("LSTMLayerConfig") {

        Arg(ENUM, "LSTMDataFormat") {
            possibleValues = listOf("TNS", "NST", "NTS", "T2NS");
            description = "for unidirectional:\n" +
                    "  TNS: shape [timeLength, numExamples, inOutSize] - sometimes referred to as \"time major\"<br>\n" +
                    "  NST: shape [numExamples, inOutSize, timeLength]<br>\n" +
                    "  NTS: shape [numExamples, timeLength, inOutSize] - TF \"time_major=false\" layout<br>\n" +
                    " for bidirectional:\n" +
                    "   T2NS: 3 = [timeLength, 2, numExamples, inOutSize] (for ONNX)"
        }







        Arg(ENUM, "LSTMDirectionMode") {
            possibleValues = listOf("FWD", "BWD", "BS", "BC", "BE"); description = "direction <br>\n" +
                " FWD: 0 = fwd\n" +
                " BWD: 1 = bwd\n" +
                " BS: 2 = bidirectional sum\n" +
                " BC: 3 = bidirectional concat\n" +
                " BE: 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)"
        }

        Arg(ENUM, "gateAct") {
            possibleValues = listOf("TANH",
                    "RELU",
                    "SIGMOID",
                    "AFFINE",
                    "LEAKY_RELU",
                    "THRESHHOLD_RELU",
                    "SCALED_TAHN",
                    "HARD_SIGMOID",
                    "ELU",
                    "SOFTSIGN",
                    "SOFTPLUS"); description = "Activations"
        }


        Arg(ENUM, "cellAct") {
            possibleValues = listOf("TANH",
                    "RELU",
                    "SIGMOID",
                    "AFFINE",
                    "LEAKY_RELU",
                    "THRESHHOLD_RELU",
                    "SCALED_TAHN",
                    "HARD_SIGMOID",
                    "ELU",
                    "SOFTSIGN",
                    "SOFTPLUS"); description = "Activations"
        }


        Arg(ENUM, "outAct") {
            possibleValues = listOf("TANH",
                    "RELU",
                    "SIGMOID",
                    "AFFINE",
                    "LEAKY_RELU",
                    "THRESHHOLD_RELU",
                    "SCALED_TAHN",
                    "HARD_SIGMOID",
                    "ELU",
                    "SOFTSIGN",
                    "SOFTPLUS"); description = "Activations"
        }





        Arg(BOOL, "retFullSequence") { description = "indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}"; defaultValue = true }
        Arg(BOOL, "retLastH") {
            description = "indicates whether to return output at last time step only,\n" +
                    " in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)";
        }
        Arg(BOOL, "retLastC") {
            description = "indicates whether to return cells state at last time step only,\n" +
                    " in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)";
        }
        Arg(NUMERIC, "cellClip") { description = "Cell clipping value, if it = 0 then do not apply clipping"; }


    }


    val GRUWeights = Config("GRUWeights") {
        Input(NUMERIC, "ruWeight")
        Input(NUMERIC, "cWeight")
        Input(NUMERIC, "ruBias")
        Input(NUMERIC, "cBias")
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.GRUWeights"
    }

    val SRUWeights = Config("SRUWeights") {
        Input(NUMERIC, "weights")
        Input(NUMERIC, "bias")
        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.SRUWeights"
    }

    val LSTMWeights = Config("LSTMWeights") {
        Input(NUMERIC, "ruWeight")
        Input(NUMERIC, "inputPeepholeWeights")
        Input(NUMERIC, "forgetPeepholeWeights")
        Input(NUMERIC, "outputPeepholeWeights")
        Input(NUMERIC, "bias")

        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMWeights"
    }

    val LSTMLayerWeights = Config("LSTMLayerWeights") {
        Input(NUMERIC, "iWeights")
        Input(NUMERIC, "iInputPeepholeWeights")
        Input(NUMERIC, "iForgetPeepholeWeights")
        Input(NUMERIC, "iOutputPeepholeWeights")
        Input(NUMERIC, "iBias")

        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights"
    }



    val namespaceJavaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
    Op("gru") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "GRUCell"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "hLast") { description = "Output of the previous cell/time step, with shape [batchSize, numUnits]" }
        useConfig(GRUWeights)
        Output(NUMERIC, "output") { description = "The cell's outputs." }

        Doc(Language.ANY, DocScope.ALL) {
            """
            The GRU cell.  Does a single time step operation
            """.trimIndent()
        }
    }



    Op("lstmCell") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "LSTMBlockCell"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "cLast") { description = "Previous cell state, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "yLast") { description = "revious cell output, with shape [batchSize, numUnits]" }
        useConfig(LSTMWeights)
        useConfig(LSTMConfiguration)

        Output(NUMERIC, "output") { description = "The cell's outputs" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            The LSTM cell.  Does a single time step operation.
            """.trimIndent()
        }
    }



    Op("lstmblock") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "LSTMBlock"
        Input(NUMERIC, "maxTSLength")
        Input(NUMERIC, "x") { description = " Input, with shape dependent on the data format (in config)." }
        Input(NUMERIC, "cLast") { description = "Previous/initial cell state, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "yLast") { description = "Previous/initial cell output, with shape [batchSize, numUnits]" }
        useConfig(LSTMWeights)
        useConfig(LSTMConfiguration)

        Output(NUMERIC, "output") { description = "The layer's outputs." }

        Doc(Language.ANY, DocScope.ALL) {
            """
             The LSTM block
            """.trimIndent()
        }
    }



    Op("lstmlayer") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "LSTMLayer"
        Input(NUMERIC, "x") { description = " Input, with shape dependent on the data format (in config)." }
        Input(NUMERIC, "cLast") { description = "Previous/initial cell state, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "yLast") { description = "Previous/initial cell output, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "maxTSLength")
        useConfig(LSTMLayerWeights)
        useConfig(LSTMLayerConfig)

        Output(NUMERIC, "output") { description = "The layer's outputs." }

        Doc(Language.ANY, DocScope.ALL) {
            """
             The LSTM layer
            """.trimIndent()
        }
    }



    Op("sruCell") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "SRUCell"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "cLast") { description = "Previous cell state, with shape [batchSize, inSize]" }
        useConfig(SRUWeights)

        Output(NUMERIC, "output") { description = "The cell's outputs." }

        Doc(Language.ANY, DocScope.ALL) {
            """
             The SRU layer.  Does a single time step operation.
            """.trimIndent()
        }
    }


    Op("sru") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "SRU"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "initialC") { description = "Initial cell state, with shape [batchSize, inSize]" }
        Input(NUMERIC, "mask") { description = "An optional dropout mask, with shape [batchSize, inSize]"; defaultValue = null }

        useConfig(SRUWeights)

        Output(NUMERIC, "output") { description = "The cell's outputs.." }

        Doc(Language.ANY, DocScope.ALL) {
            """
             The SRU layer.  Does a single time step operation.
            """.trimIndent()
        }

    }
}





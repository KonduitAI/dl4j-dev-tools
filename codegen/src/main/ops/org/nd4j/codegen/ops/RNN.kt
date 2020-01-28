package org.nd4j.codegen.ops

import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*

fun SDRNN() =  Namespace("SDRNN"){

    val LSTMConfiguration = Config("LSTMConfiguration"){

        Arg(ENUM, "RnnDataFormat"){ possibleValues = listOf("TNS", "NST","NTS") ; description = " The data format of the input      * Input shape depends on data format (in config):<br>\n" +
                "     * TNS -> [timeSteps, batchSize, inSize]<br>\n" +
                "     * NST -> [batchSize, inSize, timeSteps]<br>\n" +
                "     * NTS -> [batchSize, timeSteps, inSize]<br>"}

        Arg(BOOL, "peepHole"){ description = "Whether to provide peephole connections";}
        Arg(NUMERIC, "forgetBias"){ description = "The bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training."; }
        Arg(NUMERIC, "clippingCellValue"){ description = "The bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training."; }

        javaClassOverride = "org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration"
    }



    val baseName = Config("baseName"){
        Arg(STRING, "baseName") { description = "The base name for the gru cell"; }

    }



    val namespaceJavaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
    Op("gru"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "GRUCell"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "hLast") { description = "Output of the previous cell/time step, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "weights") { description = "The cell's weights" }

        Output(NUMERIC, "output"){ description = "The cell's outputs." }

        Doc(Language.ANY, DocScope.ALL){
            """
            The GRU cell.  Does a single time step operation
            """.trimIndent()
        }

    }



    Op("gru"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "GRUCell"
        useConfig(baseName);
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "hLast") { description = "Output of the previous cell/time step, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "weights") { description = "The cell's weights" }

        Output(NUMERIC, "output"){ description = "The cell's outputs." }

        Doc(Language.ANY, DocScope.ALL){
            """
            The GRU cell.  Does a single time step operation
            """.trimIndent()
        }

    }


    Op("lstmCell"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "LSTMBlockCell"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "cLast") { description = "Previous cell state, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "yLast") { description = "revious cell output, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "weights") { description = "The cell's weights." }
        useConfig(LSTMConfiguration)

        Output(NUMERIC, "output"){ description = "The cell's outputs" }

        Doc(Language.ANY, DocScope.ALL){
            """
            The LSTM cell.  Does a single time step operation.
            """.trimIndent()
        }

    }

    Op("lstmCell"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "LSTMBlockCell"
        useConfig(baseName);
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "cLast") { description = "Previous cell state, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "yLast") { description = "revious cell output, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "weights") { description = "The cell's weights." }
        useConfig(LSTMConfiguration)

        Output(NUMERIC, "output"){ description = "The cell's outputs" }

        Doc(Language.ANY, DocScope.ALL){
            """
            The LSTM cell.  Does a single time step operation.
            """.trimIndent()
        }

    }

    Op("lstmLayer"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "LSTMLayer"
        Input(NUMERIC, "maxTSLength") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "x") { description = " Input, with shape dependent on the data format (in config)." }
        Input(NUMERIC, "cLast") { description = "Previous/initial cell state, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "yLast") { description = "Previous/initial cell output, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "weights") { description = "The cell's weights." }
        useConfig(LSTMConfiguration)

        Output(NUMERIC, "output"){ description = "The layer's outputs." }

        Doc(Language.ANY, DocScope.ALL){
            """
             The LSTM layer.  Does multiple time steps.
            """.trimIndent()
        }

    }

    Op("lstmLayer"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "LSTMLayer"
        Arg(INT, "maxTSLength")
        Input(NUMERIC, "x") { description = " Input, with shape dependent on the data format (in config)." }
        Input(NUMERIC, "cLast") { description = "Previous/initial cell state, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "yLast") { description = "Previous/initial cell output, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "weights") { description = "The layer's weights." }
        useConfig(LSTMConfiguration)

        Output(NUMERIC, "output"){ description = "The layer's outputs." }

        Doc(Language.ANY, DocScope.ALL){
            """
             The LSTM layer.  Does multiple time steps.
            """.trimIndent()
        }

    }


    Op("lstmLayer"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "LSTMLayer"
        useConfig(baseName);
        Arg(INT, "maxTSLength")
        Input(NUMERIC, "x") { description = "Input, with shape dependent on the data format (in config)." }
        Input(NUMERIC, "cLast") { description = "Previous/initial cell state, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "yLast") { description = "Previous/initial cell output, with shape [batchSize, numUnits]" }
        Input(NUMERIC, "weights") { description = "The layer's weights." }
        useConfig(LSTMConfiguration)

        Output(NUMERIC, "output"){ description = "The layer's outputs." }

        Doc(Language.ANY, DocScope.ALL){
            """
             The LSTM layer.  Does multiple time steps.
            """.trimIndent()
        }

    }

    Op("sruCell"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "SRUCell"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "cLast") { description = "Previous cell state, with shape [batchSize, inSize]" }
        Input(NUMERIC, "weights") { description = "The cell's weights." }

        Output(NUMERIC, "output"){ description = "The cell's outputs." }

        Doc(Language.ANY, DocScope.ALL){
            """
             The SRU layer.  Does a single time step operation.
            """.trimIndent()
        }

    }


    Op("sruCell"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "SRUCell"
        useConfig(baseName)
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "cLast") { description = "Previous cell state, with shape [batchSize, inSize]" }
        Input(NUMERIC, "weights") { description = "The cell's weights." }

        Output(NUMERIC, "output"){ description = "The cell's outputs." }

        Doc(Language.ANY, DocScope.ALL){
            """
             The SRU layer.  Does a single time step operation.
            """.trimIndent()
        }

    }


    Op("sruCell"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "SRUCell"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "initialC") { description = "Initial cell state, with shape [batchSize, inSize]" }
        Input(NUMERIC, "weights") { description = "The cell's weights." }

        Output(NUMERIC, "output"){ description = "The cell's outputs.." }

        Doc(Language.ANY, DocScope.ALL){
            """
             The SRU layer.  Does a single time step operation.
            """.trimIndent()
        }

    }

    Op("sruCell"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "SRUCell"
        useConfig(baseName)
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "initialC") { description = "Initial cell state, with shape [batchSize, inSize]" }
        Input(NUMERIC, "weights") { description = "The cell's weights." }

        Output(NUMERIC, "output"){ description = "The cell's outputs.." }

        Doc(Language.ANY, DocScope.ALL){
            """
             The SRU layer.  Does a single time step operation.
            """.trimIndent()
        }

    }

    Op("sruCell"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "SRUCell"
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "initialC") { description = "Initial cell state, with shape [batchSize, inSize]" }
        Input(NUMERIC, "mask") { description = "An optional dropout mask, with shape [batchSize, inSize]" }

        Input(NUMERIC, "weights") { description = "The cell's weights." }

        Output(NUMERIC, "output"){ description = "The cell's outputs.." }

        Doc(Language.ANY, DocScope.ALL){
            """
             The SRU layer.  Does a single time step operation.
            """.trimIndent()
        }

    }


    Op("sruCell"){
        javaPackage = "org.nd4j.linalg.api.ops.impl.layers.recurrent"
        javaOpClass = "SRUCell"
        useConfig(baseName)
        Input(NUMERIC, "x") { description = "Input, with shape [batchSize, inSize]" }
        Input(NUMERIC, "initialC") { description = "Initial cell state, with shape [batchSize, inSize]" }
        Input(NUMERIC, "mask") { description = "An optional dropout mask, with shape [batchSize, inSize]" }

        Input(NUMERIC, "weights") { description = "The cell's weights." }

        Output(NUMERIC, "output"){ description = "The cell's outputs." }

        Doc(Language.ANY, DocScope.ALL){
            """
             The SRU layer.  Does a single time step operation.
            """.trimIndent()
        }

    }
}


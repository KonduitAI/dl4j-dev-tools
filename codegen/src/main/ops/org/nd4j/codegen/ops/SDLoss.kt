/**
 * Generated using ExtractFromExisting.kt
 */

package org.nd4j.codegen.ops

import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*

fun SDLoss() =  Namespace("SDLoss"){
    // val namespaceJavaPackage = "TODO"
    Op("absoluteDifference") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.loss"
        javaOpClass = "AbsoluteDifferenceLoss"
        legacy = true
        Input(NUMERIC, "label") { description = "Label array" }
        Input(NUMERIC, "predictions") { description = "Predictions array" }
        Input(NUMERIC, "weights") { description = "Weights array. May be null. If null, a weight of 1.0 is used" }
        Arg(LOSS_REDUCE, "lossReduce") { description = "Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}"}
        Output(NUMERIC, "output"){ description = "loss variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute difference loss: {@code sum_i abs( label[i] - predictions[i] )
            """.trimIndent()
        }
    }
/*
    Op("absoluteDifference") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "weights") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Absolute difference loss: {@code sum_i abs( label[i] - predictions[i] )

 @param name        Name of the operation
 @param label       Label array
 @param predictions Predictions array
 @param weights     Weights array. May be null. If null, a weight of 1.0 is used
 @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
 @return Loss variable
     
""".trimIndent()
        }
    }

    Op("absoluteDifference") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #absoluteDifference(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     
""".trimIndent()
        }
    }

    Op("cosineDistance") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "dimension") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #cosineDistance(String, SDVariable, SDVariable, SDVariable, LossReduce, int)}.
     
""".trimIndent()
        }
    }

    Op("cosineDistance") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "weights") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }
        Input(NUMERIC, "dimension") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Cosine distance loss: {@code 1 - cosineSimilarity(x,y)} or {@code 1 - sum_i label[i] * prediction[i]}, which is
 equivalent to cosine distance when both the predictions and labels are normalized.<br>
 <b>Note</b>: This loss function assumes that both the predictions and labels are normalized to have unit l2 norm.
 If this is not the case, you should normalize them first by dividing by {@link SameDiff#norm2(String, SDVariable, boolean, int...)}
 along the cosine distance dimension (with keepDims=true).

 @param name        Name of the operation
 @param label       Label array
 @param predictions Predictions array
 @param weights     Weights array. May be null. If null, a weight of 1.0 is used
 @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
 @param dimension   Dimension to perform the cosine distance over
 @return Cosine distance loss variable
     
""".trimIndent()
        }
    }

    Op("cosineDistance") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }
        Input(NUMERIC, "dimension") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #cosineDistance(String, SDVariable, SDVariable, SDVariable, LossReduce, int)}.
     
""".trimIndent()
        }
    }

    Op("hingeLoss") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #hingeLoss(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     
""".trimIndent()
        }
    }

    Op("hingeLoss") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "weights") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Hinge loss: a loss function used for training classifiers.
 Implements {@code L = max(0, 1 - t * predictions)} where t is the label values after internally converting to {-1,1}
 from the user specified {0,1}. Note that Labels should be provided with values {0,1}.

 @param name        Name of the operation
 @param label       Label array. Each value should be 0.0 or 1.0 (internally -1 to 1 is used)
 @param predictions Predictions array
 @param weights     Weights array. May be null. If null, a weight of 1.0 is used
 @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
 @return Loss variable
     
""".trimIndent()
        }
    }

    Op("hingeLoss") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #hingeLoss(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     
""".trimIndent()
        }
    }

    Op("huberLoss") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "delta") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #huberLoss(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     
""".trimIndent()
        }
    }

    Op("huberLoss") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "weights") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }
        Input(NUMERIC, "delta") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Huber loss function, used for robust regression. It is similar both squared error loss and absolute difference loss,
 though is less sensitive to outliers than squared error.<br>
 Huber loss implements:
 <pre>
 {@code L = 0.5 * (label[i] - predictions[i])^2 if abs(label[i] - predictions[i]) < delta
  L = delta * abs(label[i] - predictions[i]) - 0.5 * delta^2 otherwise
     }
 </pre>

 @param name        Name of the operation
 @param label       Label array
 @param predictions Predictions array
 @param weights     Weights array. May be null. If null, a weight of 1.0 is used
 @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
 @param delta       Loss function delta value
 @return Huber loss variable
     
""".trimIndent()
        }
    }

    Op("huberLoss") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }
        Input(NUMERIC, "delta") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #huberLoss(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     
""".trimIndent()
        }
    }

    Op("l2Loss") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "var") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 L2 loss: 1/2 * sum(x^2)

 @param name Name of the output variable
 @param var  Variable to calculate L2 loss of
 @return L2 loss
     
""".trimIndent()
        }
    }

    Op("logLoss") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #logLoss(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     
""".trimIndent()
        }
    }

    Op("logLoss") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "weights") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }
        Input(NUMERIC, "epsilon") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Log loss, i.e., binary cross entropy loss, usually used for binary multi-label classification. Implements:
 {@code -1/numExamples * sum_i (labels[i] * log(predictions[i] + epsilon) + (1-labels[i]) * log(1-predictions[i] + epsilon))}

 @param name        Name of the operation
 @param label       Label array
 @param predictions Predictions array
 @param weights     Weights array. May be null. If null, a weight of 1.0 is used
 @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
 @return Log loss variable
     
""".trimIndent()
        }
    }

    Op("logLoss") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #logLoss(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     
""".trimIndent()
        }
    }

    Op("logPoisson") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #logPoisson(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     
""".trimIndent()
        }
    }

    Op("logPoisson") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "weights") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Log poisson loss: a loss function used for training classifiers.
 Implements {@code L = exp(c) - z * c} where c is log(predictions) and z is labels.

 @param name        Name of the operation
 @param label       Label array. Each value should be 0.0 or 1.0
 @param predictions Predictions array (has to be log(x) of actual predictions)
 @param weights     Weights array. May be null. If null, a weight of 1.0 is used
 @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
 @return Loss variable
     
""".trimIndent()
        }
    }

    Op("logPoisson") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #logPoisson(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     
""".trimIndent()
        }
    }

    Op("logPoissonFull") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #logPoissonFull(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     
""".trimIndent()
        }
    }

    Op("logPoissonFull") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "weights") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Log poisson loss: a loss function used for training classifiers.
 Implements {@code L = exp(c) - z * c + z * log(z) - z + 0.5 * log(2 * pi * z)}
 where c is log(predictions) and z is labels.

 @param name        Name of the operation
 @param label       Label array. Each value should be 0.0 or 1.0
 @param predictions Predictions array (has to be log(x) of actual predictions)
 @param weights     Weights array. May be null. If null, a weight of 1.0 is used
 @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
 @return Loss variable
     
""".trimIndent()
        }
    }

    Op("logPoissonFull") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #logPoissonFull(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     
""".trimIndent()
        }
    }

    Op("meanPairwiseSquaredError") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #meanPairwiseSquaredError(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     
""".trimIndent()
        }
    }

    Op("meanPairwiseSquaredError") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "weights") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Mean pairwise squared error.<br>
 MPWSE loss calculates the difference between pairs of consecutive elements in the predictions and labels arrays.
 For example, if predictions = [p0, p1, p2] and labels are [l0, l1, l2] then MPWSE is:
 {@code [((p0-p1) - (l0-l1))^2 + ((p0-p2) - (l0-l2))^2 + ((p1-p2) - (l1-l2))^2] / 3}<br>

 @param name        Name of the operation
 @param label       Label array
 @param predictions Predictions array
 @param weights     Weights array. May be null. If null, a weight of 1.0 is used. Must be either null, scalar, or have shape [batchSize]
 @return Loss variable, scalar output
     
""".trimIndent()
        }
    }

    Op("meanSquaredError") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #meanSquaredError(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     
""".trimIndent()
        }
    }

    Op("meanSquaredError") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "weights") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Mean squared error loss function. Implements {@code (label[i] - prediction[i])^2} - i.e., squared error on a per-element basis.
 When averaged (using {@link LossReduce#MEAN_BY_WEIGHT} or {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT} (the default))
 this is the mean squared error loss function.

 @param name        Name of the operation
 @param label       Label array
 @param predictions Predictions array
 @param weights     Weights array. May be null. If null, a weight of 1.0 is used
 @param lossReduce  Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
 @return Loss variable
     
""".trimIndent()
        }
    }

    Op("meanSquaredError") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #meanSquaredError(String, SDVariable, SDVariable, SDVariable, LossReduce)}.
     
""".trimIndent()
        }
    }

    Op("sigmoidCrossEntropy") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #sigmoidCrossEntropy(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     
""".trimIndent()
        }
    }

    Op("sigmoidCrossEntropy") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictionLogits") { description = "" }
        Input(NUMERIC, "weights") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }
        Input(NUMERIC, "labelSmoothing") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Sigmoid cross entropy: applies the sigmoid activation function on the input logits (input "pre-sigmoid preductions")
 and implements the binary cross entropy loss function. This implementation is numerically more stable than using
 standard (but separate) sigmoid activation function and log loss (binary cross entropy) loss function.<br>
 Implements:
 {@code -1/numExamples * sum_i (labels[i] * log(sigmoid(logits[i])) + (1-labels[i]) * log(1-sigmoid(logits[i])))}
 though this is done in a mathematically equivalent but more numerical stable form.<br>
 <br>
 When label smoothing is > 0, the following label smoothing is used:<br>
 <pre>
 {@code numClasses = labels.size(1);
 label = (1.0 - labelSmoothing) * label + 0.5 * labelSmoothing}
 </pre>

 @param name             Name of the operation
 @param label            Label array
 @param predictionLogits Predictions array
 @param weights          Weights array. May be null. If null, a weight of 1.0 is used
 @param lossReduce       Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
 @return Loss variable
     
""".trimIndent()
        }
    }

    Op("sigmoidCrossEntropy") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #sigmoidCrossEntropy(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     
""".trimIndent()
        }
    }

    Op("softmaxCrossEntropy") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #softmaxCrossEntropy(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     
""".trimIndent()
        }
    }

    Op("softmaxCrossEntropy") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "oneHotLabels") { description = "" }
        Input(NUMERIC, "logitPredictions") { description = "" }
        Input(NUMERIC, "weights") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }
        Input(NUMERIC, "labelSmoothing") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Applies the softmax activation function to the input, then implement multi-class cross entropy:<br>
 {@code -sum_classes label[i] * log(p[c])} where {@code p = softmax(logits)}<br>
 If {@link LossReduce#NONE} is used, returned shape is [numExamples] out for [numExamples, numClasses] predicitons/labels;
 otherwise, the output is a scalar.<br>
 <p>
 When label smoothing is > 0, the following label smoothing is used:<br>
 <pre>
 {@code numClasses = labels.size(1);
 oneHotLabel = (1.0 - labelSmoothing) * oneHotLabels + labelSmoothing/numClasses}
 </pre>

 @param name             Name of the operation
 @param oneHotLabels     Label array. Should be one-hot per example and same shape as predictions (for example, [mb, nOut])
 @param logitPredictions Predictions array (pre-softmax)
 @param weights          Weights array. May be null. If null, a weight of 1.0 is used
 @param lossReduce       Reduction type for the loss. See {@link LossReduce} for more details. Default: {@link LossReduce#MEAN_BY_NONZERO_WEIGHT_COUNT}
 @param labelSmoothing   Label smoothing value. Default value: 0
 @return Loss variable
     
""".trimIndent()
        }
    }

    Op("softmaxCrossEntropy") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "label") { description = "" }
        Input(NUMERIC, "predictions") { description = "" }
        Input(NUMERIC, "lossReduce") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 See {@link #softmaxCrossEntropy(String, SDVariable, SDVariable, SDVariable, LossReduce, double)}.
     
""".trimIndent()
        }
    }

    Op("sparseSoftmaxCrossEntropy") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "logits") { description = "" }
        Input(NUMERIC, "labels") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 As per {@link #softmaxCrossEntropy(String, SDVariable, SDVariable, LossReduce)} but the labels variable
 is represented as an integer array instead of the equivalent one-hot array.<br>
 i.e., if logits are rank N, then labels have rank N-1

 @param name   Name of the output variable. May be null
 @param logits Logits array ("pre-softmax activations")
 @param labels Labels array. Must be an integer type.
 @return Softmax cross entropy
     
""".trimIndent()
        }
    }

    Op("weightedCrossEntropyWithLogits") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "targets") { description = "" }
        Input(NUMERIC, "inputs") { description = "" }
        Input(NUMERIC, "weights") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 TODO

 @param name
 @param targets
 @param inputs
 @param weights
 @return
     
""".trimIndent()
        }
    }
    */
}

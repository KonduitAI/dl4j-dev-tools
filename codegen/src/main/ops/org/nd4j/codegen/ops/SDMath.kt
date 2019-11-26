/**
 * Generated using ExtractFromExisting.kt
 */
package org.nd4j.codegen.ops

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*

fun SDMath() =  Namespace("SDMath"){
    val namespaceJavaPackage = "" //Seem to be different for each funtion here.
    Op("abs") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                 Elementwise absolute value operation: out = abs(x)
            """.trimIndent()
        }
    }

    Op("acos") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                 Elementwise acos (arccosine, inverse cosine) operation: out = arccos(x)
            """.trimIndent()
        }
    }

    Op("acosh") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise acosh (inverse hyperbolic cosine) function: out = acosh(x)
            """.trimIndent()
        }
    }

    // TODO: There are 2 implementations of amax in org.nd4j.linalg.api.ops.impl
    Op("amax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.same" // or "org.nd4j.linalg.api.ops.impl.transforms.same"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))
            """.trimIndent()
        }
    }

    Op("amean") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
            """.trimIndent()
        }
    }

    // TODO: There are 2 implementations of amax in org.nd4j.linalg.api.ops.impl
    Op("amin") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"  // ""org.nd4j.linalg.api.ops.impl.reduce.same"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute min array reduction operation, optionally along specified dimensions: out = min(abs(x))
            """.trimIndent()
        }
    }

    Op("and") {
        javaPackage = "org.nd4j.linalg.indexing.conditions"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "y") { description = "Input 2" }
        Output(NUMERIC, "output"){ description = "SDVariable with values 0 and 1 based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                 Boolean AND operation: elementwise (x != 0) && (y != 0)<br>
                 If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
                 Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
                 Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
            """.trimIndent()
        }
    }

    Op("asin") {
        javaPackage = " org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise asin (arcsin, inverse sine) operation: out = arcsin(x)
            """.trimIndent()
        }
    }

    // TODO: There are 2 implementations
    Op("asinh") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict" // org.nd4j.linalg.api.ops.impl.transforms.strict
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise asinh (inverse hyperbolic sine) function: out = asinh(x)
            """.trimIndent()
        }
    }

    Op("asum") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.same"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute sum array reduction operation, optionally along specified dimensions: out = sum(abs(x))
            """.trimIndent()
        }
    }

    Op("atan") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise atan (arctangent, inverse tangent) operation: out = arctangent(x)
            """.trimIndent()
        }
    }

    Op("atan2") {
        javaPackage = "rg.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "y") { description = "Input Y variable" }
        Input(NUMERIC, "x") { description = "Input X variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise atan (arctangent, inverse tangent) operation: out = atan2(x,y).
                Similar to atan(y/x) but sigts of x and y are used to determine the location of the result
            """.trimIndent()
        }
    }

    Op("atanh") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise atanh (inverse hyperbolic tangent) function: out = atanh(x)
            """.trimIndent()
        }
    }

    Op("ceil") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise ceiling function: out = ceil(x).
                Rounds each value up to the nearest integer value (if not already an integer)
            """.trimIndent()
        }
    }

    Op("clipByNorm") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.clip"
        Input(NUMERIC, "x") { description = "Input variable" }
        Input(NUMERIC, "clipValue") { description = "Clipping value (maximum l2 norm)" }
        Output(NUMERIC, "output"){ description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Clipping by L2 norm<br>
                if l2Norm(x) < clipValue, then input is returned unmodifed<br>
                Otherwise, out[i] = in[i] * clipValue / l2Norm(in)
            """.trimIndent()
        }
    }

    Op("clipByNorm") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.clip"
        Input(NUMERIC, "x") { description = "Input variable" }
        Input(NUMERIC, "clipValue") { description = "Clipping value (maximum l2 norm)" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Clipping by L2 norm, optionally along dimension(s)<br>
                if l2Norm(x,dimension) < clipValue, then input is returned unmodifed<br>
                Otherwise, out[i] = in[i] * clipValue / l2Norm(in, dimensions) where each value is clipped according
                to the corresponding l2Norm along the specified dimensions
            """.trimIndent()
        }
    }

    Op("clipByValue") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.clip"
        Input(NUMERIC, "x") { description = "Input variable" }
        Input(NUMERIC, "clipValueMin") { description = "Minimum value for clipping" }
        Input(NUMERIC, "clipValueMax") { description = "Maximum value for clipping" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise clipping function:<br>
                out[i] = in[i] if in[i] >= clipValueMin and in[i] <= clipValueMax<br>
                out[i] = clipValueMin if in[i] < clipValueMin<br>
                out[i] = clipValueMax if in[i] > clipValueMax<br>
            """.trimIndent()
        }
    }

    Op("confusionMatrix") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "labels") { description = "Labels - 1D array of integer values representing label values" }
        Input(NUMERIC, "pred") { description = "Predictions - 1D array of integer values representing predictions. Same length as labels" }
        Input(INT, "dataType") { description = "Data type" } //TODO: Mapped DataType to INT.

        Output(NUMERIC, "output"){ description = "variable (2D, shape [numClasses, numClasses})" }

        Doc(Language.ANY, DocScope.ALL){
            """ 
                Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
                which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))<br>
                For example, if labels = [0, 1, 1] and predicted = [0, 2, 1] then output is:<br>
                [1, 0, 0]<br>
                [0, 1, 1]<br>
                [0, 0, 0]<br>
            """.trimIndent()
        }
    }

    Op("confusionMatrix") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "labels") { description = "Labels - 1D array of integer values representing label values" }
        Input(NUMERIC, "pred") { description = "Predictions - 1D array of integer values representing predictions. Same length as labels" }
        Input(INT, "numClasses") { description = "Number of classes" }
        Output(NUMERIC, "output"){ description = "variable (2D, shape [numClasses, numClasses})" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
                which are represented as integer values.<br>
                For example, if labels = [0, 1, 1], predicted = [0, 2, 1], and numClasses=4 then output is:<br>
                [1, 0, 0, 0]<br>
                [0, 1, 1, 0]<br>
                [0, 0, 0, 0]<br>
                [0, 0, 0, 0]<br>
            """.trimIndent()
        }
    }

    Op("confusionMatrix") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "labels") { description = "Labels - 1D array of integer values representing label values" }
        Input(NUMERIC, "pred") { description = "Predictions - 1D array of integer values representing predictions. Same length as labels" }
        Input(NUMERIC, "weights") { description = "Weights - 1D array of values (may be real/decimal) representing the weight/contribution of each prediction. Must be same length as both labels and predictions arrays" }
        Output(NUMERIC, "output"){ description = "variable (2D, shape [numClasses, numClasses})" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
                which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))<br>
                For example, if labels = [0, 1, 1], predicted = [0, 2, 1] and weights = [1, 2, 3]
                [1, 0, 0]<br>
                [0, 3, 2]<br>
                [0, 0, 0]<br>
            """.trimIndent()
        }
    }

    Op("confusionMatrix") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "labels") { description = "Labels - 1D array of integer values representing label values" }
        Input(NUMERIC, "pred") { description = "Predictions - 1D array of integer values representing predictions. Same length as labels" }
        Input(INT, "numClasses") { description = "" }
        Input(NUMERIC, "weights") { description = "Weights - 1D array of values (may be real/decimal) representing the weight/contribution of each prediction. Must be same length as both labels and predictions arrays" }
        Output(NUMERIC, "output"){ description = "Output variable (2D, shape [numClasses, numClasses})" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Compute the 2d confusion matrix of size [numClasses, numClasses] from a pair of labels and predictions, both of
                which are represented as integer values.<br>
                For example, if labels = [0, 1, 1], predicted = [0, 2, 1], numClasses = 4, and weights = [1, 2, 3]
                [1, 0, 0, 0]<br>
                [0, 3, 2, 0]<br>
                [0, 0, 0, 0]<br>
                [0, 0, 0, 0]<br>
            """.trimIndent()
        }
    }

    Op("cos") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise cosine operation: out = cos(x)
            """.trimIndent()
        }
    }

    Op("cosh") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise cosh (hyperbolic cosine) operation: out = cosh(x)
            """.trimIndent()
        }
    }

    Op("cosineDistance") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce3"
        Input(NUMERIC, "x") { description = "Input variable x" }
        Input(NUMERIC, "y") { description = "Input variable y" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to calculate cosine similarity over" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Cosine distance reduction operation. The output contains the cosine distance for each
                tensor/subset along the specified dimensions:<br>
                out = 1.0 - cosineSimilarity(x,y)<br>
                See {@link #cosineSimilarity(String, SDVariable, SDVariable, int...)}
            """.trimIndent()
        }
    }

    Op("cosineSimilarity") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce3"
        Input(NUMERIC, "x") { description = "Input variable x" }
        Input(NUMERIC, "y") { description = "Input variable y" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to calculate cosine similarity over" }
        Output(NUMERIC, "output"){ description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL){
            """ 
                Cosine similarity pairwise reduction operation. The output contains the cosine similarity for each tensor/subset
                along the specified dimensions:<br>
                out = (sum_i x[i] * y[i]) / ( sqrt(sum_i x[i]^2) * sqrt(sum_i y[i]^2)
            """.trimIndent()
        }
    }

    Op("countNonZero") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.longer"
        Input(NUMERIC, "input") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Count non zero array reduction operation, optionally along specified dimensions: out = count(x != 0)
            """.trimIndent()
        }
    }

    Op("countZero") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.longer"
        Input(NUMERIC, "input") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Count zero array reduction operation, optionally along specified dimensions: out = count(x == 0)
            """.trimIndent()
        }
    }

    Op("cross") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "a") { description = "First input" }
        Input(NUMERIC, "b") { description = "Second input" }
        Output(NUMERIC, "output"){ description = "Element-wise cross product" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Returns the pair-wise cross product of equal size arrays a and b: a x b = ||a||x||b|| sin(theta).
                Can take rank 1 or above inputs (of equal shapes), but note that the last dimension must have dimension 3
            """.trimIndent()
        }
    }

    Op("cube") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise cube function: out = x^3
            """.trimIndent()
        }
    }

    Op("diag") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Returns an output variable with diagonal values equal to the specified values; off-diagonal values will be set to 0<br>
                For example, if input = [1,2,3], then output is given by:<br>
                [ 1, 0, 0]<br>
                [ 0, 2, 0]<br>
                [ 0, 0, 3]<br>
                <br>
                Higher input ranks are also supported: if input has shape [a,...,R-1] then output[i,...,k,i,...,k] = input[i,...,k].
                i.e., for input rank R, output has rank 2R
                """.trimIndent()
        }
    }

    Op("diagPart") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Diagonal part of the input" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Extract the diagonal part from the input array.<br>
                If input is<br>
                [ 1, 0, 0]<br>
                [ 0, 2, 0]<br>
                [ 0, 0, 3]<br>
                then output is [1, 2, 3].<br>
                Supports higher dimensions: in general, out[i,...,k] = in[i,...,k,i,...,k]
                """.trimIndent()
        }
    }

    Op("entropy") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to reduce on (null/empty for full array)" }
        Output(NUMERIC, "output"){ description = "Output variable: reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Entropy reduction: -sum(x * log(x))
            """.trimIndent()
        }
    }

    Op("erf") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = " Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable name" }
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Element-wise Gaussian error function - out = erf(in)
            """.trimIndent()
        }
    }

    Op("erfc") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise complementary Gaussian error function - out = erfc(in) = 1 - erf(in)
            """.trimIndent()
        }
    }

    Op("euclideanDistance") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce3"
        Input(NUMERIC, "x") { description = "Input variable x" }
        Input(NUMERIC, "y") { description = "Input variable y" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to calculate cosine similarity over" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Euclidean distance (l2 norm, l2 distance) reduction operation. The output contains the Euclidean distance for each
                tensor/subset along the specified dimensions:<br>
                out = sqrt( sum_i (x[i] - y[i])^2 )
                """.trimIndent()
        }
    }

    Op("exp") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise exponent function: out = exp(x) = 2.71828...^x
            """.trimIndent()
        }
    }

    Op("expm1") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise 1.0 - exponent function: out = 1.0 - exp(x) = 1.0 - 2.71828...^x
            """.trimIndent()
        }
    }

    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "rows") { description = "Number of rows" }
        Output(NUMERIC, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Generate an identity matrix with the specified number of rows and columns.
            """.trimIndent()
        }
    }

    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "rows") { description = "Number of rows" }
        Input(NUMERIC, "cols") { description = "Number of columns" }
        Output(NUMERIC, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
                As per {@link #eye(String, int, int, DataType)} but with the default datatype, {@link Eye#DEFAULT_DTYPE}
            """.trimIndent()
        }
    }

    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "rows") { description = "Number of rows" }
        Input(NUMERIC, "cols") { description = "Number of columns" }
        Input(INT, "dataType") { description = "Data type" } //TODO: Mapped DataType to INT.
        Output(NUMERIC, "output"){ description = "SDVaribable identity matrix" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Generate an identity matrix with the specified number of rows and columns
                Example:<br>
                <pre>
                {@code SDVariable eye = eye(3,2)
                eye:
                [ 1, 0]
                [ 0, 1]
                [ 0, 0]}
                </pre>
                """.trimIndent()
        }
    }

    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "rows") { description = "Number of rows" }
        Input(NUMERIC, "cols") { description = "Number of columns" }
        Input(INT, "dataType") { description = "Data type" } //TODO: Mapped DataType to INT.
        Arg(INT, "batchDimension"){ count = AtLeast(0); description = "Batch dimensions. May be null" }
        Output(NUMERIC, "output"){ description = "SDVaribable identity matrix" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Generate an identity matrix with the specified number of rows and columns, with optional leading dims<br>
                Example:<br>
                batchShape: [3,3]<br>
                numRows: 2<br>
                numCols: 4<br>
                returns a tensor of shape (3, 3, 2, 4) that consists of 3 * 3 batches of (2,4)-shaped identity matrices:<br>
                1 0 0 0<br>
                0 1 0 0<br>
            """.trimIndent()
        }
    }

    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "rows") { description = "Number of rows" }
        Input(NUMERIC, "cols") { description = "Number of columns" }
        Arg(INT, "batchDimension"){ count = AtLeast(0); description = "Batch dimensions. May be null" }
        Output(NUMERIC, "output"){ description = "SDVaribable identity matrix" }
        Doc(Language.ANY, DocScope.ALL){
            """
                As per {@link #eye(String, int, int, int...)} bit with the number of rows/columns specified as scalar SDVariables,
                and the batch dimension specified as a 1D SDVariable
            """.trimIndent()
        }
    }

    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "rows") { description = "Number of rows" }
        Input(NUMERIC, "cols") { description = "Number of columns" }
        Output(NUMERIC, "output"){ description = "SDVaribable identity matrix" }
        Doc(Language.ANY, DocScope.ALL){
            """
                As per {@link #eye(String, int, int)} bit with the number of rows/columns specified as scalar SDVariables
            """.trimIndent()
        }
    }

    Op("eye") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Input(NUMERIC, "rows") { description = "Number of rows" }
        Output(NUMERIC, "output"){ description = "SDVaribable identity matrix" }
        Doc(Language.ANY, DocScope.ALL){
            """
                As per {@link #eye(String, int)} but with the number of rows specified as a scalar SDVariable
            """.trimIndent()
        }
    }

    Op("firstIndex") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
        Input(NUMERIC, "in") { description = "Input variable" }
        Input(NUMERIC, "condition") { description = "Condition to check on input variable" } //TODO: How to map the "Condition" object.
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                First index reduction operation.<br>
                Returns a variable that contains the index of the first element that matches the specified condition (for each
                slice along the specified dimensions)
            """.trimIndent()
        }
    }

    Op("firstIndex") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
        Input(NUMERIC, "in") { description = "Input variable" }
        Input(NUMERIC, "condition") { description = "Condition to check on input variable" } //TODO: How to map the "Condition" object.
        Input(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                First index reduction operation.<br>
                Returns a variable that contains the index of the first element that matches the specified condition (for each
                slice along the specified dimensions)<br>
                Note that if keepDims = true, the output variable has the same rank as the input variable,
                with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
                the mean along a dimension).<br>
                Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
                keepDims = true: [a,1,c]<br>
                keepDims = false: [a,c]
            """.trimIndent()
        }
    }

    Op("floor") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise floor function: out = floor(x).
                Rounds each value down to the nearest integer value (if not already an integer)
            """.trimIndent()
        }
    }

    Op("hammingDistance") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce3"
        Input(NUMERIC, "x") { description = "Input variable x" }
        Input(NUMERIC, "y") { description = "Input variable y" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to calculate cosine similarity over" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Hamming distance reduction operation. The output contains the cosine distance for each
                tensor/subset along the specified dimensions:<br>
                out = count( x[i] != y[i] )
            """.trimIndent()
        }
    }

    Op("iamax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Index of the max absolute value: argmax(abs(in))
                @see SameDiff#argmax(String, SDVariable, boolean, int...)
            """.trimIndent()
        }
    }

    Op("iamax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
        Input(NUMERIC, "in") { description = "Input variable" }
        Input(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Index of the max absolute value: argmax(abs(in))
                @see SameDiff#argmax(String, SDVariable, boolean, int...)
            """.trimIndent()
        }
    }

    Op("iamin") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Index of the min absolute value: argmin(abs(in))
                @see SameDiff#argmin(String, SDVariable, boolean, int...)
            """.trimIndent()
        }
    }

    Op("iamin") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
        Input(NUMERIC, "in") { description = "Input variable" }
        Input(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Index of the min absolute value: argmin(abs(in))
                @see SameDiff#argmin(String, SDVariable, boolean, int...)
            """.trimIndent()
        }
    }

    Op("isFinite") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.bool"
        Input(NUMERIC, "x") { description = "Input array" }
        Output(NUMERIC, "output"){ description = "SDVariable with values 0 and 1 based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Is finite operation: elementwise isFinite(x)<br>
                Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
                value 0 otherwise
            """.trimIndent()
        }
    }

    Op("isInfinite") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.bool"
        Input(NUMERIC, "x") { description = "Input array" }
        Output(NUMERIC, "output"){ description = "SDVariable with values 0 and 1 based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Is infinite operation: elementwise isInfinite(x)<br>
                Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
                value 0 otherwise
            """.trimIndent()
        }
    }

    Op("isMax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.any"
        Input(NUMERIC, "x") { description = "Input array" }
        Output(NUMERIC, "output"){ description = "SDVariable with values 0 and 1 based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Is maximum operation: elementwise x == max(x)<br>
                Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
                value 0 otherwise
            """.trimIndent()
        }
    }

    Op("isNaN") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.bool"
        Input(NUMERIC, "x") { description = "Input array" }
        Output(NUMERIC, "output"){ description = "SDVariable with values 0 and 1 based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Is Not a Number operation: elementwise isNaN(x)<br>
                Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
                value 0 otherwise
            """.trimIndent()
        }
    }

    Op("isNonDecreasing") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Scalar variable with value 1 if non-decreasing, or 0 otherwise" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Is the array non decreasing?<br>
                An array is non-decreasing if for every valid i, x[i] <= x[i+1]. For Rank 2+ arrays, values are compared
                in 'c' (row major) order
            """.trimIndent()
        }
    }

    Op("isStrictlyIncreasing") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Scalar variable with value 1 if strictly increasing, or 0 otherwise" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Is the array strictly increasing?<br>
                An array is strictly increasing if for every valid i, x[i] < x[i+1]. For Rank 2+ arrays, values are compared
                in 'c' (row major) order
            """.trimIndent()
        }
    }

    Op("jaccardDistance") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce3"
        Input(NUMERIC, "x") { description = "Input variable x" }
        Input(NUMERIC, "y") { description = "Input variable y" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to calculate Jaccard similarity over" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """Jaccard similarity reduction operation. The output contains the Jaccard distance for each
                tensor along the specified dimensions.
            """.trimIndent()
        }
    }

    Op("lastIndex") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
        Input(NUMERIC, "in") { description = "Input variable" }
        Input(NUMERIC, "condition") { description = "Condition to check on input variable" } //TODO: How to map the "Condition" object.
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Last index reduction operation.<br>
                Returns a variable that contains the index of the last element that matches the specified condition (for each
                slice along the specified dimensions)
            """.trimIndent()
        }
    }

    Op("lastIndex") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
        Input(NUMERIC, "in") { description = "Input variable" }
        Input(NUMERIC, "condition") { description = "Condition to check on input variable" } //TODO: How to map the "Condition" object.
        Input(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Last index reduction operation.<br>
                Returns a variable that contains the index of the last element that matches the specified condition (for each
                slice along the specified dimensions)<br>
                Note that if keepDims = true, the output variable has the same rank as the input variable,
                with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
                the mean along a dimension).<br>
                Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
                keepDims = true: [a,1,c]<br>
                keepDims = false: [a,c]
            """.trimIndent()
        }
    }

    Op("log") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise logarithm function (base e - natural logarithm): out = log(x)
            """.trimIndent()
        }
    }

    Op("log") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "in") { description = "Input variable" }
        Input(NUMERIC, "base") { description = "Logarithm base" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise logarithm function (with specified base): out = log_{base}(x)
            """.trimIndent()
        }
    }

    Op("log1p") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Elementwise natural logarithm function: out = log_e (1 + x)

 @param name Output variable name
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("logEntropy") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Log entropy reduction: log(-sum(x * log(x)))

 @param name       Name of the output variable
 @param in         Input variable
 @param dimensions Dimensions to reduce on (null for full array)
 @return Output variable: reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("logSumExp") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Log-sum-exp reduction (optionally along dimension).
 Computes log(sum(exp(x))

 @param name       Name of the output variable
 @param input      Input variable
 @param dimensions Optional dimensions to reduce along
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("manhattanDistance") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Manhattan distance (l1 norm, l1 distance) reduction operation. The output contains the Manhattan distance for each
 tensor/subset along the specified dimensions:<br>
 out = sum_i abs(x[i]-y[i])

 @param name       Name of the output variable
 @param x          Input variable x
 @param y          Input variable y
 @param dimensions Dimensions to calculate cosine similarity over
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("matrixDeterminant") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Matrix determinant op. For 2D input, this returns the standard matrix determinant.
 For higher dimensional input with shape [..., m, m] the matrix determinant is returned for each
 shape [m,m] sub-matrix.

 @param name Name of the output variable
 @param in   Input
 @return Matrix determinant variable
     
""".trimIndent()
        }
    }

    Op("matrixInverse") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Matrix inverse op. For 2D input, this returns the standard matrix inverse.
 For higher dimensional input with shape [..., m, m] the matrix inverse is returned for each
 shape [m,m] sub-matrix.

 @param name Name of the output variable
 @param in   Input
 @return Matrix inverse variable
     
""".trimIndent()
        }
    }

    Op("mergeAdd") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "inputs") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Merge add function: merges an arbitrary number of equal shaped arrays using element-wise addition:
 out = sum_i in[i]

 @param name   Name of the output variable
 @param inputs Input variables
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("mergeAvg") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "inputs") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Merge average function: merges an arbitrary number of equal shaped arrays using element-wise mean operation:
 out = mean_i in[i]

 @param name   Name of the output variable
 @param inputs Input variables
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("mergeMax") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "inputs") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Merge max function: merges an arbitrary number of equal shaped arrays using element-wise maximum operation:
 out = max_i in[i]

 @param inputs Input variables
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("moments") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }
        Input(NUMERIC, "axes") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Calculate the mean and (population) variance for the input variable, for the specified axis

 @param name  Name of the output variables. Can be null; if non-null, must be length 2
 @param input Input to calculate moments for
 @param axes  Dimensions to perform calculation over
 @return Mean and variance variables
     
""".trimIndent()
        }
    }

    Op("neg") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Elementwise negative operation: out = -x

 @param name Name of the output variable
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("normalizeMoments") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "counts") { description = "" }
        Input(NUMERIC, "means") { description = "" }
        Input(NUMERIC, "variances") { description = "" }
        Input(NUMERIC, "shift") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Calculate the mean and variance from the sufficient statistics

 @param name      Name of the output variables. Can be null; if non-null, must be length 2
 @param counts    Rank 0 (scalar) value with the total number of values used to calculate the sufficient statistics
 @param means     Mean-value sufficient statistics: this is the SUM of all data values
 @param variances Variaance sufficient statistics: this is the squared sum of all data values
 @param shift     Shift value, possibly 0, used when calculating the sufficient statistics (for numerical stability)
 @return Output variables: mean and population variance
     
""".trimIndent()
        }
    }

    Op("or") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Boolean OR operation: elementwise (x != 0) || (y != 0)<br>
 If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
 Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
 Returns an array with values 1 where condition is satisfied, or value 0 otherwise.

 @param name Name of the output variable
 @param x    Input 1
 @param y    Input 2
 @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     
""".trimIndent()
        }
    }

    Op("pow") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "value") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise power function: out = x^value

 @param name  Output variable name
 @param x     Input variable
 @param value Power to raise each element to
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("pow") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise (broadcastable) power function: out = x[i]^y[i]

 @param name Output variable name
 @param x    Input variable
 @param y    Power
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("reciprocal") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "a") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise reciprocal (inverse) function: out[i] = 1 / in[i]

 @param name Name of the output variable
 @param a    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("round") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise round function: out = round(x).
 Rounds (up or down depending on value) to the nearest integer value.

 @param name Output variable name
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("rsqrt") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise reciprocal (inverse) of square root: out = 1.0 / sqrt(x)

 @param name Output variable name
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("setDiag") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "diag") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Set the diagonal value to the specified values<br>
 If input is<br>
 [ a, b, c]<br>
 [ d, e, f]<br>
 [ g, h, i]<br>
 and diag = [ 1, 2, 3] then output is<br>
 [ 1, b, c]<br>
 [ d, 2, f]<br>
 [ g, h, 3]<br>

 @param name Name of the output variable
 @param in   Input variable
 @param diag Diagonal
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("shannonEntropy") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Shannon Entropy reduction: -sum(x * log2(x))

 @param name       Name of the output variable
 @param in         Input variable
 @param dimensions Dimensions to reduce on (null/empty for full array)
 @return Output variable: reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("sign") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise sign (signum) function:<br>
 out = -1 if in < 0<br>
 out = 0 if in = 0<br>
 out = 1 if in > 0

 @param name Output variable name
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("sin") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Elementwise sine operation: out = sin(x)

 @param name Output variable name
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("sinh") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Elementwise sinh (hyperbolic sine) operation: out = sinh(x)

 @param name Output variable name
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("sqrt") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise square root function: out = sqrt(x)

 @param name Output variable name
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("square") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise square function: out = x^2

 @param name Output variable name
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("step") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "cutoff") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Elementwise step function:<br>
 out(x) = 1 if x >= cutoff<br>
 out(x) = 0 otherwise<br>

 @param name   Name of the output variable
 @param in     Input variable
 @param cutoff Cutoff value for step function
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("standardize") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Standardize input variable along given axis
 <p>
 out = (x - mean) / stdev
 <p>
 with mean and stdev being calculated along the given dimension.

 <p>
 For example: given x as a mini batch of the shape [numExamples, exampleLength]:
 <ul>
  <li>use dimension 1 too use the statistics (mean, stdev) for each example</li>
  <li>use dimension 0 if you want to use the statistics for each column across all examples</li>
  <li>use dimensions 0,1 if you want to use the statistics across all columns and examples</li>
 </ul>

 @param name Name of the output variable
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("tan") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Elementwise tangent operation: out = tan(x)

 @param name Output variable name
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("tanh") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)

 @param name Output variable name
 @param x    Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("trace") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Matrix trace operation
 For rank 2 matrices, the output is a scalar vith the trace - i.e., sum of the main diagonal.<br>
 For higher rank inputs, output[a,b,c] = trace(in[a,b,c,:,:])

 @param name Name of the output variable. May be null.
 @param in   Input variable
 @return Trace
     
""".trimIndent()
        }
    }

    Op("xor") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Boolean XOR (exclusive OR) operation: elementwise (x != 0) XOR (y != 0)<br>
 If x and y arrays have equal shape, the output shape is the same as these inputs.<br>
 Note: supports broadcasting if x and y have different shapes and are broadcastable.<br>
 Returns an array with values 1 where condition is satisfied, or value 0 otherwise.

 @param name Name of the output variable
 @param x    Input 1
 @param y    Input 2
 @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     
""".trimIndent()
        }
    }

    Op("bitShift") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "shift") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Shift integer bits to the left, i.e. var << 4

 @param name Name of the output variable
 @param x    Input 1
 @return Output SDVariable with shifted bits
     
""".trimIndent()
        }
    }

    Op("bitShiftRight") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "shift") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Shift integer bits to the right, i.e. var >> 4

 @param name Name of the output variable
 @param x    Input 1
 @return Output SDVariable with shifted bits
     
""".trimIndent()
        }
    }

    Op("bitRotl") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "shift") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Roll integer bits to the left, i.e. var << 4 | var >> (32 - 4)

 @param name Name of the output variable
 @param x    Input 1
 @return Output SDVariable with shifted bits
     
""".trimIndent()
        }
    }

    Op("bitRotr") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "shift") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Roll integer bits to the right, i.e. var >> 4 | var << (32 - 4)

 @param name Name of the output variable
 @param x    Input 1
 @return Output SDVariable with shifted bits
     
""".trimIndent()
        }
    }

    Op("zeroFraction") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Full array zero fraction array reduction operation, optionally along specified dimensions: out = (count(x == 0) / length(x))

 @param name  Name of the output variable
 @param input Input variable
 @return Reduced array of rank 0 (scalar)
     
""".trimIndent()
        }
    }
}

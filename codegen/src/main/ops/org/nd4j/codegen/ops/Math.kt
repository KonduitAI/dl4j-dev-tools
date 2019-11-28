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

    val transform = Op("transform"){
        isAbstract = true
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
    }

    val transformStrict = Op("transformStrict", transform){
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
    }

    val transformSame = Op("transformSame", transform){
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
    }

    val reduce = Op("reduce"){
        isAbstract = true
        legacy = true
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }
    }

    val reduceFloating = Op("reduceFloating", reduce){
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
    }

    val reduceSame = Op("reduceSame", transform){
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.same"
    }

    Op("abs", transformSame) {
        javaOpClass = "Abs"
        Doc(Language.ANY, DocScope.ALL){
            """
                 Elementwise absolute value operation: out = abs(x)
            """.trimIndent()
        }
    }

    Op("acos", transformStrict) {
        javaOpClass = "ACos"
        Doc(Language.ANY, DocScope.ALL){
            """
                 Elementwise acos (arccosine, inverse cosine) operation: out = arccos(x)
            """.trimIndent()
        }
    }

    Op("acosh", transformStrict) {
        javaOpClass = "ACosh"
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise acosh (inverse hyperbolic cosine) function: out = acosh(x)
            """.trimIndent()
        }
    }

    // TODO should we call these "reduceAMax", "reduceAMean", "reduceMin" etc?
    // TODO: There are 2 implementations of amax in org.nd4j.linalg.api.ops.impl
    Op("amax", reduceSame) {
        javaOpClass = "AMax"
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute max array reduction operation, optionally along specified dimensions: out = max(abs(x))
            """.trimIndent()
        }
    }

    Op("amean", reduceFloating) {
        javaOpClass = "AMean"
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute mean array reduction operation, optionally along specified dimensions: out = mean(abs(x))
            """.trimIndent()
        }
    }

    // TODO: There are 2 implementations of amax in org.nd4j.linalg.api.ops.impl
    Op("amin", reduceSame) {
        javaOpClass = "AMin"
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
                 Boolean AND operation: elementwise (x != 0) && (y != 0)
                 If x and y arrays have equal shape, the output shape is the same as these inputs.
                 Note: supports broadcasting if x and y have different shapes and are broadcastable.
                 Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
            """.trimIndent()
        }
    }

    Op("asin", transformStrict) {
        javaOpClass = "ASin"
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise asin (arcsin, inverse sine) operation: out = arcsin(x)
            """.trimIndent()
        }
    }

    // TODO: There are 2 implementations
    Op("asinh", transformStrict) {
        javaOpClass = "ASinh"
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise asinh (inverse hyperbolic sine) function: out = asinh(x)
            """.trimIndent()
        }
    }

    Op("asum", reduceSame) {
        javaOpClass = "ASum"
        Doc(Language.ANY, DocScope.ALL){
            """
                Absolute sum array reduction operation, optionally along specified dimensions: out = sum(abs(x))
            """.trimIndent()
        }
    }

    Op("atan", transformStrict) {
        javaOpClass = "ATan"
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise atan (arctangent, inverse tangent) operation: out = arctangent(x)
            """.trimIndent()
        }
    }

    Op("atan2") {
        javaPackage = "rg.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "ATan2"
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

    Op("atanh", transformStrict) {
        javaOpClass = "ATanh"
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise atanh (inverse hyperbolic tangent) function: out = atanh(x)
            """.trimIndent()
        }
    }

    Op("ceil", transformSame) {
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
        Arg(INT, "dimensions"){ description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed"; defaultValue = null }
        Output(NUMERIC, "output"){ description = "Output variable" }

        //Signature(x, clipValue)
        Doc(Language.ANY, DocScope.ALL){
            """ 
                Clipping by L2 norm, optionally along dimension(s)
                if l2Norm(x,dimension) < clipValue, then input is returned unmodifed
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
                Element-wise clipping function:
                out[i] = in[i] if in[i] >= clipValueMin and in[i] <= clipValueMax
                out[i] = clipValueMin if in[i] < clipValueMin
                out[i] = clipValueMax if in[i] > clipValueMax
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
                which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))
                For example, if labels = [0, 1, 1] and predicted = [0, 2, 1] then output is:
                [1, 0, 0]
                [0, 1, 1]
                [0, 0, 0]
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
                which are represented as integer values.
                For example, if labels = [0, 1, 1], predicted = [0, 2, 1], and numClasses=4 then output is:
                [1, 0, 0, 0]
                [0, 1, 1, 0]
                [0, 0, 0, 0]
                [0, 0, 0, 0]
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
                which are represented as integer values. This version assumes the number of classes is 1 + max(max(labels), max(pred))
                For example, if labels = [0, 1, 1], predicted = [0, 2, 1] and weights = [1, 2, 3]
                [1, 0, 0]
                [0, 3, 2]
                [0, 0, 0]
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
                which are represented as integer values.
                For example, if labels = [0, 1, 1], predicted = [0, 2, 1], numClasses = 4, and weights = [1, 2, 3]
                [1, 0, 0, 0]
                [0, 3, 2, 0]
                [0, 0, 0, 0]
                [0, 0, 0, 0]
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
                tensor/subset along the specified dimensions:
                out = 1.0 - cosineSimilarity(x,y)
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
                along the specified dimensions:
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
                Returns an output variable with diagonal values equal to the specified values; off-diagonal values will be set to 0
                For example, if input = [1,2,3], then output is given by:
                [ 1, 0, 0]
                [ 0, 2, 0]
                [ 0, 0, 3]
                
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
                Extract the diagonal part from the input array.
                If input is
                [ 1, 0, 0]
                [ 0, 2, 0]
                [ 0, 0, 3]
                then output is [1, 2, 3].
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
                tensor/subset along the specified dimensions:
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
                Example:
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
                Generate an identity matrix with the specified number of rows and columns, with optional leading dims
                Example:
                batchShape: [3,3]
                numRows: 2
                numCols: 4
                returns a tensor of shape (3, 3, 2, 4) that consists of 3 * 3 batches of (2,4)-shaped identity matrices:
                1 0 0 0
                0 1 0 0
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
                First index reduction operation.
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
                First index reduction operation.
                Returns a variable that contains the index of the first element that matches the specified condition (for each
                slice along the specified dimensions)
                Note that if keepDims = true, the output variable has the same rank as the input variable,
                with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
                the mean along a dimension).
                Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
                keepDims = true: [a,1,c]
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
                tensor/subset along the specified dimensions:
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
                Is finite operation: elementwise isFinite(x)
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
                Is infinite operation: elementwise isInfinite(x)
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
                Is maximum operation: elementwise x == max(x)
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
                Is Not a Number operation: elementwise isNaN(x)
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
                Is the array non decreasing?
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
                Is the array strictly increasing?
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
                Last index reduction operation.
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
                Last index reduction operation.
                Returns a variable that contains the index of the last element that matches the specified condition (for each
                slice along the specified dimensions)
                Note that if keepDims = true, the output variable has the same rank as the input variable,
                with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
                the mean along a dimension).
                Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
                keepDims = true: [a,1,c]
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
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise natural logarithm function: out = log_e (1 + x)
            """.trimIndent()
        }
    }

    Op("logEntropy") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce on (null for full array)" }
        Output(NUMERIC, "output"){ description = "variable: reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Log entropy reduction: log(-sum(x * log(x)))
            """.trimIndent()
        }
    }

    Op("logSumExp") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.custom"
        Input(NUMERIC, "input") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Optional dimensions to reduce along" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Log-sum-exp reduction (optionally along dimension).
                Computes log(sum(exp(x))
            """.trimIndent()
        }
    }

    Op("manhattanDistance") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce3"
        Input(NUMERIC, "x") { description = "Input variable x" }
        Input(NUMERIC, "y") { description = "Input variable y" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to calculate cosine similarity over" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Manhattan distance (l1 norm, l1 distance) reduction operation. The output contains the Manhattan distance for each
                tensor/subset along the specified dimensions:
                out = sum_i abs(x[i]-y[i])
            """.trimIndent()
        }
    }

    Op("matrixDeterminant") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "in") { description = "Input" }
        Output(NUMERIC, "output"){ description = "Matrix determinant variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Matrix determinant op. For 2D input, this returns the standard matrix determinant.
                For higher dimensional input with shape [..., m, m] the matrix determinant is returned for each 
                shape [m,m] sub-matrix.
            """.trimIndent()
        }
    }

    Op("matrixInverse") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "in") { description = "Input" }
        Output(NUMERIC, "output"){ description = "Matrix inverse variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Matrix inverse op. For 2D input, this returns the standard matrix inverse.
                For higher dimensional input with shape [..., m, m] the matrix inverse is returned for each
                shape [m,m] sub-matrix.
            """.trimIndent()
        }
    }

    Op("mergeAdd") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic"
        Arg(NUMERIC, "inputs"){ count = AtLeast(1); description = "Input variables" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Merge add function: merges an arbitrary number of equal shaped arrays using element-wise addition:
                out = sum_i in[i]
            """.trimIndent()
        }
    }

    Op("mergeAvg") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Arg(NUMERIC, "inputs"){ count = AtLeast(1); description = "Input variables" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Merge average function: merges an arbitrary number of equal shaped arrays using element-wise mean operation:
                out = mean_i in[i]
            """.trimIndent()
        }
    }

    Op("mergeMax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        Arg(NUMERIC, "inputs"){ count = AtLeast(1); description = "Input variables" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Merge max function: merges an arbitrary number of equal shaped arrays using element-wise maximum operation:
                out = max_i in[i]
            """.trimIndent()
        }
    }

    Op("moments") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce"
        Input(NUMERIC, "input") { description = "Input to calculate moments for" }
        Arg(INT, "axes"){ count = AtLeast(0); description = "Dimensions to perform calculation over" }
        Output(NUMERIC, "output"){ description = "Mean and variance variables" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Calculate the mean and (population) variance for the input variable, for the specified axis
            """.trimIndent()
        }
    }

    Op("neg") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise negative operation: out = -x
            """.trimIndent()
        }
    }

    Op("normalizeMoments") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce"
        Input(NUMERIC, "counts") { description = "Rank 0 (scalar) value with the total number of values used to calculate the sufficient statistics" }
        Input(NUMERIC, "means") { description = "Mean-value sufficient statistics: this is the SUM of all data values" }
        Input(NUMERIC, "variances") { description = "Variaance sufficient statistics: this is the squared sum of all data values" }
        Input(NUMERIC, "shift") { description = "Shift value, possibly 0, used when calculating the sufficient statistics (for numerical stability)" }
        Output(NUMERIC, "output"){ description = "Output variables: mean and population variance" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Calculate the mean and variance from the sufficient statistics
            """.trimIndent()
        }
    }

    Op("or") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "y") { description = "Input 2" }
        Output(NUMERIC, "output"){ description = "SDVariable with values 0 and 1 based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Boolean OR operation: elementwise (x != 0) || (y != 0)
                If x and y arrays have equal shape, the output shape is the same as these inputs.
                Note: supports broadcasting if x and y have different shapes and are broadcastable.
                Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
            """.trimIndent()
        }
    }

    Op("pow") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        Input(NUMERIC, "x") { description = "Input variable" }
        Input(NUMERIC, "value") { description = "Power to raise each element to" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise power function: out = x^value
            """.trimIndent()
        }
    }

    Op("pow") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        Input(NUMERIC, "x") { description = "Input variable" }
        Input(NUMERIC, "y") { description = "Power" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise (broadcastable) power function: out = x[i]^y[i]
            """.trimIndent()
        }
    }

    Op("reciprocal") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
        Input(NUMERIC, "a") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise reciprocal (inverse) function: out[i] = 1 / in[i]
            """.trimIndent()
        }
    }

    Op("round") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise round function: out = round(x).
                Rounds (up or down depending on value) to the nearest integer value.
            """.trimIndent()
        }
    }

    Op("rsqrt") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.floating"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise reciprocal (inverse) of square root: out = 1.0 / sqrt(x)
            """.trimIndent()
        }
    }

    Op("setDiag") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "in") { description = "Input variable" }
        Input(NUMERIC, "diag") { description = "Diagonal" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Set the diagonal value to the specified values
                If input is
                [ a, b, c]
                [ d, e, f]
                [ g, h, i]
                and diag = [ 1, 2, 3] then output is
                [ 1, b, c]
                [ d, 2, f]
                [ g, h, 3]
            """.trimIndent()
        }
    }

    Op("shannonEntropy") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.floating"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce on (null/empty for full array)" }
        Output(NUMERIC, "output"){ description = "reduced array of rank (input rank - num dimensions)" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Shannon Entropy reduction: -sum(x * log2(x))
            """.trimIndent()
        }
    }

    Op("sign") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise sign (signum) function:
                out = -1 if in < 0
                out = 0 if in = 0
                out = 1 if in > 0
            """.trimIndent()
        }
    }

    Op("sin") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise sine operation: out = sin(x)
            """.trimIndent()
        }
    }

    Op("sinh") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise sinh (hyperbolic sine) operation: out = sinh(x)
            """.trimIndent()
        }
    }

    Op("sqrt") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.floating"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise square root function: out = sqrt(x)
            """.trimIndent()
        }
    }

    Op("square") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.same"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Element-wise square function: out = x^2
            """.trimIndent()
        }
    }

    Op("step") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        Input(NUMERIC, "in") { description = "Input variable" }
        Input(NUMERIC, "cutoff") { description = "Cutoff value for step function" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise step function:
                out(x) = 1 if x >= cutoff
                out(x) = 0 otherwise
            """.trimIndent()
        }
    }

    Op("standardize") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(1); description = "" } //TODO: Missing description for dimension.
        Output(NUMERIC, "output"){ description = "Output variable" }
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
            """.trimIndent()
        }
    }

    Op("tan") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise tangent operation: out = tan(x)
            """.trimIndent()
        }
    }

    Op("tanh") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.strict"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)
            """.trimIndent()
        }
    }

    Op("trace") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "in") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Trace" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Matrix trace operation
                For rank 2 matrices, the output is a scalar vith the trace - i.e., sum of the main diagonal.
                For higher rank inputs, output[a,b,c] = trace(in[a,b,c,:,:])
            """.trimIndent()
        }
    }

    Op("xor") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "y") { description = "Input 2" }
        Output(NUMERIC, "output"){ description = "SDVariable with values 0 and 1 based on where the condition is satisfied" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Boolean XOR (exclusive OR) operation: elementwise (x != 0) XOR (y != 0)
                If x and y arrays have equal shape, the output shape is the same as these inputs.
                Note: supports broadcasting if x and y have different shapes and are broadcastable.
                Returns an array with values 1 where condition is satisfied, or value 0 otherwise.
            """.trimIndent()
        }
    }

    Op("bitShift") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "shift") { description = "Number of bits to shift." }
        Output(NUMERIC, "output"){ description = "SDVariable with shifted bits" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Shift integer bits to the left, i.e. var << 4
            """.trimIndent()
        }
    }

    Op("bitShiftRight") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "shift") { description = "Number of bits to shift." }
        Output(NUMERIC, "output"){ description = "SDVariable with shifted bits" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Shift integer bits to the right, i.e. var >> 4
            """.trimIndent()
        }
    }

    Op("bitRotl") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "shift") { description = "Number of bits to shift." }
        Output(NUMERIC, "output"){ description = "SDVariable with shifted bits" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Roll integer bits to the left, i.e. var << 4 | var >> (32 - 4)
            """.trimIndent()
        }
    }

    Op("bitRotr") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { description = "Input 1" }
        Input(NUMERIC, "shift") { description = "Number of bits to shift." }
        Output(NUMERIC, "output"){ description = "SDVariable with shifted bits" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Roll integer bits to the right, i.e. var >> 4 | var << (32 - 4)
            """.trimIndent()
        }
    }

    Op("zeroFraction") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce"
        Input(NUMERIC, "input") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Reduced array of rank 0 (scalar)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Full array zero fraction array reduction operation, optionally along specified dimensions: out = (count(x == 0) / length(x))
            """.trimIndent()
        }
    }
}

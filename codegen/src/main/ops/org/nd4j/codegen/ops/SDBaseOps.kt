/**
 * Generated using ExtractFromExisting.kt
 */
package org.nd4j.codegen.ops

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*

fun SDBaseOps() =  Namespace("SDBaseOps"){
    val namespaceJavaPackage = "TODO"
    Op("argmax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
        legacy = true
        javaOpClass = "IMax"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue = false }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }

        Output(NUMERIC, "output"){ description = "reduced array of rank (input rank - num dimensions) if keepDims = false, or\n" +
                " of rank (input rank) if keepdims = true" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Argmax array reduction operation, optionally along specified dimensions.
                Output values are the index of the maximum value of each slice along the specified dimension.
                
                Note that if keepDims = true, the output variable has the same rank as the input variable,
                with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
                the mean along a dimension).
                Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
                keepDims = true: [a,1,c]
                keepDims = false: [a,c]
            """.trimIndent()
        }
    }

    Op("argmin") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
        legacy = true
        javaOpClass = "IMin"
        Input(NUMERIC, "in") { description = "Input variable" }
        Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions"; defaultValue = false }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "reduced array of rank (input rank - num dimensions) if keepDims = false, or of rank (input rank) if keepdims = true" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Argmin array reduction operation, optionally along specified dimensions.
                Output values are the index of the minimum value of each slice along the specified dimension.
                
                Note that if keepDims = true, the output variable has the same rank as the input variable,
                with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
                the mean along a dimension).
                Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
                keepDims = true: [a,1,c]
                keepDims = false: [a,c]
            """.trimIndent()
        }
    }

    Op("assign") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "x") { count = AtLeast(1); description = "Input variable x" }
        Input(NUMERIC, "y") { count = AtLeast(1); description = "Input variable y" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Assign/copy op: out = x.assign(y). Supports broadcasting
            """.trimIndent()
        }
    }

    Op("concat") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "dimension") { description = "" }
        Input(NUMERIC, "inputs") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Concatenate a set of inputs along the specified dimension.<br>
 Note that inputs must have identical rank and identical dimensions, other than the dimension to stack on.<br>
 For example, if 2 inputs have shape [a, x, c] and [a, y, c] and dimension = 1, then the output has shape [a, x+y, c]

 @param name      Name of the output variable
 @param dimension Dimension to concatenate on
 @param inputs    Input variables
 @return Output variable
 @see #stack(String, int, SDVariable...)
     
""".trimIndent()
        }
    }

    Op("cumprod") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "exclusive") { description = "" }
        Input(NUMERIC, "reverse") { description = "" }
        Input(NUMERIC, "axis") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Cumulative product operation.<br>
 For input: [ a, b, c], output is:<br>
 exclusize=false, reverse=false: [a, a*b, a*b*c]<br>
 exclusive=true, reverse=false, [0, a, a*b]<br>
 exclusive=false, reverse=true: [a*b*c, b*c, c]<br>
 exclusive=true, reverse=true: [b*c, c, 0]<br><br>

 @param name      Name of the output variable
 @param in        Input variable
 @param axis      Scalar axis argument for dimension to perform cumululative sum operations along
 @param exclusive If true: exclude the first value
 @param reverse   If true: reverse the direction of the accumulation
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("cumsum") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "exclusive") { description = "" }
        Input(NUMERIC, "reverse") { description = "" }
        Input(NUMERIC, "axis") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Cumulative sum operation.<br>
 For input: [ a, b, c], output is:<br>
 exclusize=false, reverse=false: [a, a+b, a+b+c]<br>
 exclusive=true, reverse=false, [0, a, a+b]<br>
 exclusive=false, reverse=true: [a+b+c, b+c, c]<br>
 exclusive=true, reverse=true: [b+c, c, 0]<br><br>

 @param name      Name of the output variable
 @param in        Input variable
 @param axis      Scalar axis argument for dimension to perform cumululative sum operations along
 @param exclusive If true: exclude the first value
 @param reverse   If true: reverse the direction of the accumulation
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("dot") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 TODO doc string

 @param name
 @param x
 @param y
 @param dimensions
 @return
     
""".trimIndent()
        }
    }

    Op("dynamicPartition") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "partitions") { description = "" }
        Input(NUMERIC, "numPartitions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Dynamically partition the input variable values into the specified number of paritions, using the indices.<br>
 Example:<br>
 <pre>
 {@code input = [1,2,3,4,5]
 numPartitions = 2
 partitions = [1,0,0,1,0]
 out[0] = [2,3,5]
 out[1] = [1,4] }
 </pre>

 @param name          Names for the output variables. Length must be equal to numPartitions
 @param x             Input variable
 @param partitions    1D input with values 0 to numPartitions-1
 @param numPartitions Number of partitions, >= 1
 @return Output variables (equal in number to numPartitions)
     
""".trimIndent()
        }
    }

    Op("dynamicStitch") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Dynamically merge the specified input arrays into a single array, using the specified indices

 @param name    Name of the output variable
 @param indices Indices to use when merging. Must be >= 1, same length as input variables
 @param x       Input variables.
 @return Merged output variable
     
""".trimIndent()
        }
    }

    Op("eq") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Equals operation: elementwise x == y<br>
 Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
 value 0 otherwise

 @param name Name of the output variable
 @param x    Input array
 @param y    Double value argument to use in operation
 @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     
""".trimIndent()
        }
    }

    Op("eq") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Equal to operation: elementwise x == y<br>
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

    Op("expandDims") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "axis") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Reshape the input by adding a 1 at the specified location.<br>
 For example, if input has shape [a, b], then output shape is:<br>
 axis = 0: [1, a, b]<br>
 axis = 1: [a, 1, b]<br>
 axis = 2: [a, b, 1]<br>

 @param name Name of the output variable
 @param x    Input variable
 @param axis Axis to expand
 @return Output variable
 @see #squeeze(String, SDVariable, int)
     
""".trimIndent()
        }
    }

    Op("fill") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "shape") { description = "" }
        Input(NUMERIC, "dataType") { description = "" }
        Input(NUMERIC, "value") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Generate an output variable with the specified (dynamic) shape with all elements set to the specified value

 @param name  Name of the output variable
 @param shape Shape: must be a 1D array/variable
 @param value Value to set all elements to
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("gather") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "df") { description = "" }
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "axis") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Gather slices from the input variable where the indices are specified as fixed int[] values.<br>
 Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.

 @param name    name of the output variable
 @param df      Input variable
 @param indices Indices to get
 @param axis    Axis that the indices refer to
 @return Output variable with slices pulled from the specified axis
     
""".trimIndent()
        }
    }

    Op("gather") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "df") { description = "" }
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "axis") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Gather slices from the input variable where the indices are specified as dynamic SDVariable values.<br>
 Output shape is same as input shape, except for axis dimension, which has size equal to indices.length.

 @param name    name of the output variable
 @param df      Input variable
 @param indices Indices to get slices for. Rank 0 or 1 input
 @param axis    Axis that the indices refer to
 @return Output variable with slices pulled from the specified axis
     
""".trimIndent()
        }
    }

    Op("gatherNd") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "df") { description = "" }
        Input(NUMERIC, "indices") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 TODO doc string

 @param name
 @param df
 @param indices
 @return
     
""".trimIndent()
        }
    }

    Op("gt") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Greater than operation: elementwise x > y<br>
 Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
 value 0 otherwise

 @param name Name of the output variable
 @param x    Input array
 @param y    Double value argument to use in operation
 @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     
""".trimIndent()
        }
    }

    Op("gt") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Greater than operation: elementwise x > y<br>
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

    Op("gte") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Greater than or equals operation: elementwise x >= y<br>
 Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
 value 0 otherwise

 @param name Name of the output variable
 @param x    Input array
 @param y    Double value argument to use in operation
 @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     
""".trimIndent()
        }
    }

    Op("gte") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Greater than or equal to operation: elementwise x >= y<br>
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

    Op("identity") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Elementwise identity operation: out = x

 @param name  name of the output variable
 @param input Input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("invertPermutation") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Compute the inverse permutation indices for a permutation operation<br>
 Example: if input is [2, 0, 1] then output is [1, 2, 0]<br>
 The idea is that x.permute(input).permute(invertPermutation(input)) == x

 @param name  name of the output variable
 @param input 1D indices for permutation
 @return 1D inverted permutation
     
""".trimIndent()
        }
    }

    Op("isNumericTensor") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Is the director a numeric tensor? In the current version of ND4J/SameDiff, this always returns true/1

 @param name Output variable name
 @param x    Input variable
 @return Scalar variable with value 1
     
""".trimIndent()
        }
    }

    Op("linspace") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "dataType") { description = "" }
        Input(NUMERIC, "start") { description = "" }
        Input(NUMERIC, "stop") { description = "" }
        Input(NUMERIC, "number") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Create a new 1d array with values evenly spaced between values 'start' and 'stop'
 For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]

 @param name     Name of the new variable
 @param dataType Data type of the output array
 @param start    Start value
 @param stop     Stop value
 @param number   Number of values to generate
 @return SDVariable with linearly spaced elements
     
""".trimIndent()
        }
    }

    Op("linspace") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "from") { description = "" }
        Input(NUMERIC, "to") { description = "" }
        Input(NUMERIC, "length") { description = "" }
        Input(NUMERIC, "dt") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Create a new 1d array with values evenly spaced between values 'start' and 'stop'
 For example, linspace(start=3.0, stop=4.0, number=3) will generate [3.0, 3.5, 4.0]

 @param name   Name of the new variable
 @param from   Start value
 @param to     Stop value
 @param length Number of values to generate
 @param dt     Data type of the output array
 @return SDVariable with linearly spaced elements
     
""".trimIndent()
        }
    }

    Op("lt") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Less than operation: elementwise x < y<br>
 Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
 value 0 otherwise

 @param name Name of the output variable
 @param x    Input array
 @param y    Double value argument to use in operation
 @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     
""".trimIndent()
        }
    }

    Op("lt") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Less than operation: elementwise x < y<br>
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

    Op("lte") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Less than or equals operation: elementwise x <= y<br>
 Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
 value 0 otherwise

 @param name Name of the output variable
 @param x    Input array
 @param y    Double value argument to use in operation
 @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     
""".trimIndent()
        }
    }

    Op("lte") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Less than or equal to operation: elementwise x <= y<br>
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

    Op("matchCondition") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "condition") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Returns a boolean mask of equal shape to the input, where the condition is satisfied - value 1 where satisfied, 0 otherwise

 @param in        Input
 @param condition Condition
 @return Boolean mask
     
""".trimIndent()
        }
    }

    Op("matchConditionCount") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "condition") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Returns a count of the number of elements that satisfy the condition

 @param name      Name of the output variable
 @param in        Input
 @param condition Condition
 @return Number of elements that the condition is satisfied for
     
""".trimIndent()
        }
    }

    Op("matchConditionCount") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "condition") { description = "" }
        Input(NUMERIC, "keepDim") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Returns a count of the number of elements that satisfy the condition (for each slice along the specified dimensions)<br>
 Note that if keepDims = true, the output variable has the same rank as the input variable,
 with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
 the mean along a dimension).<br>
 Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
 keepDims = true: [a,1,c]<br>
 keepDims = false: [a,c]

 @param name       Name of the output variable
 @param in         Input variable
 @param condition  Condition
 @param keepDim    If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
 @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Number of elements that the condition is satisfied for
     
""".trimIndent()
        }
    }

    Op("max") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Max array reduction operation, optionally along specified dimensions

 @param name       Output variable name
 @param x          Input variable
 @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("max") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "keepDims") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Max array reduction operation, optionally along specified dimensions<br>
 Note that if keepDims = true, the output variable has the same rank as the input variable,
 with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
 the mean along a dimension).<br>
 Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
 keepDims = true: [a,1,c]<br>
 keepDims = false: [a,c]

 @param name       Output variable name
 @param x          Input variable
 @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
 @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("max") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "first") { description = "" }
        Input(NUMERIC, "second") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise maximum operation: out[i] = max(first[i], second[i])<br>
 Supports broadcasting

 @param name   Name of the output variable
 @param first  First input array
 @param second Second input array
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("mean") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimension") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Mean (average) array reduction operation, optionally along specified dimensions

 @param name      Output variable name
 @param x         Input variable
 @param dimension Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("mean") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "keepDims") { description = "" }
        Input(NUMERIC, "dimension") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Mean (average) array reduction operation, optionally along specified dimensions<br>
 Note that if keepDims = true, the output variable has the same rank as the input variable,
 with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
 the mean along a dimension).<br>
 Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
 keepDims = true: [a,1,c]<br>
 keepDims = false: [a,c]

 @param name      Output variable name
 @param x         Input variable
 @param keepDims  If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
 @param dimension Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("min") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Minimum array reduction operation, optionally along specified dimensions. out = min(in)

 @param name       Output variable name
 @param x          Input variable
 @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("min") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "keepDims") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Minimum array reduction operation, optionally along specified dimensions. out = min(in)<br>
 Note that if keepDims = true, the output variable has the same rank as the input variable,
 with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
 the mean along a dimension).<br>
 Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
 keepDims = true: [a,1,c]<br>
 keepDims = false: [a,c]

 @param name       Output variable name
 @param x          Input variable
 @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
 @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("min") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "first") { description = "" }
        Input(NUMERIC, "second") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise minimum operation: out[i] = min(first[i], second[i])<br>
 Supports broadcasting

 @param name   Name of the output variable
 @param first  First input array
 @param second Second input array
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("mmul") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }
        Input(NUMERIC, "transpose") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Matrix multiplication: out = mmul(x,y)<br>
 Supports specifying a {@link MMulTranspose} argument to perform operation such as mmul(a^T, b), etc.

 @param name      Output variable name
 @param x         First input variable
 @param y         Second input variable
 @param transpose Transpose arguments
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("mmul") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Matrix multiplication: out = mmul(x,y)

 @param name Output variable name
 @param x    First input variable
 @param y    Second input variable
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("neq") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Not equals operation: elementwise x != y<br>
 Returns an array with the same shape/size as the input, with values 1 where condition is satisfied, or
 value 0 otherwise

 @param name Name of the output variable
 @param x    Input array
 @param y    Double value argument to use in operation
 @return Output SDVariable with values 0 and 1 based on where the condition is satisfied
     
""".trimIndent()
        }
    }

    Op("neq") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Not equal to operation: elementwise x != y<br>
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

    Op("norm1") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions:<br>
 out = sum_i abs(x[i])

 @param name       Output variable name
 @param x          Input variable
 @param dimensions dimensions to reduce over
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("norm1") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "keepDims") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Norm1 (L1 norm) reduction operation: The output contains the L1 norm for each tensor/subset along the specified dimensions:<br>
 out = sum_i abs(x[i])<br>
 Note that if keepDims = true, the output variable has the same rank as the input variable,
 with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
 the mean along a dimension).<br>
 Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
 keepDims = true: [a,1,c]<br>
 keepDims = false: [a,c]

 @param name       Output variable name
 @param x          Input variable
 @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
 @param dimensions dimensions to reduce over
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("norm2") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:<br>
 out = sqrt(sum_i x[i]^2)

 @param name       Output variable name
 @param x          Input variable
 @param dimensions dimensions to reduce over
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("norm2") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "keepDims") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Norm2 (L2 norm) reduction operation: The output contains the L2 norm for each tensor/subset along the specified dimensions:<br>
 out = sqrt(sum_i x[i]^2)<br>
 Note that if keepDims = true, the output variable has the same rank as the input variable,
 with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
 the mean along a dimension).<br>
 Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
 keepDims = true: [a,1,c]<br>
 keepDims = false: [a,c]

 @param name       Output variable name
 @param x          Input variable
 @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
 @param dimensions dimensions to reduce over
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("normmax") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the
 specified dimensions

 @param name       Output variable name
 @param x          Input variable
 @param dimensions dimensions to reduce over
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("normmax") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "keepDims") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Max norm (infinity norm) reduction operation: The output contains the max norm for each tensor/subset along the
 specified dimensions:<br>
 out = max(abs(x[i]))<br>
 Note that if keepDims = true, the output variable has the same rank as the input variable,
 with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
 the mean along a dimension).<br>
 Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
 keepDims = true: [a,1,c]<br>
 keepDims = false: [a,c]

 @param name       Output variable name
 @param x          Input variable
 @param keepDims   If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
 @param dimensions dimensions to reduce over
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("oneHot") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "depth") { description = "" }
        Input(NUMERIC, "axis") { description = "" }
        Input(NUMERIC, "on") { description = "" }
        Input(NUMERIC, "off") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Convert the array to a one-hot array with walues {@code on} and {@code off} for each entry<br>
 If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],
 with {@code out[i, ..., j, in[i,...,j]] = on} with other values being set to {@code off}

 @param name    Output variable name
 @param indices Indices - value 0 to depth-1
 @param depth   Number of classes
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("oneHot") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "depth") { description = "" }
        Input(NUMERIC, "axis") { description = "" }
        Input(NUMERIC, "on") { description = "" }
        Input(NUMERIC, "off") { description = "" }
        Input(NUMERIC, "dataType") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 As per {@link #oneHot(String, SDVariable, int, int, double, double)} but allows configuring the output datatype
     
""".trimIndent()
        }
    }

    Op("oneHot") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "depth") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Convert the array to a one-hot array with walues 0 and 1 for each entry<br>
 If input has shape [ a, ..., n] then output has shape [ a, ..., n, depth],
 with out[i, ..., j, in[i,...,j]] = 1 with other values being set to 0

 @param name    Output variable name
 @param indices Indices - value 0 to depth-1
 @param depth   Number of classes
 @return Output variable
 @see #oneHot(SDVariable, int, int, double, double)
     
""".trimIndent()
        }
    }

    Op("onesLike") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Return a variable of all 1s, with the same shape as the input variable. Note that this is dynamic:
 if the input shape changes in later execution, the returned variable's shape will also be updated

 @param name  Name of the new SDVariable
 @param input Input SDVariable
 @return A new SDVariable with the same (dynamic) shape as the input
     
""".trimIndent()
        }
    }

    Op("onesLike") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }
        Input(NUMERIC, "dataType") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 As per {@link #onesLike(String, SDVariable)} but the output datatype may be specified
     
""".trimIndent()
        }
    }

    Op("parallel_stack") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "values") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 @see #stack(String, int, SDVariable...)
     
""".trimIndent()
        }
    }

    Op("permute") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Array permutation operation: permute the dimensions according to the specified permutation indices.<br>
 Example: if input has shape [a,b,c] and dimensions = [2,0,1] the output has shape [c,a,b]

 @param name Output variable name
 @param x    Input variable
 @return Output variable (permuted input)
     
""".trimIndent()
        }
    }

    Op("permute") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 As per {@link #permute(String, SDVariable, int...)} but with SDVariable permute dimension
     
""".trimIndent()
        }
    }

    Op("prod") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Product array reduction operation, optionally along specified dimensions

 @param name       Output variable name
 @param x          Input variable
 @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Output variable: reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("prod") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "keepDims") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Product array reduction operation, optionally along specified dimensions<br>
 Note that if keepDims = true, the output variable has the same rank as the input variable,
 with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
 the mean along a dimension).<br>
 Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
 keepDims = true: [a,1,c]<br>
 keepDims = false: [a,c]

 @param name       Output variable name
 @param x          Input variable
 @param keepDims   If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
 @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Output variable: reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("range") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "from") { description = "" }
        Input(NUMERIC, "to") { description = "" }
        Input(NUMERIC, "step") { description = "" }
        Input(NUMERIC, "dataType") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Create a new variable with a 1d array, where the values start at {@code from} and increment by {@code step}
 up to (but not including) limit.<br>
 For example, {@code range(1.0, 3.0, 0.5)} will return {@code [1.0, 1.5, 2.0, 2.5]}

 @param name Name of the new variable
 @param from Initial/smallest value
 @param to   Largest value (exclusive)
 @param step Step size
 @return 1D SDVariable with the specified values
     
""".trimIndent()
        }
    }

    Op("range") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "from") { description = "" }
        Input(NUMERIC, "to") { description = "" }
        Input(NUMERIC, "step") { description = "" }
        Input(NUMERIC, "dataType") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 As per {@link #range(String, double, double, double, DataType)} but with SDVariable arguments
     
""".trimIndent()
        }
    }

    Op("rank") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Returns the rank (number of dimensions, i.e., length(shape)) of the specified SDVariable as a 0D scalar variable

 @param name Name of the output variable
 @param in   Input variable
 @return 0D (scalar) output variable with value equal to the rank of the input variable
     
""".trimIndent()
        }
    }

    Op("repeat") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "df") { description = "" }
        Input(NUMERIC, "axis") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 @see #repeat(String, SDVariable, int)
     
""".trimIndent()
        }
    }

    Op("replaceWhere") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "update") { description = "" }
        Input(NUMERIC, "from") { description = "" }
        Input(NUMERIC, "condition") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise replace where condition:<br>
 out[i] = from[i] if condition(update[i]) is satisfied, or<br>
 out[i] = update[i] if condition(update[i]) is NOT satisfied

 @param name      Name of the output variable
 @param update    Source array
 @param from      Replacement values array (used conditionally). Must be same shape as 'update' array
 @param condition Condition to check on update array elements
 @return New array with values replaced where condition is satisfied
     
""".trimIndent()
        }
    }

    Op("replaceWhere") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "update") { description = "" }
        Input(NUMERIC, "value") { description = "" }
        Input(NUMERIC, "condition") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise replace where condition:<br>
 out[i] = value if condition(update[i]) is satisfied, or<br>
 out[i] = update[i] if condition(update[i]) is NOT satisfied

 @param name      Name of the output variable
 @param update    Source array
 @param value     Value to set at the output, if the condition is satisfied
 @param condition Condition to check on update array elements
 @return New array with values replaced where condition is satisfied
     
""".trimIndent()
        }
    }

    Op("reshape") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "shape") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the
 input, but with the specified shape.<br>
 Note that prod(shape) must match length(input) == prod(input.shape)

 @param name  Output variable name
 @param x     Input variable
 @param shape New shape for variable
 @return Output variable
 @see #reshape(SDVariable, SDVariable)
     
""".trimIndent()
        }
    }

    Op("reshape") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "shape") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Reshape the input variable to the specified (fixed) shape. The output variable will have the same values as the
 input, but with the specified shape.<br>
 Note that prod(shape) must match length(input) == prod(input.shape)

 @param name  Output variable name
 @param x     Input variable
 @param shape New shape for variable
 @return Output variable
 @see #reshape(SDVariable, SDVariable)
     
""".trimIndent()
        }
    }

    Op("reshape") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "shape") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Reshape the input variable to the specified (dynamic) shape. The output variable will have the same values as the
 input, but with the specified shape.<br>
 Note that prod(shape) must match length(input) == prod(input.shape)

 @param name  Output variable name
 @param x     Input variable
 @param shape New shape for variable
 @return Output variable
 @see #reshape(SDVariable, int[])
     
""".trimIndent()
        }
    }

    Op("reverse") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Reverse the values of an array for the specified dimensions<br>
 If input is:<br>
 [ 1, 2, 3]<br>
 [ 4, 5, 6]<br>
 then<br>
 reverse(in, 0):<br>
 [3, 2, 1]<br>
 [6, 5, 4]<br>
 reverse(in, 0):<br>
 [4, 5, 6]<br>
 [1, 2 3]<br>

 @param x          Input variable
 @param dimensions Dimensions
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("reverseSequence") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "seq_lengths") { description = "" }
        Input(NUMERIC, "seqDim") { description = "" }
        Input(NUMERIC, "batchDim") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Reverse sequence op: for each slice along dimension seqDimension, the first seqLength values are reversed

 @param name        Name of the output variable
 @param x           Input variable
 @param seq_lengths Length of the sequences
 @param seqDim      Sequence dimension
 @param batchDim    Batch dimension
 @return Reversed sequences
     
""".trimIndent()
        }
    }

    Op("reverseSequence") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "seq_lengths") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 @see #reverseSequence(String, SDVariable, SDVariable, int, int)
     
""".trimIndent()
        }
    }

    Op("scalarFloorMod") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "value") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise scalar floor modulus operation: out = floorMod(in, value).
 i.e., returns the remainder after division by 'value'

 @param name  Name of the output variable
 @param in    Input variable
 @param value Scalar value to compare
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("scalarMax") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "value") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise scalar maximum operation: out = max(in, value)

 @param name  Name of the output variable
 @param in    Input variable
 @param value Scalar value to compare
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("scalarMin") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "value") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Element-wise scalar minimum operation: out = min(in, value)

 @param name  Name of the output variable
 @param in    Input variable
 @param value Scalar value to compare
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("scalarSet") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "set") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Return a variable with equal shape to the input, but all elements set to value 'set'

 @param name Name of the output variable
 @param in   Input variable
 @param set  Value to set
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("scatterAdd") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "ref") { description = "" }
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "updates") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Scatter addition operation.<br>
 If indices is rank 0 (a scalar), then out[index, ...] += updates[...]<br>
 If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] += updates[i, ...]<br>
 If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] += updates[i, ..., k, ...]<br>
 Note that if multiple indices refer to the same location, the contributions from each is handled correctly.

 @param name    Name of the output variable
 @param ref     Initial/source variable
 @param indices Indices array
 @param updates Updates to add to the initial/source array
 @return The updated variable
     
""".trimIndent()
        }
    }

    Op("scatterDiv") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "ref") { description = "" }
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "updates") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Scatter division operation.<br>
 If indices is rank 0 (a scalar), then out[index, ...] /= updates[...]<br>
 If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] /= updates[i, ...]<br>
 If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] /= updates[i, ..., k, ...]<br>
 Note that if multiple indices refer to the same location, the contributions from each is handled correctly.

 @param name    Name of the output variable
 @param ref     Initial/source variable
 @param indices Indices array
 @param updates Updates to add to the initial/source array
 @return The updated variable
     
""".trimIndent()
        }
    }

    Op("scatterMax") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "ref") { description = "" }
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "updates") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Scatter max operation.<br>
 If indices is rank 0 (a scalar), then out[index, ...] = max(updates[...], in[index,...])<br>
 If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = max(updates[i,...], in[indices[i],...])<br>
 If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = max(updates[i, ..., k, ...], in[indices[i], ..., indices[k], ...]<br>
 Note that if multiple indices refer to the same location, the contributions from each is handled correctly.

 @param name    Name of the output variable
 @param ref     Initial/source variable
 @param indices Indices array
 @param updates Updates to add to the initial/source array
 @return The updated variable
     
""".trimIndent()
        }
    }

    Op("scatterMin") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "ref") { description = "" }
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "updates") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Scatter min operation.<br>
 If indices is rank 0 (a scalar), then out[index, ...] = min(updates[...], in[index,...])<br>
 If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = min(updates[i,...], in[indices[i],...])<br>
 If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = min(updates[i, ..., k, ...], in[indices[i], ..., indices[k], ...]<br>
 Note that if multiple indices refer to the same location, the contributions from each is handled correctly.

 @param name    Name of the output variable
 @param ref     Initial/source variable
 @param indices Indices array
 @param updates Updates to add to the initial/source array
 @return The updated variable
     
""".trimIndent()
        }
    }

    Op("scatterMul") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "ref") { description = "" }
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "updates") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Scatter multiplication operation.<br>
 If indices is rank 0 (a scalar), then out[index, ...] *= updates[...]<br>
 If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] *= updates[i, ...]<br>
 If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] *= updates[i, ..., k, ...]<br>
 Note that if multiple indices refer to the same location, the contributions from each is handled correctly.

 @param name    Name of the output variable
 @param ref     Initial/source variable
 @param indices Indices array
 @param updates Updates to add to the initial/source array
 @return The updated variable
     
""".trimIndent()
        }
    }

    Op("scatterSub") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "ref") { description = "" }
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "updates") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Scatter subtraction operation.<br>
 If indices is rank 0 (a scalar), then out[index, ...] -= updates[...]<br>
 If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] -= updates[i, ...]<br>
 If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] -= updates[i, ..., k, ...]<br>
 Note that if multiple indices refer to the same location, the contributions from each is handled correctly.

 @param name    Name of the output variable
 @param ref     Initial/source variable
 @param indices Indices array
 @param updates Updates to add to the initial/source array
 @return The updated variable
     
""".trimIndent()
        }
    }

    Op("scatterUpdate") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "ref") { description = "" }
        Input(NUMERIC, "indices") { description = "" }
        Input(NUMERIC, "updates") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Scatter update operation.<br>
 If indices is rank 0 (a scalar), then out[index, ...] = updates[...]<br>
 If indices is rank 1 (a vector), then for each position i, out[indices[i], ...] = updates[i, ...]<br>
 If indices is rank 2+, then for each position (i,...,k), out[indices[i], ..., indices[k], ...] = updates[i, ..., k, ...]<br>
 Note that if multiple indices refer to the same location, the output at those locations is undefined - different
 updates may occur in different orders

 @param name    Name of the output variable
 @param ref     Initial/source variable
 @param indices Indices array
 @param updates Updates to add to the initial/source array
 @return The updated variable
     
""".trimIndent()
        }
    }

    Op("segmentMax") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "data") { description = "" }
        Input(NUMERIC, "segmentIds") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Segment max operation.<br>
 If data =     [3, 6, 1, 4, 9, 2, 8]<br>
 segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
 then output = [6, 9, 8] = [max(3,6), max(1,4,9), max(2,8)]<br>
 Note that the segment IDs must be sorted from smallest to largest segment.
 See {@link #unsortedSegmentMax(String, SDVariable, SDVariable, int)}
 for the same op without this sorted requirement

 @param name       Name of the output variable. May be null
 @param data       Data to perform segment max on
 @param segmentIds Variable for the segment IDs
 @return Segment max output
     
""".trimIndent()
        }
    }

    Op("segmentMean") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "data") { description = "" }
        Input(NUMERIC, "segmentIds") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Segment mean operation.<br>
 If data =     [3, 6, 1, 4, 9, 2, 8]<br>
 segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
 then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>
 Note that the segment IDs must be sorted from smallest to largest segment.
 See {@link #unsortedSegmentMean(String, SDVariable, SDVariable, int)} for the same op without this sorted requirement

 @param name       Name of the output variable. May be null
 @param data       Data to perform segment max on
 @param segmentIds Variable for the segment IDs
 @return Segment mean output
     
""".trimIndent()
        }
    }

    Op("segmentMin") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "data") { description = "" }
        Input(NUMERIC, "segmentIds") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Segment min operation.<br>
 If data =     [3, 6, 1, 4, 9, 2, 8]<br>
 segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
 then output = [3, 1, 2] = [min(3,6), min(1,4,9), min(2,8)]<br>
 Note that the segment IDs must be sorted from smallest to largest segment.
 See {@link #unsortedSegmentMin(String, SDVariable, SDVariable, int)} for the same op without this sorted requirement

 @param name       Name of the output variable. May be null
 @param data       Data to perform segment max on
 @param segmentIds Variable for the segment IDs
 @return Segment min output
     
""".trimIndent()
        }
    }

    Op("segmentProd") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "data") { description = "" }
        Input(NUMERIC, "segmentIds") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Segment product operation.<br>
 If data =     [3, 6, 1, 4, 9, 2, 8]<br>
 segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
 then output = [18, 36, 16] = [prod(3,6), prod(1,4,9), prod(2,8)]<br>
 Note that the segment IDs must be sorted from smallest to largest segment.
 See {@link #unsortedSegmentProd(String, SDVariable, SDVariable, int)} for the same op without this sorted requirement

 @param name       Name of the output variable. May be null
 @param data       Data to perform segment max on
 @param segmentIds Variable for the segment IDs
 @return Segment product output
     
""".trimIndent()
        }
    }

    Op("segmentSum") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "data") { description = "" }
        Input(NUMERIC, "segmentIds") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Segment sum operation.<br>
 If data =     [3, 6, 1, 4, 9, 2, 8]<br>
 segmentIds =  [0, 0, 1, 1, 1, 2, 2]<br>
 then output = [9, 14, 10] = [sum(3,6), sum(1,4,9), sum(2,8)]<br>
 Note that the segment IDs must be sorted from smallest to largest segment.
 See {@link #unsortedSegmentSum(String, SDVariable, SDVariable, int)} for the same op without this sorted requirement

 @param name       Name of the output variable. May be null
 @param data       Data to perform segment max on
 @param segmentIds Variable for the segment IDs
 @return Segment sum output
     
""".trimIndent()
        }
    }

    Op("sequenceMask") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "lengths") { description = "" }
        Input(NUMERIC, "maxLen") { description = "" }
        Input(NUMERIC, "dataType") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 @see #sequenceMask(String, SDVariable, SDVariable, DataType)
     
""".trimIndent()
        }
    }

    Op("sequenceMask") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "lengths") { description = "" }
        Input(NUMERIC, "dataType") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 @see #sequenceMask(String, SDVariable, SDVariable, DataType)
     
""".trimIndent()
        }
    }

    Op("sequenceMask") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "lengths") { description = "" }
        Input(NUMERIC, "maxLen") { description = "" }
        Input(NUMERIC, "dataType") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Generate a sequence mask (with values 0 or 1) based on the specified lengths<br>
 Specifically, out[i, ..., k, j] = (j < lengths[i, ..., k] ? 1.0 : 0.0)

 @param name    Name of the output variable
 @param lengths Lengths of the sequences
 @param maxLen  Maximum sequence length
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("shape") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Returns the shape of the specified SDVariable as a 1D SDVariable

 @param name  Name of the output variable
 @param input Input variable
 @return 1D output variable with contents equal to the shape of the input
     
""".trimIndent()
        }
    }

    Op("size") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Returns the size (number of elements, i.e., prod(shape)) of the specified SDVariable as a 0D scalar variable

 @param name Name of the output variable
 @param in   Input variable
 @return 0D (scalar) output variable with value equal to the number of elements in the specified array
     
""".trimIndent()
        }
    }

    Op("sizeAt") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "dimension") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Returns a rank 0 (scalar) variable for the size of the specified dimension.
 For example, if X has shape [10,20,30] then sizeAt(X,1)=20. Similarly, sizeAt(X,-1)=30

 @param name      Name of the output variable
 @param in        Input variable
 @param dimension Dimension to get size of
 @return Scalar SDVariable for size at specified variable
     
""".trimIndent()
        }
    }

    Op("slice") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }
        Input(NUMERIC, "begin") { description = "" }
        Input(NUMERIC, "size") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Get a subset of the specified input, by specifying the first element and the size of the array.<br>
 For example, if input is:<br>
 [a, b, c]<br>
 [d, e, f]<br>
 then slice(input, begin=[0,1], size=[2,1] will return:<br>
 [b]<br>
 [e]<br>
 <br>
 Note that for each dimension i, begin[i] + size[i] <= input.size(i)

 @param name  Output variable name
 @param input Variable to get subset of
 @param begin Beginning index. Must be same length as rank of input array
 @param size  Size of the output array. Must be same length as rank of input array
 @return Subset of the input
     
""".trimIndent()
        }
    }

    Op("squaredNorm") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "keepDims") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Squared L2 norm: see {@link #norm2(String, SDVariable, boolean, int...)}
     
""".trimIndent()
        }
    }

    Op("squaredNorm") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Squared L2 norm: see {@link #norm2(String, SDVariable, int...)}
     
""".trimIndent()
        }
    }

    Op("squeeze") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "axis") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Remove a single dimension of size 1.
 For example, if input has shape [a,b,1,c] then squeeze(input, 2) returns an array of shape [a,b,c]

 @param name Name of the output variable
 @param x    Input variable
 @param axis Size 1 dimension to remove
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("stack") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "axis") { description = "" }
        Input(NUMERIC, "values") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Stack a set of N SDVariables of rank X into one rank X+1 variable.
 If inputs have shape [a,b,c] then output has shape:<br>
 axis = 0: [N,a,b,c]<br>
 axis = 1: [a,N,b,c]<br>
 axis = 2: [a,b,N,c]<br>
 axis = 3: [a,b,c,N]<br>

 @param name   Name of the output variable
 @param axis   Axis to stack on
 @param values Input variables to stack. Must have the same shape for all inputs
 @return Output variable
 @see #unstack(String[], SDVariable, int, int)
     
""".trimIndent()
        }
    }

    Op("standardDeviation") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "biasCorrected") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Stardard deviation array reduction operation, optionally along specified dimensions

 @param name          Output variable name
 @param x             Input variable
 @param biasCorrected If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)
 @param dimensions    Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Output variable: reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("standardDeviation") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "biasCorrected") { description = "" }
        Input(NUMERIC, "keepDims") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Stardard deviation array reduction operation, optionally along specified dimensions<br>
 Note that if keepDims = true, the output variable has the same rank as the input variable,
 with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
 the mean along a dimension).<br>
 Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
 keepDims = true: [a,1,c]<br>
 keepDims = false: [a,c]

 @param x             Input variable
 @param biasCorrected If true: divide by (N-1) (i.e., sample stdev). If false: divide by N (population stdev)
 @param keepDims      If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
 @param dimensions    Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Output variable: reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("stridedSlice") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }
        Input(NUMERIC, "begin") { description = "" }
        Input(NUMERIC, "end") { description = "" }
        Input(NUMERIC, "strides") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 @see #stridedSlice(String, SDVariable, long[], long[], long[])
     
""".trimIndent()
        }
    }

    Op("stridedSlice") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "begin") { description = "" }
        Input(NUMERIC, "end") { description = "" }
        Input(NUMERIC, "strides") { description = "" }
        Input(NUMERIC, "beginMask") { description = "" }
        Input(NUMERIC, "endMask") { description = "" }
        Input(NUMERIC, "ellipsisMask") { description = "" }
        Input(NUMERIC, "newAxisMask") { description = "" }
        Input(NUMERIC, "shrinkAxisMask") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 @see #stridedSlice(String, SDVariable, long[], long[], long[], int, int, int, int, int)
     
""".trimIndent()
        }
    }

    Op("stridedSlice") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }
        Input(NUMERIC, "begin") { description = "" }
        Input(NUMERIC, "end") { description = "" }
        Input(NUMERIC, "strides") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Get a subset of the specified input, by specifying the first element, last element, and the strides.<br>
 For example, if input is:<br>
 [a, b, c]<br>
 [d, e, f]<br>
 [g, h, i]<br>
 then stridedSlice(input, begin=[0,1], end=[2,2], strides=[2,1]) will return:<br>
 [b, c]<br>
 [h, i]<br>
 <br>

 @param name    Output variable name
 @param input   Variable to get subset of
 @param begin   Beginning index. Must be same length as rank of input array
 @param end     End index. Must be same length as the rank of the array
 @param strides Stride ("step size") for each dimension. Must be same length as the rank of the array. For example,
                stride of 2 means take every second element.
 @return Subset of the input
     
""".trimIndent()
        }
    }

    Op("stridedSlice") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "in") { description = "" }
        Input(NUMERIC, "begin") { description = "" }
        Input(NUMERIC, "end") { description = "" }
        Input(NUMERIC, "strides") { description = "" }
        Input(NUMERIC, "beginMask") { description = "" }
        Input(NUMERIC, "endMask") { description = "" }
        Input(NUMERIC, "ellipsisMask") { description = "" }
        Input(NUMERIC, "newAxisMask") { description = "" }
        Input(NUMERIC, "shrinkAxisMask") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Get a subset of the specified input, by specifying the first element, last element, and the strides.<br>
 Operates as described in {@link #stridedSlice(SDVariable, long[], long[], long[])} with some extra mask arrays
 as described below.

 @param name           Output variable name
 @param in             Variable to get subset of
 @param begin          Beginning index
 @param end            End index
 @param strides        Stride ("step size") for each dimension. For example,
                       stride of 2 means take every second element.
 @param beginMask      Bit mask: If the ith bit is set to 1, then the value in the begin long[] is ignored,
                       and a value of 0 is used instead for the beginning index for that dimension
 @param endMask        Bit mask: If the ith bit is set to 1, then the value in the end long[] is ignored,
                       and a value of size(i)-1 is used instead for the end index for that dimension
 @param ellipsisMask   Bit mask: only one non-zero value is allowed here. If a non-zero value is set, then other
                       dimensions are inserted as required at the specified position
 @param newAxisMask    Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and
                       a size 1 dimension is inserted at this point
 @param shrinkAxisMask Bit mask: if the ith bit is set to 1, then the begin/end/stride values are ignored, and
                       a size 1 dimension is removed at this point. Note that begin/end/stride values must
                       result in a size 1 output for these dimensions
 @return A subset of the input array
     
""".trimIndent()
        }
    }

    Op("sum") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Sum array reduction operation, optionally along specified dimensions

 @param x          Input variable
 @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Output variable: reduced array of rank (input rank - num dimensions) if keepDims = false, or
 of rank (input rank) if keepdims = true
     
""".trimIndent()
        }
    }

    Op("sum") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "keepDims") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Sum array reduction operation, optionally along specified dimensions.<br>
 Note that if keepDims = true, the output variable has the same rank as the input variable,
 with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
 the mean along a dimension).<br>
 Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
 keepDims = true: [a,1,c]<br>
 keepDims = false: [a,c]

 @param name       Output variable name
 @param x          Input variable
 @param keepDims   If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions
 @param dimensions Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Output variable: reduced array of rank (input rank - num dimensions) if keepDims = false, or
 of rank (input rank) if keepdims = true
     
""".trimIndent()
        }
    }

    Op("tensorMmul") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "y") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 @param x          Input variable x
 @param y          Input variable y
 @param dimensions dimensions
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("tile") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "repeat") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Repeat (tile) the input tensor the specified number of times.<br>
 For example, if input is<br>
 [1, 2]<br>
 [3, 4]<br>
 and repeat is [2, 3]<br>
 then output is<br>
 [1, 2, 1, 2, 1, 2]<br>
 [3, 4, 3, 4, 3, 4]<br>
 [1, 2, 1, 2, 1, 2]<br>
 [3, 4, 3, 4, 3, 4]<br>
 <br>

 @param name   Output variable name
 @param x      Input variable
 @param repeat Number of times to repeat in each axis. Must have length equal to the rank of the input array
 @return Output variable
     
""".trimIndent()
        }
    }

    Op("tile") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "repeat") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 @see #tile(String, SDVariable, int...)
     
""".trimIndent()
        }
    }

    Op("transpose") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Matrix transpose operation: If input has shape [a,b] output has shape [b,a]

 @param name Output variable name
 @param x    Input variable
 @return Output variable (transposed input)
     
""".trimIndent()
        }
    }

    Op("unsortedSegmentMax") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "data") { description = "" }
        Input(NUMERIC, "segmentIds") { description = "" }
        Input(NUMERIC, "numSegments") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Unsorted segment max operation. As per {@link #segmentMax(String, SDVariable, SDVariable)} but without
 the requirement for the indices to be sorted.<br>
 If data =     [1, 3, 2, 6, 4, 9, 8]<br>
 segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
 then output = [6, 9, 8] = [max(3,6), max(1,4,9), max(2,8)]<br>

 @param name        Name of the output variable
 @param data        Data (variable) to perform unsorted segment max on
 @param segmentIds  Variable for the segment IDs
 @param numSegments Number of segments
 @return Unsorted segment max output
     
""".trimIndent()
        }
    }

    Op("unsortedSegmentMean") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "data") { description = "" }
        Input(NUMERIC, "segmentIds") { description = "" }
        Input(NUMERIC, "numSegments") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Unsorted segment mean operation. As per {@link #segmentMean(String, SDVariable, SDVariable)} but without
 the requirement for the indices to be sorted.<br>
 If data =     [1, 3, 2, 6, 4, 9, 8]<br>
 segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
 then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>

 @param name        Name of the output variable
 @param data        Data (variable) to perform unsorted segment mean on
 @param segmentIds  Variable for the segment IDs
 @param numSegments Number of segments
 @return Unsorted segment mean output
     
""".trimIndent()
        }
    }

    Op("unsortedSegmentMin") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "data") { description = "" }
        Input(NUMERIC, "segmentIds") { description = "" }
        Input(NUMERIC, "numSegments") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Unsorted segment min operation. As per {@link #segmentMin(String, SDVariable, SDVariable)} but without
 the requirement for the indices to be sorted.<br>
 If data =     [1, 3, 2, 6, 4, 9, 8]<br>
 segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
 then output = [3, 1, 2] = [min(3,6), min(1,4,9), min(2,8)]<br>

 @param name        Name of the output variable
 @param data        Data (variable) to perform unsorted segment min on
 @param segmentIds  Variable for the segment IDs
 @param numSegments Number of segments
 @return Unsorted segment min output
     
""".trimIndent()
        }
    }

    Op("unsortedSegmentProd") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "data") { description = "" }
        Input(NUMERIC, "segmentIds") { description = "" }
        Input(NUMERIC, "numSegments") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Unsorted segment product operation. As per {@link #segmentProd(String, SDVariable, SDVariable)} but without
 the requirement for the indices to be sorted.<br>
 If data =     [1, 3, 2, 6, 4, 9, 8]<br>
 segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
 then output = [4.5, 4.666, 5] = [mean(3,6), mean(1,4,9), mean(2,8)]<br>

 @param name       Name of the output variable
 @param data       Data (variable) to perform unsorted segment product on
 @param segmentIds Variable for the segment IDs
 @return Unsorted segment product output
     
""".trimIndent()
        }
    }

    Op("unsortedSegmentSqrtN") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "data") { description = "" }
        Input(NUMERIC, "segmentIds") { description = "" }
        Input(NUMERIC, "numSegments") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Unsorted segment sqrtN operation. Simply returns the sqrt of the count of the number of values in each segment<br>
 If data =     [1, 3, 2, 6, 4, 9, 8]<br>
 segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
 then output = [1.414, 1.732, 1.414] = [sqrt(2), sqrtN(3), sqrtN(2)]<br>

 @param name       Name of the output variable
 @param data       Data (variable) to perform unsorted segment sqrtN on
 @param segmentIds Variable for the segment IDs
 @return Unsorted segment sqrtN output
     
""".trimIndent()
        }
    }

    Op("unsortedSegmentSum") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "data") { description = "" }
        Input(NUMERIC, "segmentIds") { description = "" }
        Input(NUMERIC, "numSegments") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Unsorted segment sum operation. As per {@link #segmentSum(String, SDVariable, SDVariable)} but without
 the requirement for the indices to be sorted.<br>
 If data =     [1, 3, 2, 6, 4, 9, 8]<br>
 segmentIds =  [1, 0, 2, 0, 1, 1, 2]<br>
 then output = [9, 14, 10] = [sum(3,6), sum(1,4,9), sum(2,8)]<br>

 @param name        Name of the output variable
 @param data        Data (variable) to perform unsorted segment sum on
 @param segmentIds  Variable for the segment IDs
 @param numSegments Number of segments
 @return Unsorted segment sum output
     
""".trimIndent()
        }
    }

    Op("variance") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "biasCorrected") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Variance array reduction operation, optionally along specified dimensions

 @param name          Output variable name
 @param x             Input variable
 @param biasCorrected If true: divide by (N-1) (i.e., sample variable). If false: divide by N (population variance)
 @param dimensions    Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Output variable: reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("variance") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "x") { description = "" }
        Input(NUMERIC, "biasCorrected") { description = "" }
        Input(NUMERIC, "keepDims") { description = "" }
        Input(NUMERIC, "dimensions") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Variance array reduction operation, optionally along specified dimensions<br>
 Note that if keepDims = true, the output variable has the same rank as the input variable,
 with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
 the mean along a dimension).<br>
 Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
 keepDims = true: [a,1,c]<br>
 keepDims = false: [a,c]

 @param name          Output variable name
 @param x             Input variable
 @param biasCorrected If true: divide by (N-1) (i.e., sample variable). If false: divide by N (population variance)
 @param keepDims      If true: keep the dimensions that are reduced on (as size 1). False: remove the reduction dimensions
 @param dimensions    Dimensions to reduce over. If dimensions are not specified, full array reduction is performed
 @return Output variable: reduced array of rank (input rank - num dimensions)
     
""".trimIndent()
        }
    }

    Op("zerosLike") {
        javaPackage = namespaceJavaPackage
        Input(NUMERIC, "input") { description = "" }

        Output(NUMERIC, "output"){ description = "" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Return a variable of all 0s, with the same shape as the input variable. Note that this is dynamic:
 if the input shape changes in later execution, the returned variable's shape will also be updated

 @param name  Name of the new SDVariable
 @param input Input SDVariable
 @return A new SDVariable with the same (dynamic) shape as the input
     
""".trimIndent()
        }
    }

    Op("any") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.bool"
        legacy = true
        Input(NUMERIC, "x") { description = " Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Boolean or array reduction operation, optionally along specified dimensions
            """.trimIndent()
        }
    }

    Op("all") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce.bool"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(INT, "dimensions"){ count = AtLeast(0); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
        Output(NUMERIC, "output"){ description = "reduced array of rank (input rank - num dimensions)" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Boolean and array reduction operation, optionally along specified dimensions
            """.trimIndent()
        }
    }
}

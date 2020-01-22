package org.nd4j.codegen.ops

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*



fun SDImage() =  Namespace("SDImage"){
    val namespaceJavaPackage = "org.nd4j.linalg.api.ops.custom"
    Op("CropAndResize") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.image"
        javaOpClass = "CropAndResize"
        Input(NUMERIC, "image") { description = "Input image, with shape [batch, height, width, channels]" }
        Input(NUMERIC, "cropBoxes") { description = "Float32 crop, shape [numBoxes, 4] with values in range 0 to 1" }
        Input(NUMERIC, "boxIndices") { description = "Indices: which image (index to dimension 0) the cropBoxes belong to. Rank 1, shape [numBoxes]" }
        Input(NUMERIC, "cropOutSize") { description = "Output size for the images - int32, rank 1 with values [outHeight, outWidth]" }
        Arg(NUMERIC, "extrapolationValue") { description = "Used for extrapolation, when applicable. 0.0 should be used for the default" }

        Output(NUMERIC, "output"){ description = "Cropped and resized images" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Given an input image and some crop boxes, extract out the image subsets and resize them to the specified size.

 @param image              Input image, with shape [batch, height, width, channels]
 @param cropBoxes          Float32 crop, shape [numBoxes, 4] with values in range 0 to 1
 @param boxIndices         Indices: which image (index to dimension 0) the cropBoxes belong to. Rank 1, shape [numBoxes]
 @param cropOutSize        Output size for the images - int32, rank 1 with values [outHeight, outWidth]
 @param extrapolationValue Used for extrapolation, when applicable. 0.0 should be used for the default
 @return Cropped and resized images
     
""".trimIndent()
        }
    }

    Op("extractImagePatches") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.image"
        javaOpClass = "ExtractImagePatches"
        Input(NUMERIC, "image") { description = "Input image to extract image patches from - shape [batch, height, width, channels]" }
        Arg(INT, "kSizes") { count = AtLeast(0); description = "Kernel size - size of the image patches, [height, width]" }
        Arg(INT, "strides") { count = AtLeast(0);description = "Stride in the input dimension for extracting image patches, [stride_height, stride_width]" }
        Arg(INT, "rates") { count = AtLeast(0); description = "Usually [1,1]. Equivalent to dilation rate in dilated convolutions - how far apart the output pixels\n" +
                "                 in the patches should be, in the input. A dilation of [a,b] means every {@code a}th pixel is taken\n" +
                "                 along the height/rows dimension, and every {@code b}th pixel is take along the width/columns dimension" }
        Arg(BOOL, "sameMode") { description = "Padding algorithm. If true: use Same padding" }

        Output(NUMERIC, "output"){ description = "The extracted image patches" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Given an input image, extract out image patches (of size kSizes - h x w) and place them in the depth dimension.

 @param image    Input image to extract image patches from - shape [batch, height, width, channels]
 @param kSizes   Kernel size - size of the image patches, [height, width]
 @param strides  Stride in the input dimension for extracting image patches, [stride_height, stride_width]
 @param rates    Usually [1,1]. Equivalent to dilation rate in dilated convolutions - how far apart the output pixels
                 in the patches should be, in the input. A dilation of [a,b] means every {@code a}th pixel is taken
                 along the height/rows dimension, and every {@code b}th pixel is take along the width/columns dimension
 @param sameMode Padding algorithm. If true: use Same padding
 @return The extracted image patches
     
""".trimIndent()
        }
    }

    Op("nonMaxSuppression") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.image"
        javaOpClass = "NonMaxSuppression"
        Input(NUMERIC, "boxes") { description = "Might be null. Name for the output variable" }
        Input(NUMERIC, "scores") { description = "vector of shape [num_boxes]" }
        Arg(INT, "maxOutSize") { description = "scalar representing the maximum number of boxes to be selected" }
        Arg(NUMERIC, "iouThreshold") { description = "threshold for deciding whether boxes overlap too much with respect to IOU" }
        Arg(NUMERIC, "scoreThreshold") { description = "threshold for deciding when to remove boxes based on score" }

        Output(NUMERIC, "output"){ description = "vectort of shape [M] representing the selected indices from the boxes tensor, where M <= max_output_size" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Greedily selects a subset of bounding boxes in descending order of score
 @param boxes    2D array of shape [num_boxes,4]
 @param scores   vector of shape [num_boxes]
 @param maxOutSize scalar representing the maximum number of boxes to be selected
 @param iouThreshold  float - threshold for deciding whether boxes overlap too much with respect to IOU
 @param scoreThreshold float - threshold for deciding when to remove boxes based on score
 @return vectort of shape [M] representing the selected indices from the boxes tensor, where M <= max_output_size
     
""".trimIndent()
        }
    }

    Op("adjustContrast") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "AdjustContrast"
        Input(NUMERIC, "in") { description = "images to adjust. 3D shape or higher" }
        Arg(FLOATING_POINT, "factor") { description = "multiplier for adjusting contrast" }
        Output(NUMERIC, "output"){ description = "Contrast-adjusted image" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Adjusts contrast of RGB or grayscale images.
 @param in images to adjust. 3D shape or higher.
 @param factor float multiplier for adjusting contrast.
 @return Contrast-adjusted image
     
""".trimIndent()
        }
    }

    Op("adjustSaturation") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "AdjustSaturation"
        Input(NUMERIC, "in") { description = "RGB image as 3D array" }
        Arg(FLOATING_POINT, "factor") { description = "factor for saturation" }

        Output(NUMERIC, "output"){ description = "adjusted image" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Adjust saturation of RGB images
 @param in RGB image as 3D array
 @param factor factor for saturation
 @return adjusted image
     
""".trimIndent()
        }
    }

    Op("adjustHue") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "AdjustHue"
        Input(NUMERIC, "in") { description = "image as 3D array" }
        Arg(NUMERIC, "delta") { description = "value to add to hue channel" }

        Output(NUMERIC, "output"){ description = "adjusted image" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Adjust hue of RGB image
 @param in RGB image as 3D array
 @param delta value to add to hue channel
 @return adjusted image
     
""".trimIndent()
        }
    }

    Op("randomCrop") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "RandomCrop"
        Input(NUMERIC, "input") { description = "input array" }
        Input(NUMERIC, "shape") { description = "shape for crop" }

        Output(NUMERIC, "output"){ description = "cropped array" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Randomly crops image
 @param input input array
 @param shape shape for crop
 @return cropped array
     
""".trimIndent()
        }
    }

    Op("rgbToHsv") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "RgbToHsv"
        Input(NUMERIC, "input") { description = "3D image" }

        Output(NUMERIC, "output"){ description = "3D image" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Converting array from HSV to RGB format
 @param input 3D image
 @return 3D image
     
""".trimIndent()
        }
    }

    Op("hsvToRgb") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "HsvToRgb"
        Input(NUMERIC, "input") { description = "3D image" }

        Output(NUMERIC, "output"){ description = "3D image" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Converting image from HSV to RGB format
 @param input 3D image
 @return 3D image
     
""".trimIndent()
        }
    }

    Op("rgbToYiq") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "RgbToYiq"
        Input(NUMERIC, "input") { description = "3D image" }

        Output(NUMERIC, "output"){ description = "3D image" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Converting array from RGB to YIQ format
 @param input 3D image
 @return 3D image
     
""".trimIndent()
        }
    }

    Op("yiqToRgb") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "YiqToRgb"
        Input(NUMERIC, "input") { description = "3D image" }

        Output(NUMERIC, "output"){ description = "3D image" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Converting image from YIQ to RGB format
 @param input 3D image
 @return 3D image
     
""".trimIndent()
        }
    }

    Op("rgbToYuv") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "RgbToYuv"

        Input(NUMERIC, "input") { description = "3D image" }

        Output(NUMERIC, "output"){ description = "3D image" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Converting array from RGB to YUV format
 @param input 3D image
 @return 3D image
     
""".trimIndent()
        }
    }

    Op("yuvToRgb") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "YuvToRgb"

        Input(NUMERIC, "input") { description = "3D image" }

        Output(NUMERIC, "output"){ description = "3D image" }

        Doc(Language.ANY, DocScope.ALL){
            """
 Converting image from YUV to RGB format
 @param input 3D image
 @return 3D image
     
""".trimIndent()
        }
    }
}
package org.nd4j.codegen.ir.onnx

import onnx.Onnx
import org.nd4j.shade.protobuf.ByteString

fun NodeProto(block: Onnx.NodeProto.Builder.() -> Unit): Onnx.NodeProto {
    return Onnx.NodeProto.newBuilder().apply(block).build()
}

fun AttributeProto(block: Onnx.AttributeProto.Builder.() -> Unit) : Onnx.AttributeProto {
    return Onnx.AttributeProto.newBuilder().apply { block }.build()
}

fun Onnx.NodeProto.Builder.Input(inputName: String) {
    this.addInput(inputName)
}

fun GraphProto(block: Onnx.GraphProto.Builder.() -> Unit): Onnx.GraphProto {
    return Onnx.GraphProto.newBuilder().apply(block).build()
}

fun Onnx.GraphProto.Builder.Node(inputNode: Onnx.NodeProto) {
    this.addNode(inputNode)
}

fun Onnx.AttributeProto.Builder.Tensor(inputTensor: Onnx.TensorProto) {
    this.addTensors(inputTensor)
}

fun OnnxTensorProto(block: Onnx.TensorProto.Builder.() -> Unit): Onnx.TensorProto {
    return Onnx.TensorProto.newBuilder().apply { block }.build()
}

fun Onnx.TensorProto.Builder.OnnxDataType(value: Onnx.TensorProto.DataType) {
    this.dataType = value.ordinal
}

fun Onnx.TensorProto.Builder.OnnxRawData(byteArray: ByteArray) {
    this.rawData = ByteString.copyFrom(byteArray)
}

fun Onnx.TensorProto.Builder.Shape(shape: List<Long>) {
    this.dimsList.clear()
    this.dimsList.addAll(shape)
}

fun Onnx.TensorProto.Builder.LongData(longData: List<Long>) {
    this.addAllInt64Data(longData)
}

fun Onnx.TensorProto.Builder.IntData(intData: List<Int>) {
    this.addAllInt32Data(intData)
}

fun Onnx.TensorProto.Builder.FloatData(floatData: List<Float>) {
    this.addAllFloatData(floatData)
}

fun Onnx.TensorProto.Builder.DoubleData(doubleData: List<Double>) {
    this.addAllDoubleData(doubleData)
}

fun Onnx.TensorProto.Builder.BoolData(boolData: List<Boolean>) {
    this.addAllInt32Data(boolData.map { input -> if(input) 1 else 0  })
}
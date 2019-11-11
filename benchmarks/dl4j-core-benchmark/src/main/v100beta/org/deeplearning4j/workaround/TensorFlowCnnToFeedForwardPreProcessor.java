package org.deeplearning4j.workaround;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Specialized CnnToFeedForwardInputPreProcessor for use with
 * Convolutional layers imported from Keras using the TensorFlow
 * backend.
 *
 * @author dave@skymind.io
 */
@Slf4j
public class TensorFlowCnnToFeedForwardPreProcessor extends CnnToFeedForwardPreProcessor {

    @JsonCreator
    public TensorFlowCnnToFeedForwardPreProcessor(@JsonProperty("inputHeight") long inputHeight,
                                                  @JsonProperty("inputWidth") long inputWidth,
                                                  @JsonProperty("numChannels") long numChannels) {
        super((int)inputHeight, (int)inputWidth, (int)numChannels);
    }

    public TensorFlowCnnToFeedForwardPreProcessor(long inputHeight, long inputWidth) {
        super((int)inputHeight, (int)inputWidth);
    }

    public TensorFlowCnnToFeedForwardPreProcessor() {
        super();
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (input.rank() == 2)
            return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input); //Should usually never happen
        /* DL4J convolutional input:       # channels, # rows, # cols
         * TensorFlow convolutional input: # rows, # cols, # channels
         * Theano convolutional input:     # channels, # rows, # cols
         */
        INDArray permuted = workspaceMgr.dup(ArrayType.ACTIVATIONS, input.permute(0, 2, 3, 1), 'c'); //To: [n, h, w, c]

        val inShape = input.shape(); //[miniBatch,depthOut,outH,outW]
        val outShape = new int[]{(int)inShape[0], (int)(inShape[1] * inShape[2] * inShape[3])};

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, permuted.reshape('c', outShape));
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        if (epsilons.ordering() != 'c' || !Shape.hasDefaultStridesForShape(epsilons))
            epsilons = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilons, 'c');

        INDArray epsilonsReshaped = epsilons.reshape('c', epsilons.size(0), inputHeight, inputWidth, numChannels);

        return workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilonsReshaped.permute(0, 3, 1, 2));    //To [n, c, h, w]
    }

    @Override
    public org.deeplearning4j.nn.modelimport.keras.preprocessors.TensorFlowCnnToFeedForwardPreProcessor clone() {
        return (org.deeplearning4j.nn.modelimport.keras.preprocessors.TensorFlowCnnToFeedForwardPreProcessor) super.clone();
    }
}

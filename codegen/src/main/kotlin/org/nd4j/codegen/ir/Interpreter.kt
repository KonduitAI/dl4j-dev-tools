package org.nd4j.codegen.ir

import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.gen.OpDeclarationDescriptor
import org.nd4j.linalg.api.ops.CustomOp
import org.nd4j.linalg.api.ops.CustomOpDescriptor

class Interpreter {


    fun buildGraph(opDescriptorList: List<OpDeclarationDescriptor>): SameDiff {
        val sameDiff = SameDiff.create()
        opDescriptorList.forEach {
            for(inputName in it.inArgNames) {

            }
            for(outputName in it.outArgNames) {

            }
        }

        //TODO: Graph structure?
        //TODO: Handle sameDiff.addArgsFor()
        //Could create empty ops?
        //Infer variable names from input/output tensors in the descriptor
        //how to track unique names? use internal samediff naming for that?
        //either way, need a unique name generator

        return sameDiff
    }


    fun convertToCustomOp(descriptor: OpDeclarationDescriptor): CustomOpDescriptor {
        /***
         * Load descriptor with particular values?
         */
        return CustomOpDescriptor.builder().build()
    }

}
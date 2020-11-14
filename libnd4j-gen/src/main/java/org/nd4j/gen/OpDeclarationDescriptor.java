/*******************************************************************************
 * Copyright (c) 2020 Konduit KK.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.gen;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.common.primitives.Pair;
import org.nd4j.ir.OpNamespace;
import org.nd4j.ir.MapperNamespace;
import org.nd4j.ir.TensorNamespace;

/**
 * The op descriptor for the libnd4j code base.
 * Each op represents a serialized version of
 * {@link org.nd4j.linalg.api.ops.DynamicCustomOp}
 * that has naming metadata attached.
 *
 * @author Adam Gibson
 */
@Data
@Builder(toBuilder = true)
public class OpDeclarationDescriptor implements Serializable  {
    private String name;
    private int nIn,nOut,tArgs,iArgs;
    private boolean inplaceAble;
    private List<String> inArgNames;
    private List<String> outArgNames;
    private List<String> tArgNames;
    private List<String> iArgNames;
    private List<String> bArgNames;

    private List<Integer> inArgIndices;
    private List<Integer> outArgIndices;
    private List<Integer> iArgIndices;
    private List<Integer> tArgIndices;
    private List<Integer> bArgIndices;

    private OpDeclarationType opDeclarationType;
    @Builder.Default
    private Map<String,Boolean> argOptional = new HashMap<>();


    public enum OpDeclarationType {
        CUSTOM_OP_IMPL,
        BOOLEAN_OP_IMPL,
        LIST_OP_IMPL,
        LOGIC_OP_IMPL,
        OP_IMPL,
        DIVERGENT_OP_IMPL,
        CONFIGURABLE_OP_IMPL,
        REDUCTION_OP_IMPL,
        BROADCASTABLE_OP_IMPL,
        BROADCASTABLE_BOOL_OP_IMPL,
        LEGACY_XYZ,
        PLATFORM_IMPL
    }




    /**
     * Get all the arguments names of this descriptor
     * by {@link OpNamespace.ArgDescriptor.ArgType}
     * @return the map of arg type
     */
    public Map<String, Pair<Integer,OpNamespace.ArgDescriptor.ArgType>> argsByType() {
        Map<String,Pair<Integer,OpNamespace.ArgDescriptor.ArgType>> argsByType = new HashMap<>();
        for(String s : inArgNames) {
            argsByType.put(s,Pair.of(inArgIndices.get(inArgNames.indexOf(s)),OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR));
        }
        for(String s : outArgNames) {
            argsByType.put(s,Pair.of(outArgIndices.get(outArgNames.indexOf(s)),OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR));
        }
        for(String s : iArgNames) {
            argsByType.put(s,Pair.of(iArgIndices.get(iArgNames.indexOf(s)),OpNamespace.ArgDescriptor.ArgType.INT64));
        }
        for(String s : tArgNames) {
            argsByType.put(s,Pair.of(tArgIndices.get(tArgNames.indexOf(s)),OpNamespace.ArgDescriptor.ArgType.FLOAT));
        }
        for(String s : bArgNames) {
            argsByType.put(s,Pair.of(bArgIndices.get(bArgNames.indexOf(s)),OpNamespace.ArgDescriptor.ArgType.BOOL));
        }

        return argsByType;
    }


    public void validate() {
        if(nIn >= 0 && nIn != inArgNames.size() && !isVariableInputSize()) {
            System.err.println("In arg names was not equal to number of inputs found for op " + name);
        }

        if(nOut >= 0 && nOut != outArgNames.size() && !isVariableOutputSize()) {
            System.err.println("Output arg names was not equal to number of outputs found for op " + name);
        }

        if(tArgs >= 0 && tArgs != tArgNames.size() && !isVariableTArgs()) {
            System.err.println("T arg names was not equal to number of T found for op " + name);
        }
        if(iArgs >= 0 && iArgs != iArgNames.size() && !isVariableIntArgs()) {
            System.err.println("Integer arg names was not equal to number of integer args found for op " + name);
        }
    }


    /**
     * Returns true if there is a variable number
     * of integer arguments for an op
     * @return
     */
    public boolean isVariableIntArgs() {
        return iArgs < 0;
    }

    /**
     * Returns true if there is a variable
     * number of t arguments for an op
     * @return
     */
    public boolean isVariableTArgs() {
        return tArgs < 0;
    }

    /**
     * Returns true if the number of outputs is variable size
     * @return
     */
    public boolean isVariableOutputSize() {
        return nOut < 0;
    }

    /**
     * Returns true if the number of
     * inputs is variable size
     * @return
     */
    public boolean isVariableInputSize() {
        return nIn < 0;
    }


}

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

import org.apache.commons.io.FileUtils;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.common.primitives.Counter;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.SetUtils;
import org.nd4j.gen.proposal.ArgDescriptorProposal;
import org.nd4j.gen.proposal.ArgDescriptorSource;
import org.nd4j.gen.proposal.impl.JavaSourceArgDescriptorSource;
import org.nd4j.gen.proposal.impl.Libnd4jArgDescriptorSource;
import org.nd4j.gen.proposal.utils.ArgDescriptorParserUtils;
import org.nd4j.ir.OpNamespace;
import org.nd4j.shade.protobuf.TextFormat;
import org.reflections.Reflections;

import java.io.File;
import java.nio.charset.Charset;
import java.util.*;
import java.util.stream.Collectors;


/**
 * Parses the libnd4j code base based on a relative path
 * default of ../deeplearning4j/libnd4j
 * or a passed in file path.
 * It generates a descriptor for each op.
 * The file properties can be found at {@link OpDeclarationDescriptor}
 *
 *
 * @author Adam Gibson
 */
public class ParseOpFile {


    public static void main(String...args) throws Exception {
        String libnd4jPath = args.length > 0 ? args[0] : Libnd4jArgDescriptorSource.DEFAULT_LIBND4J_DIRECTORY;
        String outputFilePath = args.length > 1 ? args[1] : ArgDescriptorParserUtils.DEFAULT_OUTPUT_FILE;

        File libnd4jRootDir = new File(libnd4jPath);
        StringBuilder nd4jApiSourceDir = new StringBuilder();
        nd4jApiSourceDir.append("nd4j");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("nd4j-backends");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("nd4j-api-parent");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("nd4j-api");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("src");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("main");
        nd4jApiSourceDir.append(File.separator);
        nd4jApiSourceDir.append("java");
        File nd4jApiRootDir = new File(new File(libnd4jPath).getParent(),nd4jApiSourceDir.toString());
        System.out.println("Parsing  libnd4j code base at " + libnd4jRootDir.getAbsolutePath() + " and writing to " + outputFilePath);
        List<OpDeclarationDescriptor> opDeclarationDescriptors = new ArrayList<>();
        Libnd4jArgDescriptorSource libnd4jArgDescriptorSource = Libnd4jArgDescriptorSource.builder()
                .libnd4jPath(libnd4jPath)
                .weight(999.0)
                .build();


    /*    Set<String> opsFoundInDeclarations = new HashSet<>();

        OpNamespace.OpDescriptorList.Builder listBuilder = OpNamespace.OpDescriptorList.newBuilder();
        for(OpDeclarationDescriptor declarationDescriptor : opDeclarationDescriptors) {
            Map<String, Pair<Integer, OpNamespace.ArgDescriptor.ArgType>> stringArgTypeMap = declarationDescriptor.argsByType();
            OpNamespace.OpDescriptor.Builder opDescriptorBuilder = OpNamespace.OpDescriptor.newBuilder();
            opDescriptorBuilder.setOpDeclarationType(OpNamespace.OpDescriptor.OpDeclarationType.values()[declarationDescriptor.getOpDeclarationType().ordinal()]);
            for(Map.Entry<String, Pair<Integer, OpNamespace.ArgDescriptor.ArgType>> argTypeEntry : stringArgTypeMap.entrySet()) {
                OpNamespace.ArgDescriptor argDescriptor = OpNamespace.ArgDescriptor.newBuilder()
                        .setArgType(argTypeEntry.getValue().getValue())
                        .setArgIndex(argTypeEntry.getValue().getKey())
                        .setName(argTypeEntry.getKey())
                        .build();
                opDescriptorBuilder.addArgDescriptor(argDescriptor);
            }

            opDescriptorBuilder.setName(declarationDescriptor.getName());
            OpNamespace.OpDescriptor opDescriptor = opDescriptorBuilder.build();
            listBuilder.addOpList(opDescriptor);
            System.out.println(declarationDescriptor);
            opsFoundInDeclarations.add(declarationDescriptor.getName());
        }
*/


        JavaSourceArgDescriptorSource javaSourceArgDescriptorSource = JavaSourceArgDescriptorSource.builder()
                .nd4jApiRootDir(nd4jApiRootDir)
                .weight(1.0)
                .build();

        Map<String,List<ArgDescriptorProposal>> proposals = new HashMap<>();
        for(ArgDescriptorSource argDescriptorSource : new ArgDescriptorSource[] {libnd4jArgDescriptorSource,javaSourceArgDescriptorSource}) {
            Map<String, List<ArgDescriptorProposal>> currProposals = argDescriptorSource.getProposals();
            for(Map.Entry<String,List<ArgDescriptorProposal>> entry : currProposals.entrySet()) {
                if(proposals.containsKey(entry.getKey())) {
                    List<ArgDescriptorProposal> currProposalsList = proposals.get(entry.getKey());
                    currProposalsList.addAll(entry.getValue());
                }
                else {
                    proposals.put(entry.getKey(),entry.getValue());
                }
            }
        }

        OpNamespace.OpDescriptorList.Builder listBuilder = OpNamespace.OpDescriptorList.newBuilder();
        for(Map.Entry<String,List<ArgDescriptorProposal>> proposal : proposals.entrySet()) {
            Map<String, List<ArgDescriptorProposal>> collect = proposal.getValue().stream()
                    .collect(Collectors.groupingBy(input -> input.getDescriptor().getName()));
            //merge boolean and int64
            collect.entrySet().forEach(entry -> {
                ArgDescriptorParserUtils.standardizeTypes(entry.getValue());
            });

            Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>, OpNamespace.ArgDescriptor> rankedProposals = ArgDescriptorParserUtils.
                    standardizeNames(collect);
            OpNamespace.OpDescriptor.Builder opDescriptorBuilder = OpNamespace.OpDescriptor.newBuilder()
                    .setName(proposal.getKey());
            rankedProposals.entrySet().stream().map(input -> input.getValue())
                    .forEach(argDescriptor -> {
                        opDescriptorBuilder.addArgDescriptor(argDescriptor);
                    });

            listBuilder.addOpList(opDescriptorBuilder.build());

        }


        String write = TextFormat.printToString(listBuilder.build());
        FileUtils.writeStringToFile(new File(outputFilePath),write, Charset.defaultCharset());



/*

        //merge op descriptors from multiple names
        OpNamespace.OpDescriptorList.Builder retBuilder = OpNamespace.OpDescriptorList.newBuilder();
        //build list for sorting before final addition in post processing
        List<OpNamespace.OpDescriptor> opDescriptorsToSort = new ArrayList<>();
        Map<String, List<OpNamespace.OpDescriptor>> groupedByName = ret.getOpListList().stream()
                .collect(Collectors.groupingBy(input -> input.getName()));
        groupedByName.forEach((k,v) -> {
            List<OpNamespace.ArgDescriptor> collect = v.stream()
                    .flatMap(input -> input.getArgDescriptorList().stream())
                    .collect(Collectors.toList());
            Collections.sort(collect,Comparator.comparing(OpNamespace.ArgDescriptor::getArgType));
            Set<OpNamespace.ArgDescriptor> iterateOver = new LinkedHashSet<>(collect);
            Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>, OpNamespace.ArgDescriptor> argDescriptorByIndex = new HashMap<>();
            List<OpNamespace.ArgDescriptor> namesNotIncluded = new ArrayList<>();
            int numNegative = 1;
            for(OpNamespace.ArgDescriptor argDescriptor : iterateOver) {
                if(ArgDescriptorParserUtils.fieldNameFiltersDynamicCustomOps.contains(argDescriptor.getName())) {
                    //ban on certain field names
                    continue;
                }
                if(!argDescriptorByIndex.containsKey(Pair.of(argDescriptor.getArgIndex(),argDescriptor.getArgType())) || argDescriptor.getArgIndex() < 0) {
                    if(argDescriptor.getArgIndex() < 0) {
                        argDescriptorByIndex.put(Pair.of(-numNegative, argDescriptor.getArgType()), argDescriptor);
                        numNegative++;
                    }
                    else {
                        argDescriptorByIndex.put(Pair.of(argDescriptor.getArgIndex(), argDescriptor.getArgType()), argDescriptor);

                    }
                }
                else if(argDescriptor.getArgIndex() >= 0 && argDescriptorByIndex.containsKey(Pair.of(argDescriptor.getArgIndex(),argDescriptor.getArgType()))) {
                    //merge old and new in to new one
                    OpNamespace.ArgDescriptor old = argDescriptorByIndex.get(Pair.of(argDescriptor.getArgIndex(),argDescriptor.getArgType()));
                    OpNamespace.ArgDescriptor argDescriptor1 = ArgDescriptorParserUtils.mergeDescriptorsOfSameIndex(old, argDescriptor);
                    if(argDescriptor1.getName().equals("output") || argDescriptor1.getName().equals("z") && argDescriptor1.getArgType() == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR) {
                        throw new IllegalStateException("Output tensor sent in as input tensor");
                    }
                    for(OpNamespace.ArgDescriptor value : argDescriptorByIndex.values()) {
                        if(argDescriptor1.getName().equals(value.getName())) {
                            System.err.println("Found arg name of " + argDescriptor1.getName() + " with arg type " + value.getArgType() + " with duplicate index ");
                            continue;
                        }
                    }
                    argDescriptorByIndex.put(Pair.of(argDescriptor1.getArgIndex(),argDescriptor1.getArgType()),argDescriptor1);
                }
            }

            */
/**
 * TODO: Handle same name/same type/different indices here.
 * Example op to look into:  ParallelConcat
 *
 * Potentially look in to java and see when indices are different
 * and prefer the c++ source?
 *
 * Could also check if already exists and see which one is verifiable.
 *
 * TODO: Look at batchnorm. C++ and java code both generate invalid identifiers.
 * A few issues:
 *
 * 1. Valid parameter names are captured, but then used within method invocations for conversions.
 * Maybe what we could do is strip out the names from the parameters in post processing.
 * The same thing seems to happen in c++
 *
 * 2. For C++, an int parameter is invoked adding to a collection, maybe we could parse the
 * variable name of the collection?
 *
 * 3. Maybe create a set of merge rules for known variable names to reduce redundancies?
 * Another thing we could do is get a list of type/name combinations from op constructors
 * where available to get a hint as to what types things should be and used for clarification?
 *
 * 4. Bit of a stretch: Shortest edit  distance based on constructor names to replace
 * embedded variable names found? Could also replace field names with constructor names?
 *
 * 5. Strip out punctuation an invalid identifiers.
 *
 * 6. Ensure indices aren't < 0. Validate this when parsing and adding parameters.
 *
 * 7. Double check parameter names vs ops in both the tf import and onnx import.


 Output tensors use input tensor indexing causing indexing issues
 when used with the arg descriptor.

 *//*


            OpNamespace.OpDescriptor.Builder newBuilder = OpNamespace.OpDescriptor.newBuilder();
            //ensure name and type are set
            newBuilder.setName(v.get(0).getName());
            newBuilder.setOpDeclarationType(v.get(0).getOpDeclarationType());
            final List<OpNamespace.ArgDescriptor> valuesToAdd = new ArrayList<>();

            argDescriptorByIndex.entrySet().forEach(entry -> {
                valuesToAdd.add(entry.getValue());
            });

            Collections.sort(valuesToAdd,Comparator.comparing(OpNamespace.ArgDescriptor::getArgIndex));
            List<OpNamespace.ArgDescriptor> newValuesToAdd = new ArrayList<>(valuesToAdd.size());
            Set<Pair<OpNamespace.ArgDescriptor.ArgType,String>> argTypeNameCheckSet = new HashSet<>();
            Map<OpNamespace.ArgDescriptor.ArgType,Integer> typeToIndex = new HashMap<>();
            //ensure arg indices are consistent after post processing
            for(int i = 0; i < valuesToAdd.size(); i++) {
                if(!ArgDescriptorParserUtils.isValidIdentifier(valuesToAdd.get(i).getName()))
                    continue;
                //arg name and type already found, just continue
                if(argTypeNameCheckSet.contains(Pair.of(valuesToAdd.get(i).getArgType(),valuesToAdd.get(i).getName())))
                    continue;
                Integer currTypeIndex = null;
                if(!typeToIndex.containsKey(valuesToAdd.get(i).getArgType())) {
                    currTypeIndex = 0;
                    typeToIndex.put(valuesToAdd.get(i).getArgType(),currTypeIndex);
                }
                else {
                    currTypeIndex = typeToIndex.get(valuesToAdd.get(i).getArgType());
                }

                OpNamespace.ArgDescriptor argDescriptor = OpNamespace.ArgDescriptor.newBuilder()
                        .setName(valuesToAdd.get(i).getName())
                        .setArgType(valuesToAdd.get(i).getArgType())
                        .setDataTypeValue(valuesToAdd.get(i).getDataTypeValue())
                        .setArgIndex(currTypeIndex)
                        .setArgOptional(false)
                        .build();

                //update type index for next addition
                currTypeIndex++;
                typeToIndex.put(valuesToAdd.get(i).getArgType(),currTypeIndex);

                argTypeNameCheckSet.add(Pair.of(valuesToAdd.get(i).getArgType(),valuesToAdd.get(i).getName()));
                newValuesToAdd.add(argDescriptor);
            }



            List<OpNamespace.ArgDescriptor> postProcessedAdd = new ArrayList<>();
            for(int i = 0; i < newValuesToAdd.size(); i++) {
                if( ArgDescriptorParserUtils.argsListContainsEquivalentAttribute(postProcessedAdd,newValuesToAdd.get(i))) {
                    continue;
                }

                postProcessedAdd.add(newValuesToAdd.get(i));

            }

            //update values reference and add in final post processed values
            postProcessedAdd.stream().forEach(input -> {
                newBuilder.addArgDescriptor(input);
            });


            opDescriptorsToSort.add(newBuilder.build());
        });


        Collections.sort(opDescriptorsToSort, Comparator.comparing(OpNamespace.OpDescriptor::getName));
        Counter<String> opNameCounter = new Counter<>();
        opDescriptorsToSort.stream().forEach(opDescriptor ->  {
            opNameCounter.incrementCount(opDescriptor.getName(),1.0);
        });

        opNameCounter.dropElementsBelowThreshold(2.0);
        if(!opNameCounter.isEmpty()) {
            throw new IllegalStateException("Ops found with duplicate names/mappings: " + opNameCounter.keySet());
        }

        opDescriptorsToSort.forEach(input -> {
            retBuilder.addOpList(input);
        });


        //TODO: Detect multiple of same op. reverse seems to have multiples with incomplete mappings
        //as well as a complete one


        ret = retBuilder.build();
        String write = TextFormat.printToString(ret);
        FileUtils.writeStringToFile(outputFile,write, Charset.defaultCharset());
*/

        //System.out.println("Number of op descriptors " + opDeclarationDescriptors);
    }


}

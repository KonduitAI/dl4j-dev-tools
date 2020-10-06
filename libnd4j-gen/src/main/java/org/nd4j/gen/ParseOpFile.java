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

import org.nd4j.common.util.SetUtils;
import org.nd4j.linalg.api.ops.CustomOpDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.util.*;

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

    public final static String CUSTOM_OP_IMPL = "CUSTOM_OP_IMPL";
    public final static String BOOLEAN_OP_IMPL = "BOOLEAN_OP_IMPL";
    public final static String LIST_OP_IMPL = "LIST_OP_IMPL";
    public final static String LOGIC_OP_IMPL = "LOGIC_OP_IMPL";
    public final static String OP_IMPL = "OP_IMPL";
    public final static String DIVERGENT_OP_IMPL = "DIVERGENT_OP_IMPL";
    public final static String CONFIGURABLE_OP_IMPL = "CONFIGURABLE_OP_IMPL";
    public final static String REDUCTION_OP_IMPL = "REDUCTION_OP_IMPL";
    public final static String BROADCASTABLE_OP_IMPL = "BROADCASTABLE_OP_IMPL";
    public final static String BROADCASTABLE_BOOL_OP_IMPL = "BROADCASTABLE_BOOL_OP_IMPL";

    public final static String RETURN = "return";
    public final static String INT_ARG = "INT_ARG";
    public final static String I_ARG = "I_ARG";
    public final static String INPUT_VARIABLE = "INPUT_VARIABLE";
    public final static String OUTPUT_VARIABLE = "OUTPUT_VARIABLE";
    public final static String OUTPUT_NULLIFIED = "OUTPUT_NULLIFIED";
    public final static String T_ARG = "T_ARG";
    public final static String B_ARG = "B_ARG";
    public final static String DECLARE_SYN = "DECLARE_SYN";

    public final static String DEFAULT_LIBND4J_DIRECTORY = "../deeplearning4j/libnd4j";


    public static void main(String...args) throws Exception {
        String libnd4jPath = args.length > 0 ? args[0] : DEFAULT_LIBND4J_DIRECTORY;
        File libnd4jRootDir = new File(libnd4jPath);
        System.out.println("Parsing  libnd4j code base at " + libnd4jRootDir.getAbsolutePath());
        List<OpDeclarationDescriptor> opDeclarationDescriptors = new ArrayList<>();
        Map<String, OpDeclarationDescriptor> descriptorMap = new HashMap<>();

        Map<String, CustomOpDescriptor> customOperations = Nd4j.getExecutioner().getCustomOperations();

        Files.walk(libnd4jRootDir.toPath(), new FileVisitOption[]{
                FileVisitOption.FOLLOW_LINKS
        }).filter(path -> path.toFile().getAbsolutePath().endsWith(".cpp")).forEach(path -> {
            try {
                List<String> lines = Files.readAllLines(path);
                boolean inOpBlock = false;
                boolean foundOp = false;
                boolean oneLineOp = false;
                List<String> inArgNames = new ArrayList<>();
                List<String> outArgNames = new ArrayList<>();
                List<String> tArgNames = new ArrayList<>();
                List<String> iArgNames = new ArrayList<>();
                List<String> bArgNames = new ArrayList<>();
                OpDeclarationDescriptor opDeclarationDescriptor = null;
                OpDeclarationDescriptor.OpDeclarationDescriptorBuilder builder = OpDeclarationDescriptor.builder();
                for (String line : lines) {
                    if(line.isEmpty())
                        continue;
                    if (line.contains(CUSTOM_OP_IMPL)) {
                        // CUSTOM_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                        foundOp = true;
                        line = removeBracesFromDeclarationMacro(line,CUSTOM_OP_IMPL);
                        String[] split = line.trim().split(",");
                        String name = split[0];

                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                        int tArgs = Integer.parseInt(split[4].trim());
                        int iArgs = Integer.parseInt(split[5].trim());
                        builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.CUSTOM_OP_IMPL)
                                .nIn(nIn).nOut(nOut)
                                .inplaceAble(inplaceAble)
                                .iArgs(iArgs).tArgs(tArgs);

                        inOpBlock = true;

                    } else if(line.contains(BOOLEAN_OP_IMPL)) {
                        // BOOLEAN_OP_IMPL(NAME, NIN, SCALAR)
                        foundOp = true;
                        if(line.contains(");")) {
                            oneLineOp = true;
                        }

                        line = removeBracesFromDeclarationMacro(line,BOOLEAN_OP_IMPL);

                        String[] split = line.trim().split(",");
                        String name = split[0];

                        // BOOLEAN_OP_IMPL(NAME, NIN, SCALAR)
                        int nIn = Integer.parseInt(split[1].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[2].trim());
                        builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.BOOLEAN_OP_IMPL)
                                .nIn(nIn)
                                .inplaceAble(inplaceAble);

                        inOpBlock = true;
                    } else if(line.contains(LIST_OP_IMPL)) {
                        // LIST_OP_IMPL(NAME, NIN, NOUT, TARGS, IARGS)
                        foundOp = true;
                        line = removeBracesFromDeclarationMacro(line,LIST_OP_IMPL);

                        String[] split = line.trim().split(",");
                        String name = split[0];

                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        int tArgs = Integer.parseInt(split[3].trim());
                        int iArgs = Integer.parseInt(split[4].trim());
                        builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.LIST_OP_IMPL)
                                .nIn(nIn).nOut(nOut)
                                .iArgs(iArgs).tArgs(tArgs);

                        inOpBlock = true;
                    } else if(line.contains(LOGIC_OP_IMPL)) {
                        // LOGIC_OP_IMPL(NAME)
                        foundOp = true;
                        if(line.contains(");"))
                            oneLineOp = true;
                        line = removeBracesFromDeclarationMacro(line,LOGIC_OP_IMPL);

                        String[] split = line.trim().split(",");
                        String name = split[0];

                        builder.name(name)
                                .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.LOGIC_OP_IMPL);

                        inOpBlock = true;
                    } else if(line.contains(DIVERGENT_OP_IMPL)) {
                        foundOp = true;
                        //DIVERGENT_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE)
                        line = removeBracesFromDeclarationMacro(line,DIVERGENT_OP_IMPL);
                        String[] split = line.trim().split(",");
                        String name = split[0];

                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                        builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.DIVERGENT_OP_IMPL)
                                .nIn(nIn).nOut(nOut)
                                .inplaceAble(inplaceAble);

                        inOpBlock = true;
                    } else if(line.contains(CONFIGURABLE_OP_IMPL)) {
                        // CONFIGURABLE_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                        foundOp = true;
                        line = removeBracesFromDeclarationMacro(line,CONFIGURABLE_OP_IMPL);
                        String[] split = line.trim().split(",");
                        String name = split[0];

                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                        int tArgs = Integer.parseInt(split[4].trim());
                        int iArgs = Integer.parseInt(split[5].trim());
                        builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.CONFIGURABLE_OP_IMPL)
                                .nIn(nIn).nOut(nOut)
                                .inplaceAble(inplaceAble)
                                .iArgs(iArgs).tArgs(tArgs);

                        inOpBlock = true;
                    } else if(line.contains(REDUCTION_OP_IMPL)) {
                        //REDUCTION_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                        foundOp = true;
                        line = removeBracesFromDeclarationMacro(line,REDUCTION_OP_IMPL);
                        String[] split = line.trim().split(",");
                        String name = split[0];

                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                        int tArgs = Integer.parseInt(split[4].trim());
                        int iArgs = Integer.parseInt(split[5].trim());
                        builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.REDUCTION_OP_IMPL)
                                .nIn(nIn).nOut(nOut)
                                .inplaceAble(inplaceAble)
                                .iArgs(iArgs).tArgs(tArgs);

                        inOpBlock = true;
                    } else if(line.contains(BROADCASTABLE_OP_IMPL)) {
                        //BROADCASTABLE_OP_IMPL(NAME, TARGS, IARGS)
                        foundOp = true;
                        line = removeBracesFromDeclarationMacro(line,BROADCASTABLE_OP_IMPL);

                        String[] split = line.trim().split(",");
                        String name = split[0];

                        int tArgs = Integer.parseInt(split[1].trim());
                        int iArgs = Integer.parseInt(split[2].trim());
                        builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.BROADCASTABLE_OP_IMPL)
                                .iArgs(iArgs).tArgs(tArgs);

                        inOpBlock = true;
                    } else if(line.contains(BROADCASTABLE_BOOL_OP_IMPL)) {
                        //BROADCASTABLE_BOOL_OP_IMPL(NAME, TARGS, IARGS)
                        foundOp = true;
                        line = line.replace(BROADCASTABLE_BOOL_OP_IMPL + "(", "");
                        line = line.replace(")", "");
                        line = line.replace("{", "");
                        String[] split = line.trim().split(",");
                        String name = split[0];

                        int tArgs = Integer.parseInt(split[1].trim());
                        int iArgs = Integer.parseInt(split[2].trim());
                        builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.BROADCASTABLE_BOOL_OP_IMPL)
                                .iArgs(iArgs).tArgs(tArgs);

                        inOpBlock = true;
                    }  else if(line.contains(OP_IMPL)) {
                        //OP_IMPL(NAME, NIN, NOUT, INPLACEABLE)
                        foundOp = true;
                        line = removeBracesFromDeclarationMacro(line,OP_IMPL);
                        String[] split = line.trim().split(",");
                        String name = split[0];
                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                        builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.OP_IMPL)
                                .nIn(nIn).nOut(nOut)
                                .inplaceAble(inplaceAble);

                        inOpBlock = true;
                    }

                    line = line.trim();

                    //reset just in case we encounter another op in the file
                    if (inOpBlock && line.contains(RETURN) || oneLineOp) {
                        //reset op after 1 is found and current code block ends
                        if (foundOp) {
                            builder.inArgNames(inArgNames);
                            builder.outArgNames(outArgNames);
                            builder.tArgNames(tArgNames);
                            builder.iArgNames(iArgNames);
                            builder.bArgNames(bArgNames);

                            opDeclarationDescriptor = builder.build();
                            System.out.println(opDeclarationDescriptor);

                            opDeclarationDescriptors.add(opDeclarationDescriptor);

                            if (opDeclarationDescriptor != null) {
                                System.out.println("Op descriptor " + opDeclarationDescriptor);
                                System.out.println("Input arg name " + inArgNames);
                                System.out.println("Output arg names " + outArgNames);
                                System.out.println("T Arg names " + tArgNames);
                                System.out.println("Integer arg names " + iArgNames);
                                System.out.println("Boolean arg names " + bArgNames);
                                opDeclarationDescriptor.validate();
                            }
                        }

                        descriptorMap.put(opDeclarationDescriptor.getName(), opDeclarationDescriptor);

                        inOpBlock = false;
                        foundOp = false;
                        oneLineOp = false;
                        opDeclarationDescriptor = null;
                        builder = OpDeclarationDescriptor.builder();
                        //clear list references
                        inArgNames = new ArrayList<>();
                        outArgNames = new ArrayList<>();
                        tArgNames = new ArrayList<>();
                        iArgNames = new ArrayList<>();
                        bArgNames = new ArrayList<>();
                    }

                    if (inOpBlock) {
                        if (line.isEmpty()) {
                            //ignore
                        } else if (line.contains(INT_ARG) || line.contains(I_ARG)) {
                            addNameToList(line, iArgNames);
                        } else if (line.contains(OUTPUT_NULLIFIED) || line.contains(OUTPUT_VARIABLE)) {
                            addNameToList(line, outArgNames);
                        } else if (line.contains(T_ARG)) {
                            addNameToList(line, tArgNames);
                        } else if (line.contains(INPUT_VARIABLE)) {
                            addNameToList(line, inArgNames);
                        } else if (line.contains(B_ARG)) {
                            addNameToList(line, bArgNames);
                        }
                    }

                    //add alias descriptors
                    if (line.contains(DECLARE_SYN)) {
                        line = removeBracesFromDeclarationMacro(line,DECLARE_SYN);
                        String[] args2 = line.split(",");
                        String aliasFor = args2[1].trim();
                        String newKey = args2[0].trim();
                        if(descriptorMap.isEmpty()) {
                            throw new IllegalStateException("Descriptor map should not be empty here");
                        }

                        OpDeclarationDescriptor.OpDeclarationDescriptorBuilder opDescriptor2 = descriptorMap.get(aliasFor).toBuilder();
                        opDescriptor2.name(newKey);
                        OpDeclarationDescriptor newDescriptor = opDescriptor2.build();
                        opDeclarationDescriptors.add(newDescriptor);
                        descriptorMap.put(args2[1],newDescriptor);
                    }
                }


            } catch (IOException e) {
                e.printStackTrace();
            }
        });

        Set<String> opNamesForCompare = new HashSet<>(customOperations.keySet());
        Set<String> opsFoundInDeclarations = new HashSet<>();

        for(OpDeclarationDescriptor declarationDescriptor : opDeclarationDescriptors) {
            System.out.println(declarationDescriptor);
            opsFoundInDeclarations.add(declarationDescriptor.getName());
        }

        Set<String> differences = SetUtils.difference(opsFoundInDeclarations,opNamesForCompare);
        if(!differences.isEmpty()) {
            System.out.println("Differences found in declarations vs registered ops " + differences);
        }

        //System.out.println("Number of op descriptors " + opDeclarationDescriptors);
    }

    public static String removeBracesFromDeclarationMacro(String line,String nameOfMacro) {
        line = line.replace(nameOfMacro + "(", "");
        line = line.replace(")", "");
        line = line.replace("{", "");
        line = line.replace(";","");
        return line;
    }

    public static void addNameToList(String line,List<String> list) {
        String[] split = line.split(" = ");
        String[] arrSplit = split[0].split(" ");
        //type + name
        String name = arrSplit[arrSplit.length - 1];
        if(!list.contains(name))
            list.add(name);
    }

}

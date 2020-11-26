package org.nd4j.gen.proposal.impl;

import lombok.Builder;
import lombok.SneakyThrows;
import org.nd4j.common.base.Preconditions;
import org.nd4j.gen.OpDeclarationDescriptor;
import org.nd4j.gen.proposal.ArgDescriptorProposal;
import org.nd4j.gen.proposal.ArgDescriptorSource;
import org.nd4j.ir.OpNamespace;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import static org.nd4j.gen.proposal.impl.ArgDescriptorParserUtils.*;


public class Libnd4jArgDescriptorSource implements ArgDescriptorSource {

    private String libnd4jPath;
    private File libnd4jRootDir;
    private double weight;

    public final static String OP_IMPL = "OP_IMPL";
    public final static String DIVERGENT_OP_IMPL = "DIVERGENT_OP_IMPL";
    public final static String CONFIGURABLE_OP_IMPL = "CONFIGURABLE_OP_IMPL";
    public final static String REDUCTION_OP_IMPL = "REDUCTION_OP_IMPL";
    public final static String BROADCASTABLE_OP_IMPL = "BROADCASTABLE_OP_IMPL";
    public final static String BROADCASTABLE_BOOL_OP_IMPL = "BROADCASTABLE_BOOL_OP_IMPL";
    public final static String PLATFORM_IMPL = "PLATFORM_IMPL";
    public final static String RETURN = "return";
    public final static String INT_ARG = "INT_ARG";
    public final static String I_ARG = "I_ARG";
    public final static String INPUT_VARIABLE = "INPUT_VARIABLE";
    public final static String OUTPUT_VARIABLE = "OUTPUT_VARIABLE";
    public final static String OUTPUT_NULLIFIED = "OUTPUT_NULLIFIED";
    public final static String INPUT_LIST = "INPUT_LIST";
    public final static String T_ARG = "T_ARG";
    public final static String B_ARG = "B_ARG";
    public final static String DECLARE_SYN = "DECLARE_SYN";
    public final static String DEFAULT_LIBND4J_DIRECTORY = "../deeplearning4j/libnd4j";
    public final static int BROADCASTABLE_OP_IMPL_DEFAULT_NIN = 2;
    public final static int BROADCASTABLE_OP_IMPL_DEFAULT_NOUT = 1;
    public final static String CUSTOM_OP_IMPL = "CUSTOM_OP_IMPL";
    public final static String BOOLEAN_OP_IMPL = "BOOLEAN_OP_IMPL";
    public final static String LIST_OP_IMPL = "LIST_OP_IMPL";
    public final static String LOGIC_OP_IMPL = "LOGIC_OP_IMPL";
    //note this allows either a declaration like: auto variableNum = SOME_DECLARATION(0); or auto variableNum = SOME_DECLARATION(0) == 1;
    public final static String ARG_DECLARATION = "(\\w+\\s)+\\w+\\s*=\\s*[A-Z]+_[A-Z]+\\(\\d+\\);";
    public final static String ARG_BOOL_EQUALS_DECLARATION = "(\\w+\\s)+\\w+\\s*=\\s*[A-Z]+_[A-Z]+\\(\\d+\\)\\s*==\\s*\\d+;";
    public final static String ARG_DECLARATION_WITH_VARIABLE = "(\\w+\\s)+\\w+\\s*=\\s*[A-Z]+_[A-Z]+\\([\\d\\w\\+-*\\/]+);";
    public final static String ARRAY_ASSIGNMENT = "\\w+\\[[\\w\\d]\\]\\s*=\\s*[A-Z]+_[A-Z]+\\s*\\([\\w\\d\\+\\-\\*\\/\\s]+\\);";

    @Builder.Default
    private Map<String, OpNamespace.OpDescriptor.OpDeclarationType> opTypes = new HashMap<>();

    @Builder
    public Libnd4jArgDescriptorSource(String libnd4jPath,double weight) {
        if(libnd4jPath == null)
            libnd4jPath = "../deeplearning4j/libnd4j";
        if(weight == 0)
            weight = 999;
        this.weight = weight;
        libnd4jRootDir = new File(libnd4jPath);
    }



    @SneakyThrows
    public Map<String, List<ArgDescriptorProposal>> doExtractArgDescriptors() {
        Map<String, List<ArgDescriptorProposal>> ret = new HashMap<>();
        List<OpDeclarationDescriptor> opDeclarationDescriptors = new ArrayList<>();
        Map<String,OpDeclarationDescriptor> descriptorMap = new HashMap<>();
        //only include/ops the include directory, otherwise other misc folders get scanned
        Files.walk(new File(libnd4jRootDir,"include/ops").toPath(), new FileVisitOption[]{
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
                List<Integer> inArgIndices = new ArrayList<>();
                List<Integer> outArgIndices = new ArrayList<>();
                List<Integer> tArgIndices = new ArrayList<>();
                List<Integer> iArgIndices = new ArrayList<>();
                List<Integer> bArgIndices = new ArrayList<>();

                OpDeclarationDescriptor opDeclarationDescriptor = null;
                OpDeclarationDescriptor.OpDeclarationDescriptorBuilder builder = OpDeclarationDescriptor.builder();
                int currentOpNin = -1,currentOpNout = -1,currentOpIntArgs = -1,currentOutTArgs = -1, currentOpBooleanArgs = -1;
                boolean hasNin = false,hasNout = false,hasIntArgs = false,hasTArgs = false,platformImpl = false;
                List<ArgDescriptorProposal> argDescriptorProposals = null;
                int currLineIdx = 0;
                String name = null;
                for (String line : lines) {
                    if(line.trim().isEmpty() || line.trim().startsWith("//") || line.trim().length() == 1 || line.trim().isEmpty()) {
                        currLineIdx++;
                        continue;
                    }

                    if(!inOpBlock) {
                        if (line.contains(CUSTOM_OP_IMPL)) {
                            // CUSTOM_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, CUSTOM_OP_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.CUSTOM_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                            int tArgs = Integer.parseInt(split[4].trim());
                            int iArgs = Integer.parseInt(split[5].trim());

                            currentOpIntArgs = iArgs;
                            currentOutTArgs = tArgs;
                            hasIntArgs = true;
                            hasTArgs = true;

                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.CUSTOM_OP_IMPL)
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

                            line = removeBracesFromDeclarationMacro(line, BOOLEAN_OP_IMPL);

                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.BOOLEAN_OP_IMPL);

                            // BOOLEAN_OP_IMPL(NAME, NIN, SCALAR)
                            int nIn = Integer.parseInt(split[1].trim());
                            currentOpNin = nIn;
                            hasNin = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[2].trim());
                            builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.BOOLEAN_OP_IMPL)
                                    .nIn(nIn)
                                    .inplaceAble(inplaceAble);

                            inOpBlock = true;
                        } else if(line.contains(LIST_OP_IMPL)) {
                            // LIST_OP_IMPL(NAME, NIN, NOUT, TARGS, IARGS)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, LIST_OP_IMPL);

                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.LIST_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            int tArgs = Integer.parseInt(split[3].trim());
                            int iArgs = Integer.parseInt(split[4].trim());

                            currentOpIntArgs = iArgs;
                            currentOutTArgs = tArgs;
                            hasIntArgs = true;
                            hasTArgs = true;

                            builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.LIST_OP_IMPL)
                                    .nIn(nIn).nOut(nOut)
                                    .iArgs(iArgs).tArgs(tArgs);

                            inOpBlock = true;
                        } else if(line.contains(LOGIC_OP_IMPL)) {
                            // LOGIC_OP_IMPL(NAME)
                            foundOp = true;
                            if(line.contains(");"))
                                oneLineOp = true;
                            line = removeBracesFromDeclarationMacro(line, LOGIC_OP_IMPL);

                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.LOGIC_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.LOGIC_OP_IMPL);

                            inOpBlock = true;
                        } else if(line.contains(DIVERGENT_OP_IMPL)) {
                            foundOp = true;
                            //DIVERGENT_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE)
                            line = removeBracesFromDeclarationMacro(line, DIVERGENT_OP_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.DIVERGENT_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                            builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.DIVERGENT_OP_IMPL)
                                    .nIn(nIn).nOut(nOut)
                                    .inplaceAble(inplaceAble);

                            inOpBlock = true;
                        } else if(line.contains(CONFIGURABLE_OP_IMPL)) {
                            // CONFIGURABLE_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, CONFIGURABLE_OP_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.CONFIGURABLE_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                            int tArgs = Integer.parseInt(split[4].trim());
                            int iArgs = Integer.parseInt(split[5].trim());

                            hasIntArgs = true;
                            hasTArgs = true;

                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.CONFIGURABLE_OP_IMPL)
                                    .nIn(nIn).nOut(nOut)
                                    .inplaceAble(inplaceAble)
                                    .iArgs(iArgs).tArgs(tArgs);

                            inOpBlock = true;
                        } else if(line.contains(REDUCTION_OP_IMPL)) {
                            //REDUCTION_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, REDUCTION_OP_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.REDUCTION_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);

                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                            int tArgs = Integer.parseInt(split[4].trim());
                            int iArgs = Integer.parseInt(split[5].trim());

                            hasIntArgs = true;
                            hasTArgs = true;

                            builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.REDUCTION_OP_IMPL)
                                    .nIn(nIn).nOut(nOut)
                                    .inplaceAble(inplaceAble)
                                    .iArgs(iArgs).tArgs(tArgs);

                            inOpBlock = true;
                        } else if(line.contains(BROADCASTABLE_OP_IMPL)) {
                            //BROADCASTABLE_OP_IMPL(NAME, TARGS, IARGS)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, BROADCASTABLE_OP_IMPL);

                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.BROADCASTABLE_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int tArgs = Integer.parseInt(split[1].trim());
                            int iArgs = Integer.parseInt(split[2].trim());

                            hasTArgs = true;
                            hasIntArgs = true;

                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.BROADCASTABLE_OP_IMPL)
                                    .nIn(BROADCASTABLE_OP_IMPL_DEFAULT_NIN)
                                    .nOut(BROADCASTABLE_OP_IMPL_DEFAULT_NOUT)
                                    .iArgs(iArgs).tArgs(tArgs);

                            inOpBlock = true;
                        } else if(line.contains(BROADCASTABLE_BOOL_OP_IMPL)) {
                            //BROADCASTABLE_BOOL_OP_IMPL(NAME, TARGS, IARGS)
                            foundOp = true;
                            line = line.replace(BROADCASTABLE_BOOL_OP_IMPL + "(", "");
                            line = line.replace(")", "");
                            line = line.replace("{", "");
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.BROADCASTABLE_BOOL_OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int tArgs = Integer.parseInt(split[1].trim());
                            int iArgs = Integer.parseInt(split[2].trim());

                            currentOpIntArgs = iArgs;
                            currentOutTArgs = tArgs;
                            hasIntArgs = true;
                            hasTArgs = true;


                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.BROADCASTABLE_BOOL_OP_IMPL)
                                    .nIn(BROADCASTABLE_OP_IMPL_DEFAULT_NIN)
                                    .nOut(BROADCASTABLE_OP_IMPL_DEFAULT_NOUT)
                                    .iArgs(iArgs).tArgs(tArgs);

                            inOpBlock = true;
                        } else if(line.contains(PLATFORM_IMPL)) {
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, PLATFORM_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.PLATFORM_IMPL);

                            builder.name(name)
                                    .opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.PLATFORM_IMPL);
                            inOpBlock = true;
                            hasNin = true;
                            hasNout = true;
                            platformImpl = true;
                        }

                        else if(line.contains(OP_IMPL)) {
                            //OP_IMPL(NAME, NIN, NOUT, INPLACEABLE)
                            foundOp = true;
                            line = removeBracesFromDeclarationMacro(line, OP_IMPL);
                            String[] split = line.trim().split(",");
                            name = split[0];
                            opTypes.put(name, OpNamespace.OpDescriptor.OpDeclarationType.OP_IMPL);

                            argDescriptorProposals = new ArrayList<>();
                            ret.put(name,argDescriptorProposals);
                            int nIn = Integer.parseInt(split[1].trim());
                            int nOut = Integer.parseInt(split[2].trim());
                            currentOpNin = nIn;
                            currentOpNout = nOut;
                            hasNin = true;
                            hasNout = true;
                            boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                            builder.name(name).opDeclarationType(OpDeclarationDescriptor.OpDeclarationType.OP_IMPL)
                                    .nIn(nIn).nOut(nOut)
                                    .inplaceAble(inplaceAble);

                            inOpBlock = true;
                        }
                    }

                    line = line.trim();

                    //reset just in case we encounter another op in the file
                    //TODO: End of block needs to detect short circuits
                    // resize_nearest_neighbor is broke
                    if (inOpBlock && line.contains(RETURN) && endOfBlock(currLineIdx,lines) || oneLineOp) {
                        //reset op after 1 is found and current code block ends
                        if (foundOp) {
                            builder.inArgNames(inArgNames);
                            builder.outArgNames(outArgNames);
                            builder.tArgNames(tArgNames);
                            builder.iArgNames(iArgNames);
                            builder.bArgNames(bArgNames);

                            builder.bArgIndices(bArgIndices);
                            builder.iArgIndices(iArgIndices);
                            builder.tArgIndices(tArgIndices);
                            builder.inArgIndices(inArgIndices);
                            builder.outArgIndices(outArgIndices);

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

                        iArgIndices = new ArrayList<>();
                        bArgIndices = new ArrayList<>();
                        inArgIndices = new ArrayList<>();
                        tArgIndices  = new ArrayList<>();
                        outArgIndices = new ArrayList<>();

                        currentOpNin = -1;
                        currentOpNout = -1;
                        hasNin = false;
                        hasNout = false;
                        hasIntArgs = false;
                        hasTArgs = false;
                        currentOpBooleanArgs = -1;
                        currentOpIntArgs = -1;
                        currentOutTArgs = -1;
                        platformImpl = false;
                        argDescriptorProposals = new ArrayList<>();
                    }

                    if (inOpBlock) {
                         if(argDescriptorProposals == null)
                            argDescriptorProposals = new ArrayList<>();
                        if (line.isEmpty()) {
                            //ignore
                            /**
                             * Need to add case for array matching.
                             */
                        } else if (matchesArgDeclaration(INT_ARG,line)) {
                             processLine(iArgNames, iArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.INT64);

                        } else if (matchesArgDeclaration(OUTPUT_NULLIFIED,line)
                                || matchesArgDeclaration(OUTPUT_VARIABLE,line)) {
                            processLine(outArgNames, outArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR);

                        } else if (matchesArgDeclaration(T_ARG,line)) {
                            processLine(tArgNames, tArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.FLOAT);
                        } else if (matchesArgDeclaration(INPUT_VARIABLE,line) || matchesArgDeclaration(INPUT_LIST,line)) {
                            processLine(inArgNames,inArgIndices,argDescriptorProposals,line, OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR);
                         } else if (matchesArgDeclaration(B_ARG,line)) {
                            processLine(bArgNames, bArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.BOOL);
                        } else if(matchesArrayArgDeclaration(line.trim())) {
                            if(line.contains(INT_ARG))
                                processArrayLine(iArgNames, iArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.INT64);
                            else if(line.contains(OUTPUT_NULLIFIED) || line.contains(OUTPUT_VARIABLE)) {
                                processArrayLine(outArgNames, outArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR);
                            } else if(line.contains(T_ARG)) {
                                processArrayLine(tArgNames, tArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.FLOAT);
                            } else if(line.contains(B_ARG)) {
                                processArrayLine(bArgNames, bArgIndices, argDescriptorProposals, line, OpNamespace.ArgDescriptor.ArgType.BOOL);

                            }
                        }
                    }

                    //add alias descriptors
                    if (line.contains(DECLARE_SYN)) {
                        line = removeBracesFromDeclarationMacro(line, DECLARE_SYN);
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

                    currLineIdx++;
                }


            } catch (IOException e) {
                e.printStackTrace();
            }
        });



        return ret;

    }

    private boolean endOfBlock(int lineIndex,List<String> lines) {
        if(lineIndex < lines.size() - 2) {
            for(int i = lineIndex; i < lines.size() - 2; i++) {
                //could be last brace
                if(lines.get(i + 1).trim().equals("}") || lines.get(i + 1).trim().equals("};") || lines.get(i + 1).isEmpty() || lines.get(i + 1).trim().isEmpty()) {
                    continue;
                }
                if(lines.get(i + 1).contains("DECLARE_TYPES") ||
                        lines.get(i + 1).contains("DECLARE_SHAPE_FN")||
                        lines.get(i + 1).contains("DECLARE_SYN") ||
                        lines.get(i).contains("DECLARE_TYPES") ||
                        lines.get(i).contains("DECLARE_SHAPE_FN")||
                        lines.get(i).contains("DECLARE_SYN")) {
                    return true;
                } else if(!lines.get(i + 1).contains("DECLARE_TYPES") || !lines.get(i + 1).contains("DECLARE_SHAPE_FN") || !lines.get(i + 1).contains("DECLARE_SYN")) {
                    return false;
                }
            }
        }

        return true;

    }


    private void processArrayLine(List<String> iArgNames, List<Integer> iArgIndices,
                                  List<ArgDescriptorProposal> argDescriptorProposals,
                                  String line, OpNamespace.ArgDescriptor.ArgType argType) {
        String[] split = line.split(" = ");
        if(split.length == 1) {
            //invalid line
            return;
        }

        String[] arrSplit = split[0].split(" ");
        String name = arrSplit[0].replaceAll("\\[.*\\]","");
        Preconditions.checkState(!name.isEmpty());
        ArgDescriptorParserUtils.addArrayNameToList(line, iArgNames, iArgIndices);


        OpNamespace.ArgDescriptor argDescriptor = OpNamespace.ArgDescriptor.newBuilder()
                .setArgType(argType)
                .setIsArray(true)
                .setConvertBoolToInt(argType == OpNamespace.ArgDescriptor.ArgType.BOOL || line.contains("B_ARG"))
                .setName(name)
                .setArgIndex(-1).build();

        double weightToIncrementBy = weight * 1000000;
        ArgDescriptorProposal argDescriptorProposal = ArgDescriptorProposal.builder()
                .descriptor(argDescriptor)
                .sourceLine(line)
                .sourceOfProposal("cpp")
                .proposalWeight(weightToIncrementBy)
                .build();
        argDescriptorProposals.add(argDescriptorProposal);
    }


    private void processLine(List<String> iArgNames, List<Integer> iArgIndices,
                             List<ArgDescriptorProposal> argDescriptorProposals,
                             String line, OpNamespace.ArgDescriptor.ArgType argType) {
         boolean matchesPureDeclaration = Pattern.matches(ARG_DECLARATION,line) || Pattern.matches(ARG_BOOL_EQUALS_DECLARATION,line) || Pattern.matches(ARRAY_ASSIGNMENT,line);
        String[] split = line.split(" = ");
        if(split.length == 1) {
            //invalid line
            return;
        }

        String[] arrSplit = split[0].split(" ");
        //type + name
        Integer index = extractArgFromCpp(line);
        //guess index based on current number of indices already added
        if(index < 0) {
            index = iArgIndices.size();
        }
        ArgDescriptorParserUtils.addNameToList(line, iArgNames, iArgIndices);
        //note sometimes we have individual array entries for names, we need to strip out index indicators like [i]
        String argName = arrSplit[arrSplit.length - 1].replaceAll("\\[.*\\]","");
        Preconditions.checkState(!argName.isEmpty());
        //more than a typename variable name present
        if(arrSplit.length > 2) {
            //skip type
            for(int i = 1; i < arrSplit.length; i++) {
                //handle inline comments
                arrSplit[i] = arrSplit[i].trim();
                arrSplit[i] = arrSplit[i].replace(";","");
                if(isValidIdentifier(arrSplit[i])) {
                    argName = arrSplit[i];
                    Preconditions.checkState(!argName.isEmpty());
                    break;
                }
            }
        }

        Preconditions.checkState(!argName.isEmpty());

        OpNamespace.ArgDescriptor argDescriptor = OpNamespace.ArgDescriptor.newBuilder()
                .setArgType(argType)
                .setConvertBoolToInt(argType == OpNamespace.ArgDescriptor.ArgType.BOOL && !line.contains("B_ARG"))
                .setName(argName)
                .setArgIndex(index).build();
        double weightToIncrementBy = matchesPureDeclaration ? weight * 1000000 : weight;
        if(line.contains("->")) {
            weightToIncrementBy -= 100000;
        }
        ArgDescriptorProposal argDescriptorProposal = ArgDescriptorProposal.builder()
                .descriptor(argDescriptor)
                .sourceOfProposal("cpp")
                .sourceLine(line)
                .proposalWeight(weightToIncrementBy)
                .build();
        argDescriptorProposals.add(argDescriptorProposal);


    }

    @Override
    public Map<String, List<ArgDescriptorProposal>> getProposals() {
        return doExtractArgDescriptors();
    }

    @Override
    public OpNamespace.OpDescriptor.OpDeclarationType typeFor(String name) {
        return opTypes.get(name);
    }
}

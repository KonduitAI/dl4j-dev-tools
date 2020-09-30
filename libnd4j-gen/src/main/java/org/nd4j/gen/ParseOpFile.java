package org.nd4j.gen;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.util.*;

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
    public static void main(String...args) throws Exception {
        File libnd4jRootDir = new File("C:\\Users\\agibs\\Documents\\GitHub\\deeplearning4j\\libnd4j");
        List<OpDescriptor> opDescriptors = new ArrayList<>();
        Map<String, OpDescriptor> descriptorMap = new HashMap<>();

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
                OpDescriptor opDescriptor = null;
                OpDescriptor.OpDescriptorBuilder builder = OpDescriptor.builder();
                for (String line : lines) {
                    if(line.isEmpty())
                        continue;
                    if (line.contains(CUSTOM_OP_IMPL)) {
                        // CUSTOM_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                        foundOp = true;
                        line = line.replace(CUSTOM_OP_IMPL + "(", "");
                        line = line.replace(")", "");
                        line = line.replace("{", "");
                        String[] split = line.trim().split(",");
                        String name = split[0];
                        System.out.println("Name " + name);
                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                        int tArgs = Integer.parseInt(split[4].trim());
                        int iArgs = Integer.parseInt(split[5].trim());
                        builder.name(name)
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
                        line = line.replace(BOOLEAN_OP_IMPL + "(", "");
                        line = line.replace(")", "");
                        line = line.replace("{", "");
                        String[] split = line.trim().split(",");
                        String name = split[0];
                        System.out.println("Name " + name);
                        // BOOLEAN_OP_IMPL(NAME, NIN, SCALAR)
                        int nIn = Integer.parseInt(split[1].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[2].trim());
                        builder.name(name)
                                .nIn(nIn)
                                .inplaceAble(inplaceAble);

                        inOpBlock = true;
                    } else if(line.contains(LIST_OP_IMPL)) {
                        // LIST_OP_IMPL(NAME, NIN, NOUT, TARGS, IARGS)
                        foundOp = true;
                        line = line.replace(LIST_OP_IMPL + "(", "");
                        line = line.replace(")", "");
                        line = line.replace("{", "");
                        String[] split = line.trim().split(",");
                        String name = split[0];
                        System.out.println("Name " + name);
                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        int tArgs = Integer.parseInt(split[3].trim());
                        int iArgs = Integer.parseInt(split[4].trim());
                        builder.name(name)
                                .nIn(nIn).nOut(nOut)
                                .iArgs(iArgs).tArgs(tArgs);

                        inOpBlock = true;
                    } else if(line.contains(LOGIC_OP_IMPL)) {
                        // LOGIC_OP_IMPL(NAME)
                        foundOp = true;
                        if(line.contains(");"))
                            oneLineOp = true;
                        line = line.replace(LOGIC_OP_IMPL + "(", "");
                        line = line.replace(")", "");
                        line = line.replace("{", "");
                        line = line.replace(";","");
                        String[] split = line.trim().split(",");
                        String name = split[0];
                        System.out.println("Name " + name);
                        builder.name(name);

                        inOpBlock = true;
                    } else if(line.contains(DIVERGENT_OP_IMPL)) {
                        foundOp = true;
                        //DIVERGENT_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE)
                        line = line.replace(DIVERGENT_OP_IMPL + "(", "");
                        line = line.replace(")", "");
                        line = line.replace("{", "");
                        String[] split = line.trim().split(",");
                        String name = split[0];
                        System.out.println("Name " + name);
                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                        builder.name(name)
                                .nIn(nIn).nOut(nOut)
                                .inplaceAble(inplaceAble);

                        inOpBlock = true;
                    } else if(line.contains(CONFIGURABLE_OP_IMPL)) {
                        // CONFIGURABLE_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                        foundOp = true;
                        line = line.replace(CONFIGURABLE_OP_IMPL + "(", "");
                        line = line.replace(")", "");
                        line = line.replace("{", "");
                        String[] split = line.trim().split(",");
                        String name = split[0];
                        System.out.println("Name " + name);
                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                        int tArgs = Integer.parseInt(split[4].trim());
                        int iArgs = Integer.parseInt(split[5].trim());
                        builder.name(name)
                                .nIn(nIn).nOut(nOut)
                                .inplaceAble(inplaceAble)
                                .iArgs(iArgs).tArgs(tArgs);

                        inOpBlock = true;
                    } else if(line.contains(REDUCTION_OP_IMPL)) {
                        //REDUCTION_OP_IMPL(NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS)
                        foundOp = true;
                        line = line.replace(REDUCTION_OP_IMPL + "(", "");
                        line = line.replace(")", "");
                        line = line.replace("{", "");
                        String[] split = line.trim().split(",");
                        String name = split[0];
                        System.out.println("Name " + name);
                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                        int tArgs = Integer.parseInt(split[4].trim());
                        int iArgs = Integer.parseInt(split[5].trim());
                        builder.name(name)
                                .nIn(nIn).nOut(nOut)
                                .inplaceAble(inplaceAble)
                                .iArgs(iArgs).tArgs(tArgs);

                        inOpBlock = true;
                    } else if(line.contains(BROADCASTABLE_OP_IMPL)) {
                        //BROADCASTABLE_OP_IMPL(NAME, TARGS, IARGS)
                        foundOp = true;
                        line = line.replace(BROADCASTABLE_OP_IMPL + "(", "");
                        line = line.replace(")", "");
                        line = line.replace("{", "");
                        String[] split = line.trim().split(",");
                        String name = split[0];
                        System.out.println("Name " + name);
                        int tArgs = Integer.parseInt(split[1].trim());
                        int iArgs = Integer.parseInt(split[2].trim());
                        builder.name(name)
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
                        System.out.println("Name " + name);
                        int tArgs = Integer.parseInt(split[1].trim());
                        int iArgs = Integer.parseInt(split[2].trim());
                        builder.name(name)
                                .iArgs(iArgs).tArgs(tArgs);

                        inOpBlock = true;
                    }  else if(line.contains(OP_IMPL)) {
                        //OP_IMPL(NAME, NIN, NOUT, INPLACEABLE)
                        foundOp = true;
                        line = line.replace(OP_IMPL + "(", "");
                        line = line.replace(")", "");
                        line = line.replace("{", "");
                        String[] split = line.trim().split(",");
                        String name = split[0];
                        System.out.println("Name " + name);
                        int nIn = Integer.parseInt(split[1].trim());
                        int nOut = Integer.parseInt(split[2].trim());
                        boolean inplaceAble = Boolean.parseBoolean(split[3].trim());
                        builder.name(name)
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

                            opDescriptor = builder.build();
                            System.out.println(opDescriptor);

                            //NAME, NIN, NOUT, INPLACEABLE, TARGS, IARGS

                            opDescriptors.add(opDescriptor);

                            if (opDescriptor != null) {
                                System.out.println("Op descriptor " + opDescriptor);
                                System.out.println("Input arg name " + inArgNames);
                                System.out.println("Output arg names " + outArgNames);
                                System.out.println("T Arg names " + tArgNames);
                                System.out.println("Integer arg names " + iArgNames);
                                System.out.println("Boolean arg names " + bArgNames);
                                opDescriptor.validate();
                            }
                        }

                        descriptorMap.put(opDescriptor.getName(),opDescriptor);

                        inOpBlock = false;
                        foundOp = false;
                        oneLineOp = false;
                        opDescriptor = null;
                        builder = OpDescriptor.builder();
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
                        //DECLARE_SYN(mMul, matmul);
                        line = line.replace(DECLARE_SYN,"");
                        line = line.replace("(","");
                        line = line.replace(")","");
                        line = line.replace(";","");
                        String[] args2 = line.split(",");
                        String aliasFor = args2[1].trim();
                        String newKey = args2[0].trim();
                        if(descriptorMap.isEmpty()) {
                            throw new IllegalStateException("Descriptor map should not be empty here");
                        }

                        OpDescriptor.OpDescriptorBuilder opDescriptor2 = descriptorMap.get(aliasFor).toBuilder();
                        opDescriptor2.name(newKey);
                        OpDescriptor newDescriptor = opDescriptor2.build();
                        opDescriptors.add(newDescriptor);
                        descriptorMap.put(args2[1],newDescriptor);
                    }
                }


            } catch (IOException e) {
                e.printStackTrace();
            }
        });

        System.out.println("Number of op descriptors " + opDescriptors.size());
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

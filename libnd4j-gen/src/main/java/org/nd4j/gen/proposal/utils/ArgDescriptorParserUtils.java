package org.nd4j.gen.proposal.utils;

import org.apache.commons.text.similarity.LevenshteinDistance;
import org.nd4j.common.primitives.Pair;
import org.nd4j.gen.OpDeclarationDescriptor;
import org.nd4j.gen.proposal.ArgDescriptorProposal;
import org.nd4j.ir.OpNamespace;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class ArgDescriptorParserUtils {
    public final static String DEFAULT_OUTPUT_FILE = "op-ir.proto";
    public final static Pattern numberPattern = Pattern.compile("\\([\\d]+\\)");
    /**
     *     void addTArgument(double... arg);
     *
     *     void addIArgument(int... arg);
     *
     *     void addIArgument(long... arg);
     *
     *     void addBArgument(boolean... arg);
     *
     *     void addDArgument(DataType... arg);
     */

    public final static String ADD_T_ARGUMENT_INVOCATION = "addTArgument";
    public final static String ADD_I_ARGUMENT_INVOCATION = "addIArgument";
    public final static String ADD_B_ARGUMENT_INVOCATION = "addBArgument";
    public final static String ADD_D_ARGUMENT_INVOCATION = "addDArgument";

    public static final Set<String> cppTypes = new HashSet<String>() {{
        add("int");
        add("bool");
        add("auto");
        add("string");
        add("float");
        add("double");
        add("char");
        add("class");
        add("uint");
    }};

    public final static Set<String> fieldNameFilters = new HashSet<String>() {{
        add("sameDiff");
        add("xVertexId");
        add("yVertexId");
        add("zVertexId");
        add("extraArgs");
        add("extraArgz");
        add("dimensionz");
        add("scalarValue");
        add("dimensions");
        add("jaxis");
        add("inPlace");
    }};
    public final static  Set<String> fieldNameFiltersDynamicCustomOps = new HashSet<String>() {{
        add("sameDiff");
        add("xVertexId");
        add("yVertexId");
        add("zVertexId");
        add("extraArgs");
        add("extraArgz");
        add("dimensionz");
        add("scalarValue");
        add("jaxis");
        add("inPlace");
        add("inplaceCall");
    }};

    private static Map<String,String> equivalentAttributeNames = new HashMap<String,String>() {{
        put("axis","dimensions");
        put("dimensions","axis");
        put("jaxis","dimensions");
        put("dimensions","jaxis");
        put("inplaceCall","inPlace");
        put("inPlace","inplaceCall");
    }};


    private static Set<String> dimensionNames = new HashSet<String>() {{
        add("jaxis");
        add("axis");
        add("dimensions");
        add("dimensionz");
        add("dim");
        add("axisVector");
    }};

    private static Set<String> inputNames = new HashSet<String>() {{
        add("input");
        add("inputs");
        add("i_v");
        add("x");
    }};

    private static Set<String> outputNames = new HashSet<String>() {{
        add("output");
        add("outputs");
    }};


    private static Set<String> inplaceNames = new HashSet<String>() {{
        add("inPlace");
        add("inplaceCall");
    }};


    public static boolean equivalentAttribute(OpNamespace.ArgDescriptor comp1, OpNamespace.ArgDescriptor comp2) {
        if(equivalentAttributeNames.containsKey(comp1.getName())) {
            return equivalentAttributeNames.get(comp1.getName()).equals(comp2.getName());
        }

        if(equivalentAttributeNames.containsKey(comp2.getName())) {
            return equivalentAttributeNames.get(comp2.getName()).equals(comp1.getName());
        }
        return false;
    }

    public static boolean argsListContainsEquivalentAttribute(List<OpNamespace.ArgDescriptor> argDescriptors, OpNamespace.ArgDescriptor to) {
        for(OpNamespace.ArgDescriptor argDescriptor : argDescriptors) {
            if(argDescriptor.getArgType() == to.getArgType() && equivalentAttribute(argDescriptor,to)) {
                return true;
            }
        }

        return false;
    }

    public static boolean argsListContainsSimilarArg(List<OpNamespace.ArgDescriptor> argDescriptors, OpNamespace.ArgDescriptor to, int threshold) {
        for(OpNamespace.ArgDescriptor argDescriptor : argDescriptors) {
            if(argDescriptor.getArgType() == to.getArgType() && LevenshteinDistance.getDefaultInstance().apply(argDescriptor.getName().toLowerCase(),to.getName().toLowerCase()) <= threshold) {
                return true;
            }
        }

        return false;
    }

    public static OpNamespace.ArgDescriptor mergeDescriptorsOfSameIndex(OpNamespace.ArgDescriptor one, OpNamespace.ArgDescriptor two) {
        if(one.getArgIndex() != two.getArgIndex()) {
            throw new IllegalArgumentException("Argument indices for both arg descriptors were not the same. First one was " + one.getArgIndex() + " and second was " + two.getArgIndex());
        }

        if(one.getArgType() != two.getArgType()) {
            throw new IllegalArgumentException("Merging two arg descriptors requires both be the same type. First one was " + one.getArgType().name() + " and second one was " + two.getArgType().name());
        }

        OpNamespace.ArgDescriptor.Builder newDescriptor = OpNamespace.ArgDescriptor.newBuilder();
        //arg indices will be the same
        newDescriptor.setArgIndex(one.getArgIndex());
        newDescriptor.setArgType(one.getArgType());
        if(!isValidIdentifier(one.getName()) && !isValidIdentifier(two.getName())) {
            newDescriptor.setName("arg" + newDescriptor.getArgIndex());
        } else if(!isValidIdentifier(one.getName())) {
            newDescriptor.setName(two.getName());
        } else {
            newDescriptor.setName(one.getName());
        }


        return newDescriptor.build();
    }

    public static boolean isValidIdentifier(String input) {
        for(int i = 0; i < input.length(); i++) {
            if(!Character.isJavaIdentifierPart(input.charAt(i)))
                return false;
        }

        if(cppTypes.contains(input))
            return false;

        return true;
    }

    public  List<ArgDescriptorProposal> updateOpDescriptor(OpNamespace.OpDescriptor opDescriptor, OpDeclarationDescriptor declarationDescriptor, List<String> argsByIIndex, OpNamespace.ArgDescriptor.ArgType int64) {
        List<OpNamespace.ArgDescriptor> copyValuesInt = addArgDescriptors(opDescriptor, declarationDescriptor, argsByIIndex, int64);
        List<ArgDescriptorProposal> proposals = new ArrayList<>();

        return proposals;
    }

    public static List<OpNamespace.ArgDescriptor> addArgDescriptors(OpNamespace.OpDescriptor opDescriptor, OpDeclarationDescriptor declarationDescriptor, List<String> argsByTIndex, OpNamespace.ArgDescriptor.ArgType argType) {
        List<OpNamespace.ArgDescriptor> copyValuesFloat = new ArrayList<>(opDescriptor.getArgDescriptorList());
        for(int i = 0; i < argsByTIndex.size(); i++) {
            OpNamespace.ArgDescriptor argDescriptor = OpNamespace.ArgDescriptor.newBuilder()
                    .setArgType(argType)
                    .setName(argsByTIndex.get(i))
                    .setArgIndex(i)
                    //this can happen when there are still missing names from c++
                    .setArgOptional(declarationDescriptor != null &&  i <= declarationDescriptor.getTArgs() ? false : true)
                    .build();
            copyValuesFloat.add(argDescriptor);

        }
        return copyValuesFloat;
    }

    public static Map<String,Integer> argIndexForCsv(String line) {
        Map<String,Integer> ret = new HashMap<>();
        String[] lineSplit = line.split(",");
        for(int i = 0; i < lineSplit.length; i++) {
            ret.put(lineSplit[i],i);
        }

        return ret;
    }

    public static Integer extractArgFromJava(String line) {
        Matcher matcher =  numberPattern.matcher(line);
        if(!matcher.find()) {
            throw new IllegalArgumentException("No number found for line " + line);
        }

        return Integer.parseInt(matcher.group());
    }

    public static Integer extractArgFromCpp(String line) {
        Matcher matcher =  numberPattern.matcher(line);
        if(!matcher.find()) {
            //Generally not resolvable
            return -1;
        }

        if(matcher.groupCount() > 1) {
            throw new IllegalArgumentException("Line contains more than 1 index");
        }

        try {
            return Integer.parseInt(matcher.group().replace("(","").replace(")",""));
        } catch(NumberFormatException e) {
            e.printStackTrace();
            return -1;
        }
    }

    public static List<Field> getAllFields(Class clazz) {
        if (clazz == null) {
            return Collections.emptyList();
        }

        List<Field> result = new ArrayList<>(getAllFields(clazz.getSuperclass()));
        List<Field> filteredFields = Arrays.stream(clazz.getDeclaredFields())
                .filter(f -> Modifier.isPublic(f.getModifiers()) || Modifier.isProtected(f.getModifiers()))
                .collect(Collectors.toList());
        result.addAll(filteredFields);
        return result;
    }

    public static String removeBracesFromDeclarationMacro(String line, String nameOfMacro) {
        line = line.replace(nameOfMacro + "(", "");
        line = line.replace(")", "");
        line = line.replace("{", "");
        line = line.replace(";","");
        return line;
    }

    public static void addNameToList(String line, List<String> list, List<Integer> argIndices) {
        String[] split = line.split(" = ");
        String[] arrSplit = split[0].split(" ");
        //type + name
        String name = arrSplit[arrSplit.length - 1];
        if(!list.contains(name))
            list.add(name);

        Integer index = extractArgFromCpp(line);
        if(index != null)
            argIndices.add(index);
    }



    public static void standardizeTypes(List<ArgDescriptorProposal> input) {
        input.stream().forEach(proposal -> {
            //note that if automatic conversion should not happen, set convertBoolToInt to false
            if(proposal.getDescriptor().getArgType() == OpNamespace.ArgDescriptor.ArgType.BOOL && proposal.getDescriptor().getConvertBoolToInt()) {
                OpNamespace.ArgDescriptor newDescriptor = OpNamespace.ArgDescriptor.newBuilder()
                        .setArgIndex(proposal.getDescriptor().getArgIndex())
                        .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                        .setName(proposal.getDescriptor().getName())
                        .build();
                proposal.setDescriptor(newDescriptor);
            }
        });
    }

    public static Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>, OpNamespace.ArgDescriptor> standardizeNames
            (Map<String,List<ArgDescriptorProposal>> toStandardize) {
        Map<String,List<ArgDescriptorProposal>> ret = new HashMap<>();
        List<ArgDescriptorProposal> dimensionsList = new ArrayList<>();
        List<ArgDescriptorProposal> inPlaceList = new ArrayList<>();
        List<ArgDescriptorProposal> inputsList = new ArrayList<>();
        List<ArgDescriptorProposal> outputsList = new ArrayList<>();
        for(Map.Entry<String,List<ArgDescriptorProposal>> entry : toStandardize.entrySet()) {
            if(dimensionNames.contains(entry.getKey())) {
                dimensionsList.addAll(entry.getValue());
            } else if(inplaceNames.contains(entry.getKey())) {
                inPlaceList.addAll(entry.getValue());
            } else if(inputNames.contains(entry.getKey())) {
                inputsList.addAll(entry.getValue());
            } else if(outputNames.contains(entry.getKey())) {
                outputsList.addAll(entry.getValue());
            }
            else {
                ret.put(entry.getKey(),entry.getValue());
            }
        }

        if(!dimensionsList.isEmpty()) {
            for(ArgDescriptorProposal argDescriptorProposal : dimensionsList) {
                argDescriptorProposal.setDescriptor(OpNamespace.ArgDescriptor.newBuilder()
                        .setName("dimensions")
                        .setArgType(argDescriptorProposal.getDescriptor().getArgType())
                        .setArgIndex(argDescriptorProposal.getDescriptor().getArgIndex())
                        .build());
            }
            ret.put("dimensions",dimensionsList);

        }

        if(!inPlaceList.isEmpty()) {
            for(ArgDescriptorProposal argDescriptorProposal : inPlaceList) {
                argDescriptorProposal.setDescriptor(OpNamespace.ArgDescriptor.newBuilder()
                        .setName("inPlace")
                        .setArgType(argDescriptorProposal.getDescriptor().getArgType())
                        .setArgIndex(argDescriptorProposal.getDescriptor().getArgIndex()).build());
            }
            ret.put("inPlace",inPlaceList);
        }

        if(!inputsList.isEmpty()) {
            for(ArgDescriptorProposal argDescriptorProposal : inputsList) {
                argDescriptorProposal.setDescriptor(OpNamespace.ArgDescriptor.newBuilder()
                        .setName("input")
                        .setArgType(argDescriptorProposal.getDescriptor().getArgType())
                        .setArgIndex(argDescriptorProposal.getDescriptor().getArgIndex()).build());
            }
            ret.put("input",inputsList);
        }

        if(!outputsList.isEmpty()) {
            for(ArgDescriptorProposal argDescriptorProposal : outputsList) {
                argDescriptorProposal.setDescriptor(OpNamespace.ArgDescriptor.newBuilder()
                        .setName("output")
                        .setArgType(argDescriptorProposal.getDescriptor().getArgType())
                        .setArgIndex(argDescriptorProposal.getDescriptor().getArgIndex()).build());
            }
            ret.put("output",outputsList);
        }

        //de duplicate the same index/same type
        Map<String,List<ArgDescriptorProposal>> deDedupped = new HashMap<>();
        for(Map.Entry<String,List<ArgDescriptorProposal>> entry : toStandardize.entrySet()) {
            Collections.sort(entry.getValue(),Comparator.comparingDouble(ArgDescriptorProposal::getProposalWeight));
            deDedupped.put(entry.getKey(),Arrays.asList(entry.getValue().get(entry.getValue().size() - 1)));
            Map<Integer, List<ArgDescriptorProposal>> collect = deDedupped.get(entry.getKey()).stream()
                    .collect(Collectors.groupingBy(input -> input.getDescriptor().getArgIndex()));
            collect.entrySet().forEach(input -> {
                Map<String, List<ArgDescriptorProposal>> groupedByName = input.getValue().stream().collect(Collectors.groupingBy(descriptorProposal ->
                        descriptorProposal.getDescriptor().getName()));
                groupedByName.entrySet().forEach(groupedEntry -> {
                    Collections.sort(groupedEntry.getValue(), Comparator.comparingDouble(ArgDescriptorProposal::getProposalWeight));
                    //update list to single element only
                    groupedByName.put(groupedEntry.getKey(),Arrays.asList(groupedEntry.getValue().get(groupedEntry.getValue().size() - 1)));
                });


            });
        }

        Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>,ArgDescriptorProposal> collect = new HashMap<>();
        deDedupped.entrySet().forEach(entry -> {
            /**
             * TODO: KH should be index 0 but is 1 for some reason.
             * sH should be 9 and should be correct now (op is avgpool2d)
             */
            OpNamespace.ArgDescriptor currDescriptor = entry.getValue().get(0).getDescriptor();
            OpNamespace.ArgDescriptor.ArgType argType = currDescriptor.getArgType();
            Integer index = currDescriptor.getArgIndex();
            if(!collect.containsKey(Pair.of(index,argType))) {
                if(entry.getValue().size() > 1) {
                    ArgDescriptorProposal highestProposal = null;
                    for(int i = 0; i < entry.getValue().size(); i++) {
                        if(highestProposal == null || highestProposal.getProposalWeight() < entry.getValue().get(i).getProposalWeight()) {
                            highestProposal = entry.getValue().get(i);
                        }
                        if(highestProposal.getDescriptor().getArgIndex() != index) {
                            throw new IllegalArgumentException("Potential new highest proposal index " + highestProposal.getDescriptor().getArgIndex() + " does not match attempted key " + index);

                        }
                        collect.put(Pair.of(index,argType),highestProposal);

                    }
                } else {
                    if(entry.getValue().get(0).getDescriptor().getArgIndex() != index) {
                        throw new IllegalArgumentException("Potential new highest proposal index " + entry.getValue().get(0).getDescriptor().getArgIndex() + " does not match attempted key " + index);
                    }
                    collect.put(Pair.of(index,argType),entry.getValue().get(0));
                }

            } else {
                ArgDescriptorProposal argDescriptorProposal = collect.get(Pair.of(index,argType));
                ArgDescriptorProposal potentialNewOne = entry.getValue().get(0);
                if(potentialNewOne.getDescriptor().getArgIndex() != index) {
                    throw new IllegalArgumentException("Potential new arg descriptor index " + potentialNewOne.getDescriptor().getArgIndex() + " does not match attempted key " + index);
                }
                if(potentialNewOne.getProposalWeight() > argDescriptorProposal.getProposalWeight()) {
                    collect.put(Pair.of(index,argType),potentialNewOne);
                }
            }
        });

        Map<Pair<Integer, OpNamespace.ArgDescriptor.ArgType>, OpNamespace.ArgDescriptor> ret2 = collect.entrySet().stream().collect(Collectors.toMap(entry -> entry.getKey(), entry -> entry.getValue().getDescriptor()));
        return ret2;
    }


}

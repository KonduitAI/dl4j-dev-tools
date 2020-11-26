package org.nd4j.gen.proposal.impl;

import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.resolution.declarations.ResolvedConstructorDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedFieldDeclaration;
import com.github.javaparser.resolution.declarations.ResolvedParameterDeclaration;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.Log;
import com.github.javaparser.utils.SourceRoot;
import lombok.Builder;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.primitives.Counter;
import org.nd4j.common.primitives.CounterMap;
import org.nd4j.common.primitives.Pair;
import org.nd4j.gen.proposal.ArgDescriptorProposal;
import org.nd4j.gen.proposal.ArgDescriptorSource;
import org.nd4j.ir.OpNamespace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.reflections.Reflections;

import java.io.File;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.nd4j.gen.proposal.impl.ArgDescriptorParserUtils.*;

public class JavaSourceArgDescriptorSource implements ArgDescriptorSource {


    private  SourceRoot sourceRoot;
    private File nd4jOpsRootDir;
    private double weight;

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
    public final static String ADD_INPUT_ARGUMENT = "addInputArgument";
    public final static String ADD_OUTPUT_ARGUMENT = "addOutputArgument";
    private Map<String, OpNamespace.OpDescriptor.OpDeclarationType> opTypes;
    static {
        Log.setAdapter(new Log.StandardOutStandardErrorAdapter());

    }

    @Builder
    public JavaSourceArgDescriptorSource(File nd4jApiRootDir,double weight) {
        this.sourceRoot = initSourceRoot(nd4jApiRootDir);
        this.nd4jOpsRootDir = nd4jApiRootDir;
        if(opTypes == null) {
            opTypes = new HashMap<>();
        }

        this.weight = weight;
    }

    public Map<String, List<ArgDescriptorProposal>> doReflectionsExtraction() {
        Map<String, List<ArgDescriptorProposal>> ret = new HashMap<>();

        Reflections reflections = new Reflections("org.nd4j");
        Set<Class<? extends DifferentialFunction>> subTypesOf = reflections.getSubTypesOf(DifferentialFunction.class);
        Set<Class<? extends CustomOp>> subTypesOfOp = reflections.getSubTypesOf(CustomOp.class);
        Set<Class<?>> allClasses = new HashSet<>();
        allClasses.addAll(subTypesOf);
        allClasses.addAll(subTypesOfOp);
        Set<String> opNamesForDifferentialFunction = new HashSet<>();


        for(Class<?> clazz : allClasses) {
            if(Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface()) {
                continue;
            }

            processClazz(ret, opNamesForDifferentialFunction, clazz);

        }


        return ret;
    }

    private void processClazz(Map<String, List<ArgDescriptorProposal>> ret, Set<String> opNamesForDifferentialFunction, Class<?> clazz) {
        try {
            Object funcInstance = clazz.newInstance();
            String name = null;

            if(funcInstance instanceof DifferentialFunction) {
                DifferentialFunction differentialFunction = (DifferentialFunction) funcInstance;
                name = differentialFunction.opName();
            } else if(funcInstance instanceof CustomOp) {
                CustomOp customOp = (CustomOp) funcInstance;
                name = customOp.opName();
            }


            if(name == null)
                return;
            opNamesForDifferentialFunction.add(name);
            if(!(funcInstance instanceof DynamicCustomOp))
                opTypes.put(name,OpNamespace.OpDescriptor.OpDeclarationType.LEGACY_XYZ);
            else
                opTypes.put(name,OpNamespace.OpDescriptor.OpDeclarationType.CUSTOM_OP_IMPL);


            String fileName = clazz.getName().replace(".",File.separator);
            StringBuilder fileBuilder = new StringBuilder();
            fileBuilder.append(fileName);
            fileBuilder.append(".java");
            CounterMap<Pair<String, OpNamespace.ArgDescriptor.ArgType>,Integer> paramIndicesCount = new CounterMap<>();

            // Our sample is in the root of this directory, so no package name.
            CompilationUnit cu = sourceRoot.parse(clazz.getPackage().getName(), clazz.getSimpleName() + ".java");
            cu.findAll(MethodCallExpr.class).forEach(method -> {
                        String methodInvoked = method.getNameAsString();
                        final AtomicInteger indexed = new AtomicInteger(0);
                        //need to figure out how to consolidate multiple method calls
                        //as well as the right indices
                        //typical patterns in the code base will reflect adding arguments all at once
                        //one thing we can just check for is if more than 1 argument is passed in and
                        //treat that as a complete list of arguments
                        if(methodInvoked.equals(ADD_T_ARGUMENT_INVOCATION)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.FLOAT),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().toString().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.FLOAT),indexed.get(),100.0);

                                    }
                                }
                                indexed.incrementAndGet();
                            });
                        } else if(methodInvoked.equals(ADD_B_ARGUMENT_INVOCATION)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.BOOL),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.BOOL),indexed.get(),100.0);
                                    }
                                }
                                indexed.incrementAndGet();
                            });
                        } else if(methodInvoked.equals(ADD_I_ARGUMENT_INVOCATION)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.INT64),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().toString().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.toString().replace(".ordinal()",""), OpNamespace.ArgDescriptor.ArgType.INT64),indexed.get(),100.0);

                                    }
                                }

                                indexed.incrementAndGet();
                            });
                        } else if(methodInvoked.equals(ADD_D_ARGUMENT_INVOCATION)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.DATA_TYPE),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().toString().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.toString().replace(".ordinal()",""), OpNamespace.ArgDescriptor.ArgType.DATA_TYPE),indexed.get(),100.0);

                                    }
                                }
                                indexed.incrementAndGet();
                            });
                        } else if(methodInvoked.equals(ADD_INPUT_ARGUMENT)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().toString().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.toString().replace(".ordinal()",""), OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR),indexed.get(),100.0);

                                    }
                                }
                                indexed.incrementAndGet();
                            });
                        } else if(methodInvoked.equals(ADD_OUTPUT_ARGUMENT)) {
                            method.getArguments().forEach(argument -> {
                                if(argument.isNameExpr())
                                    paramIndicesCount.incrementCount(Pair.of(argument.asNameExpr().getNameAsString(), OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR),indexed.get(),100.0);
                                else if(argument.isMethodCallExpr()) {
                                    if(argument.asMethodCallExpr().getName().toString().equals("ordinal")) {
                                        paramIndicesCount.incrementCount(Pair.of(argument.toString().replace(".ordinal()",""), OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR),indexed.get(),100.0);

                                    }
                                }
                                indexed.incrementAndGet();
                            });
                        }

                    }
            );




            List<ResolvedConstructorDeclaration> collect = cu.findAll(ConstructorDeclaration.class).stream()
                    .map(input -> input.resolve())
                    .filter(constructor -> constructor.getNumberOfParams() > 0)
                    .distinct()
                    .collect(Collectors.toList());
            List<ArgDescriptorProposal> argDescriptorProposals = ret.get(name);
            if(argDescriptorProposals == null) {
                argDescriptorProposals = new ArrayList<>();
                ret.put(name,argDescriptorProposals);
            }

            Set<ResolvedParameterDeclaration> parameters = new LinkedHashSet<>();

            int floatIdx = 0;
            int inputIdx = 0;
            int outputIdx = 0;
            int intIdx = 0;
            int boolIdx = 0;

            for(ResolvedConstructorDeclaration parameterDeclaration : collect) {
                floatIdx = 0;
                inputIdx = 0;
                outputIdx = 0;
                intIdx = 0;
                boolIdx = 0;
                for(int i = 0; i < parameterDeclaration.getNumberOfParams(); i++) {
                    ResolvedParameterDeclaration param = parameterDeclaration.getParam(i);
                    OpNamespace.ArgDescriptor.ArgType argType = argTypeForParam(param);
                    if(isValidParam(param)) {
                        parameters.add(param);
                        switch(argType) {
                            case INPUT_TENSOR:
                                paramIndicesCount.incrementCount(Pair.of(param.getName(),argType), inputIdx, 100.0);
                                inputIdx++;
                                break;
                            case INT64:
                            case INT32:
                                paramIndicesCount.incrementCount(Pair.of(param.getName(), OpNamespace.ArgDescriptor.ArgType.INT64), intIdx, 100.0);
                                intIdx++;
                                break;
                            case DOUBLE:
                            case FLOAT:
                                paramIndicesCount.incrementCount(Pair.of(param.getName(), OpNamespace.ArgDescriptor.ArgType.FLOAT), floatIdx, 100.0);
                                paramIndicesCount.incrementCount(Pair.of(param.getName(), OpNamespace.ArgDescriptor.ArgType.DOUBLE), floatIdx, 100.0);
                                floatIdx++;
                                break;
                            case BOOL:
                                paramIndicesCount.incrementCount(Pair.of(param.getName(),argType), boolIdx, 100.0);
                                boolIdx++;
                                break;
                            case OUTPUT_TENSOR:
                                paramIndicesCount.incrementCount(Pair.of(param.getName(),argType), outputIdx, 100.0);
                                outputIdx++;
                                break;
                            case UNRECOGNIZED:
                                continue;

                        }

                    }
                }
            }

            floatIdx = 0;
            inputIdx = 0;
            outputIdx = 0;
            intIdx = 0;
            boolIdx = 0;
            Set<List<Pair<String, String>>> typesAndParams = parameters.stream().map(collectedParam ->
                    Pair.of(collectedParam.describeType(), collectedParam.getName()))
                    .collect(Collectors.groupingBy(input -> input.getSecond())).entrySet()
                    .stream()
                    .map(inputPair -> inputPair.getValue())
                    .collect(Collectors.toSet());


            Set<String> constructorNamesEncountered =new HashSet<>();
            List<ArgDescriptorProposal> finalArgDescriptorProposals = argDescriptorProposals;
            typesAndParams.forEach(listOfTypesAndNames -> {

                listOfTypesAndNames.forEach(parameter -> {
                    if(typeNameOrArrayOfTypeNameMatches(parameter.getFirst(),SDVariable.class.getName(),INDArray.class.getName())) {
                        constructorNamesEncountered.add(parameter.getValue());
                        if(outputNames.contains(parameter.getValue())) {
                            Counter<Integer> counter = paramIndicesCount.getCounter(Pair.of(parameter.getSecond(), OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR));
                            if(counter != null)
                                finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .proposalWeight(99.0 * (counter == null ? 1 : counter.size()))
                                        .sourceOfProposal("java")
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
                                                .setName(parameter.getSecond())
                                                .setIsArray(parameter.getFirst().contains("[]") || parameter.getFirst().contains("..."))
                                                .setArgIndex(counter.argMax())
                                                .build()).build());

                        } else {
                            Counter<Integer> counter = paramIndicesCount.getCounter(Pair.of(parameter.getSecond(), OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR));
                            if(counter != null)
                                finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                        .proposalWeight(99.0 * (counter == null ? 1 : counter.size()))
                                        .sourceOfProposal("java")
                                        .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                                .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                                .setName(parameter.getSecond())
                                                .setIsArray(parameter.getFirst().contains("[]") || parameter.getFirst().contains("..."))
                                                .setArgIndex(counter.argMax())
                                                .build()).build());
                        }
                    } else if(typeNameOrArrayOfTypeNameMatches(parameter.getFirst(),int.class.getName(),long.class.getName(),Integer.class.getName(),Long.class.getName()) || paramIsEnum(parameter.getFirst())) {
                        constructorNamesEncountered.add(parameter.getValue());

                        Counter<Integer> counter = paramIndicesCount.getCounter(Pair.of(parameter.getSecond(), OpNamespace.ArgDescriptor.ArgType.INT64));
                        if(counter != null)
                            finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                    .sourceOfProposal("java")
                                    .proposalWeight(99.0 * (counter == null ? 1 : counter.size()))
                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                            .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                            .setName(parameter.getSecond())
                                            .setIsArray(parameter.getFirst().contains("[]") || parameter.getFirst().contains("..."))
                                            .setArgIndex(counter.argMax())
                                            .build()).build());
                    } else if(typeNameOrArrayOfTypeNameMatches(parameter.getFirst(),float.class.getName(),double.class.getName(),Float.class.getName(),Double.class.getName())) {
                        constructorNamesEncountered.add(parameter.getValue());
                        Counter<Integer> counter = paramIndicesCount.getCounter(Pair.of(parameter.getSecond(), OpNamespace.ArgDescriptor.ArgType.FLOAT));
                        if(counter != null)
                            finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                    .sourceOfProposal("java")
                                    .proposalWeight(99.0 * (counter == null ? 1 :(counter == null ? 1 : counter.size()) ))
                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                            .setArgType(OpNamespace.ArgDescriptor.ArgType.FLOAT)
                                            .setName(parameter.getSecond())
                                            .setIsArray(parameter.getFirst().contains("[]"))
                                            .setArgIndex(counter.argMax())
                                            .build()).build());
                    } else if(typeNameOrArrayOfTypeNameMatches(parameter.getFirst(),boolean.class.getName(),Boolean.class.getName())) {
                        constructorNamesEncountered.add(parameter.getValue());
                        Counter<Integer> counter = paramIndicesCount.getCounter(Pair.of(parameter.getSecond(), OpNamespace.ArgDescriptor.ArgType.BOOL));
                        if(counter != null)
                            finalArgDescriptorProposals.add(ArgDescriptorProposal.builder()
                                    .sourceOfProposal("java")
                                    .proposalWeight(99.0 * (counter == null ? 1 :(counter == null ? 1 : counter.size()) ))
                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                            .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                            .setName(parameter.getSecond())
                                            .setIsArray(parameter.getFirst().contains("[]"))
                                            .setArgIndex(counter.argMax())
                                            .build()).build());
                    }
                });
            });




            List<ResolvedFieldDeclaration> fields = cu.findAll(FieldDeclaration.class).stream()
                    .map(input -> getResolve(input))
                    .filter(input -> input != null)
                    .collect(Collectors.toList());
            floatIdx = 0;
            inputIdx = 0;
            outputIdx = 0;
            intIdx = 0;
            boolIdx = 0;

            for(ResolvedFieldDeclaration field : fields) {
                if(!constructorNamesEncountered.contains(field.getName()) && typeNameOrArrayOfTypeNameMatches(field.getType().describe(),SDVariable.class.getName(),INDArray.class.getName())) {
                    if(outputNames.contains(field.getName())) {
                        argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                .sourceOfProposal("java")
                                .proposalWeight(99.0)
                                .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                        .setArgType(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
                                        .setName(field.getName())
                                        .setIsArray(field.getType().describe().contains("[]"))
                                        .setArgIndex(outputIdx)
                                        .build()).build());
                        outputIdx++;
                    } else if(!constructorNamesEncountered.contains(field.getName())){
                        argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                .sourceOfProposal("java")
                                .proposalWeight(99.0)
                                .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                        .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                        .setName(field.getName())
                                        .setIsArray(field.getType().describe().contains("[]"))
                                        .setArgIndex(inputIdx)
                                        .build()).build());
                        inputIdx++;
                    }
                } else if(!constructorNamesEncountered.contains(field.getName()) && typeNameOrArrayOfTypeNameMatches(field.getType().describe(),int.class.getName(),long.class.getName(),Long.class.getName(),Integer.class.getName())) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("java")
                            .proposalWeight(99.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                    .setName(field.getName())
                                    .setIsArray(field.getType().describe().contains("[]"))
                                    .setArgIndex(intIdx)
                                    .build()).build());
                    intIdx++;
                } else if(!constructorNamesEncountered.contains(field.getName()) && typeNameOrArrayOfTypeNameMatches(field.getType().describe(),double.class.getName(),float.class.getName(),Double.class.getName(),Float.class.getName())) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("java")
                            .proposalWeight(99.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.FLOAT)
                                    .setName(field.getName())
                                    .setIsArray(field.getType().describe().contains("[]"))
                                    .setArgIndex(floatIdx)
                                    .build()).build());
                    floatIdx++;
                } else if(!constructorNamesEncountered.contains(field.getName()) && typeNameOrArrayOfTypeNameMatches(field.getType().describe(),Boolean.class.getName(),boolean.class.getName())) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("java")
                            .proposalWeight(99.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                    .setName(field.getName())
                                    .setIsArray(field.getType().describe().contains("[]"))
                                    .setArgIndex(boolIdx)
                                    .build()).build());
                    boolIdx++;
                }
            }

            if(funcInstance instanceof BaseReduceOp || funcInstance instanceof BaseReduceBoolOp) {
                if(!containsProposalWithDescriptorName("keepDims",argDescriptorProposals)) {
                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .sourceOfProposal("java")
                            .proposalWeight(9999.0)
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                    .setName("keepDims")
                                    .setIsArray(false)
                                    .setArgIndex(boolIdx)
                                    .build()).build());
                }
            }

        } catch(Exception e) {
            e.printStackTrace();
        }
    }


    private static boolean containsProposalWithDescriptorName(String name,Collection<ArgDescriptorProposal> proposals) {
        for(ArgDescriptorProposal proposal : proposals) {
            if(proposal.getDescriptor().getName().equals(name)) {
                return true;
            }
        }

        return false;
    }

    private static ResolvedFieldDeclaration getResolve(FieldDeclaration input) {
        try {
            return input.resolve();
        }catch(Exception e) {
            return null;
        }
    }


    private  SourceRoot initSourceRoot(File nd4jApiRootDir) {
        CombinedTypeSolver typeSolver = new CombinedTypeSolver();
        typeSolver.add(new ReflectionTypeSolver(false));
        typeSolver.add(new JavaParserTypeSolver(nd4jApiRootDir));
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(typeSolver);
        StaticJavaParser.getConfiguration().setSymbolResolver(symbolSolver);
        SourceRoot sourceRoot = new SourceRoot(nd4jApiRootDir.toPath(),new ParserConfiguration().setSymbolResolver(symbolSolver));
        return sourceRoot;
    }

    @Override
    public Map<String, List<ArgDescriptorProposal>> getProposals() {
        return doReflectionsExtraction();
    }

    @Override
    public OpNamespace.OpDescriptor.OpDeclarationType typeFor(String name) {
        return opTypes.get(name);
    }
}

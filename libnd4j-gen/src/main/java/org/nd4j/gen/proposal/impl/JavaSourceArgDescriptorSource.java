package org.nd4j.gen.proposal.impl;

import com.codepoetics.protonpack.StreamUtils;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
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
import org.nd4j.common.primitives.Pair;
import org.nd4j.gen.OpDeclarationDescriptor;
import org.nd4j.gen.proposal.ArgDescriptorProposal;
import org.nd4j.gen.proposal.ArgDescriptorSource;
import org.nd4j.gen.proposal.utils.ArgDescriptorParserUtils;
import org.nd4j.ir.OpNamespace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.reduce.bp.BaseReductionBp;
import org.nd4j.linalg.api.ops.impl.reduce3.BaseReduce3Op;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;
import org.reflections.Reflections;

import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class JavaSourceArgDescriptorSource implements ArgDescriptorSource {


    private  SourceRoot sourceRoot;
    private File nd4jOpsRootDir;
    private double weight;
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
        Set<String> opNamesForDifferentialFunction = new HashSet<>();


        for(Class<? extends DifferentialFunction> clazz : subTypesOf) {
            if(Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface()) {
                continue;
            }

            try {
                DifferentialFunction differentialFunction = clazz.newInstance();
                String name = differentialFunction.opName();

                if(name == null)
                    continue;
                opNamesForDifferentialFunction.add(name);
                if(!(differentialFunction instanceof DynamicCustomOp))
                    opTypes.put(name,OpNamespace.OpDescriptor.OpDeclarationType.LEGACY_XYZ);
                else
                    opTypes.put(name,OpNamespace.OpDescriptor.OpDeclarationType.CUSTOM_OP_IMPL);


                String fileName = clazz.getName().replace(".",File.separator);
                StringBuilder fileBuilder = new StringBuilder();
                fileBuilder.append(fileName);
                fileBuilder.append(".java");

                // Our sample is in the root of this directory, so no package name.
                CompilationUnit cu = sourceRoot.parse(clazz.getPackage().getName(), clazz.getSimpleName() + ".java");
                List<ResolvedConstructorDeclaration> collect = cu.findAll(ConstructorDeclaration.class).stream()
                        .map(input -> input.resolve())
                        .distinct()
                        .collect(Collectors.toList());
                List<ArgDescriptorProposal> argDescriptorProposals = ret.get(name);
                if(argDescriptorProposals == null) {
                    argDescriptorProposals = new ArrayList<>();
                    ret.put(name,argDescriptorProposals);
                }

                Set<ResolvedParameterDeclaration> parameters = new HashSet<>();
                for(ResolvedConstructorDeclaration parameterDeclaration : collect) {
                    for(int i = 0; i < parameterDeclaration.getNumberOfParams(); i++) {
                        ResolvedParameterDeclaration param = parameterDeclaration.getParam(i);
                        parameters.add(param);
                    }
                }


                int currIntIdx = 0;
                int currFloatIdx = 0;
                int currInputIdx = 0;
                int currOutputIdx = 0;
                Set<Pair<String, String>> typesAndParams = parameters.stream().map(collectedParam ->
                        Pair.of(collectedParam.describeType(), collectedParam.getName()))
                        .collect(Collectors.groupingBy(input -> input.getSecond()))
                        .entrySet().stream().map(nameToListOfTypes -> Pair.of(nameToListOfTypes.getKey(),nameToListOfTypes.getValue().get(0)))
                        .map(inputPair -> inputPair.getSecond())
                        .collect(Collectors.toSet());


                for(Pair<String,String> parameter : typesAndParams) {
                    if(parameter.getFirst().equals(SDVariable.class.getName())
                            || parameter.getFirst().equals(INDArray.class.getName())
                            || parameter.getFirst().equals(INDArray.class.getName() + "[]")
                            || parameter.getFirst().equals(SDVariable.class.getName() + "[]")) {
                        if(parameter.getSecond().equals("output") || parameter.getSecond().equals("z") || parameter.getSecond().equals("outputs")) {
                            argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                            .setArgType(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
                                            .setName(parameter.getSecond())
                                            .setArgIndex(currOutputIdx)
                                            .build()).build());
                            currOutputIdx++;

                        } else {
                            argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                            .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                            .setName(parameter.getSecond())
                                            .setArgIndex(currInputIdx)
                                            .build()).build());
                            currInputIdx++;
                        }
                    } else if(parameter.getFirst().equals(int.class.getName()) ||
                            parameter.getFirst().equals("int...")
                            || parameter.getFirst().equals(long.class.getName()) ||
                            parameter.getFirst().equals("long...")
                            || parameter.getFirst().equals(boolean.class.getName())) {
                        argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                        .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                        .setName(parameter.getSecond())
                                        .setArgIndex(currIntIdx)
                                        .build()).build());
                        currIntIdx++;
                    } else if(parameter.getFirst().equals(double.class.getName()) || parameter.getFirst().equals(float.class.getName())) {
                        argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                        .setArgType(OpNamespace.ArgDescriptor.ArgType.FLOAT)
                                        .setName(parameter.getSecond())
                                        .setArgIndex(currFloatIdx)
                                        .build()).build());
                        currFloatIdx++;
                    }
                }


                currIntIdx = 0;
                currFloatIdx = 0;
                currInputIdx = 0;
                currOutputIdx = 0;
                List<ResolvedFieldDeclaration> fields = cu.findAll(FieldDeclaration.class).stream()
                        .map(input -> input.resolve()).collect(Collectors.toList());
                for(ResolvedFieldDeclaration field : fields) {
                    if(field.getType().describe().equals(SDVariable.class.getName())
                            || field.getType().describe().equals(INDArray.class.getName())
                            || field.getType().describe().equals(INDArray.class.getName() + "[]")
                            || field.getType().describe().equals(SDVariable.class.getName() + "[]")) {
                        if(field.getName().equals("output") || field.getName().equals("z")  || field.getName().equals("outputs")) {
                            argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                            .setArgType(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
                                            .setName(field.getName())
                                            .setArgIndex(currOutputIdx)
                                            .build()).build());
                            currOutputIdx++;
                        } else {
                            argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                    .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                            .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                            .setName(field.getName())
                                            .setArgIndex(currInputIdx)
                                            .build()).build());
                            currInputIdx++;
                        }
                    } else if(field.getType().describe().equals(int.class.getName()) ||
                            field.getType().describe().equals("int...") ||
                            field.getType().describe().equals("int[]")
                            || field.getType().describe().equals(long.class.getName()) ||
                            field.getType().describe().equals("long...") ||
                            field.getType().describe().equals("long[]") ||
                            field.getType().describe().equals("boolean[]")
                            || field.getType().describe().equals(boolean.class.getName())) {
                        argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                        .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                        .setName(field.getName())
                                        .setArgIndex(currIntIdx)
                                        .build()).build());
                        currIntIdx++;
                    } else if(field.getType().describe().equals(double.class.getName()) || field.getType().toString().equals(float.class.getName())) {
                        argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                        .setArgType(OpNamespace.ArgDescriptor.ArgType.FLOAT)
                                        .setName(field.getName())
                                        .setArgIndex(currFloatIdx)
                                        .build()).build());
                        currFloatIdx++;
                    }
                }

                if(differentialFunction instanceof BaseReduceOp) {
                    int idx = 0;
                    //set to max possible index to avoid clashes
                    for(ArgDescriptorProposal argDescriptorProposal : argDescriptorProposals) {
                        OpNamespace.ArgDescriptor.ArgType argType = argDescriptorProposal.getDescriptor().getArgType();
                        if(argType == OpNamespace.ArgDescriptor.ArgType.BOOL || argType == OpNamespace.ArgDescriptor.ArgType.INT64) {
                            idx = Math.max(idx,argDescriptorProposal.getDescriptor().getArgIndex());
                        }
                    }


                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setName("input")
                                    .setArgIndex(0)
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                    .build())
                            .proposalWeight(999.0)
                            .build());

                    if(differentialFunction instanceof BaseReduce3Op)
                        argDescriptorProposals.add(ArgDescriptorProposal.builder()
                                .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                        .setName("y")
                                        .setArgIndex(1)
                                        .setArgType(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
                                        .build())
                                .proposalWeight(2999.0)
                                .build());

                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setName("z")
                                    .setArgIndex(0)
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
                                    .build())
                            .proposalWeight(2999.0)
                            .build());

                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setName("dimensions")
                                    .setArgIndex(0)
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.INT64)
                                    .build())
                            .proposalWeight(2999.0)
                            .build());

                    argDescriptorProposals.add(ArgDescriptorProposal.builder()
                            .descriptor(OpNamespace.ArgDescriptor.newBuilder()
                                    .setConvertBoolToInt(false)
                                    .setName("keepDims")
                                    .setArgIndex(0)
                                    .setArgType(OpNamespace.ArgDescriptor.ArgType.BOOL)
                                    .build())
                            .proposalWeight(2999.0)
                            .build());
                }



            } catch(Exception e) {
                e.printStackTrace();
            }

        }

        return ret;
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

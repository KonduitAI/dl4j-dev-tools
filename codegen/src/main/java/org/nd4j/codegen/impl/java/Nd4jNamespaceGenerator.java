package org.nd4j.codegen.impl.java;

import com.squareup.javapoet.CodeBlock;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeSpec;
import org.nd4j.base.Preconditions;
import org.nd4j.codegen.api.*;
import org.nd4j.codegen.api.doc.DocSection;
import org.nd4j.codegen.api.doc.DocTokens;
import org.nd4j.codegen.api.generator.ConstraintCodeGenerator;
import org.nd4j.codegen.api.generator.GeneratorConfig;
import org.nd4j.codegen.util.GenUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

import javax.lang.model.element.Modifier;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class Nd4jNamespaceGenerator {

    private static ConstraintCodeGenerator constraintCodeGenerator = new JavaConstraintCodeGenerator();

    private Nd4jNamespaceGenerator() { }

    public static void generate(NamespaceOps namespace, GeneratorConfig config, File directory) throws IOException {

        String className = "Nd4j" + GenUtil.ensureFirstIsCap(namespace.getName());

        TypeSpec.Builder builder = TypeSpec.classBuilder(className)
                .addModifiers(Modifier.PUBLIC);

        //Add private no-arg constructor
        MethodSpec noArg = MethodSpec.constructorBuilder()
                .addModifiers(Modifier.PRIVATE)
                .build();

        builder.addMethod(noArg);

        //Add ops
        List<Op> ops = new ArrayList<>();
        for(Op o : namespace.getOps()){
            if(o.isAbstract())
                continue;
            ops.add(o);
        }

        //Also add includes:
        List<String> include = namespace.getInclude();
        if(include != null && !include.isEmpty()){
            throw new UnsupportedOperationException("Handling generation with includes not yet supported");
        }

        Collections.sort(ops, Comparator.comparing(Op::getOpName));

        //Add ops:
        for(Op o : ops){
            builder.addMethod(creatorMethod(o));
        }

        TypeSpec ts = builder.build();

        JavaFile jf = JavaFile.builder("org.nd4j.linalg.api.ops", ts)
                .addFileComment("********** GENERATED CODE - DO NOT MODIFY THIS FILE **********")
                .build();

        jf.writeTo(directory);
    }

    private static MethodSpec creatorMethod(Op op){
        MethodSpec.Builder c = MethodSpec.methodBuilder(GenUtil.ensureFirstIsNotCap(op.getOpName()))
                .addModifiers(Modifier.PUBLIC, Modifier.STATIC);

        if(op.getArgs() != null && !op.getArgs().isEmpty()){
            final Arg lastArg = op.getArgs().get(op.getArgs().size() - 1);
            if(lastArg.getCount() != null && !(lastArg.getCount() instanceof Exactly && ((Exactly) lastArg.getCount()).getCount() == 1)){
                c.varargs(true);
            }
        }

        boolean singleOut = op.getOutputs().size() == 1;
        if(singleOut){
            c.returns(INDArray.class);
        } else {
            c.returns(INDArray[].class);
        }


        //Method javadoc:
        List<DocSection> doc = op.getDoc();
        if(doc != null && !doc.isEmpty()){
            for(DocSection ds : doc){
                if(ds.applies(Language.JAVA, CodeComponent.OP_CREATOR)){
                    String text = DocTokens.processDocText(ds.getText(), op);
                    //Add <br> tags at the end of each line, where none already exists
                    String[] lines = text.split("\n");
                    for( int i=0; i<lines.length; i++ ){
                        if(!lines[i].endsWith("<br>")){
                            lines[i] = lines[i] + "<br>";
                        }
                    }
                    text = String.join("\n", lines);
                    c.addJavadoc(text + "\n\n");
                }
            }
        }


        // Document Constraints:
        final List<Constraint> constraints = op.getConstraints();
        if(constraints != null && !constraints.isEmpty()){
            c.addJavadoc("Inputs must satisfy the following constraints: <br>\n");
            for (Constraint constraint : constraints) {
                c.addJavadoc(constraint.getMessage() +": " + constraintCodeGenerator.generateExpression(constraint.getCheck()) + "<br>\n");
            }

            c.addJavadoc("\n");
        }

        //Inputs:
        List<Input> in = op.getInputs();
        List<String> inNames = new ArrayList<>();
        if(in != null && !in.isEmpty()){
            Map<DataType, String> validationMapping = new HashMap<>();
            validationMapping.put(DataType.BOOL, "validateBool");
            validationMapping.put(DataType.FLOATING_POINT, "validateFloatingPoint");
            validationMapping.put(DataType.NUMERIC, "validateNumerical");
            validationMapping.put(DataType.INT, "validateInteger");

            for(Input i : in){
//                inNames.add(i.getName());

                if( i.getCount() == null || (i.getCount() instanceof Exactly && ((Exactly) i.getCount()).getCount() == 1)) {
                    //Single input
                    c.addParameter(INDArray.class, i.getName());
                } else {
                    //Array input
                    c.addParameter(INDArray[].class, i.getName());
                }
                c.addJavadoc("@param " + i.getName() + " " + (i.getDescription() == null ? "" : DocTokens.processDocText(i.getDescription(), op)) + " (" + i.getType() + " type)\n");
                // Check for parameter types
                c.addStatement(CodeBlock.of("$T.$L($S, $L)", NDValidation.class, validationMapping.get(i.getType()), op.getOpName(), i.getName()));
                // Check for parameter counts
                if(i.getCount() != null && !(i.getCount() instanceof Exactly && ((Exactly) i.getCount()).getCount() == 1)){
                    final Count count = i.getCount();
                    final String errorMessage = i.getName() + " has incorrect size/length. Expected: " + count.toString() + ", got %s";
                    if(count instanceof Exactly){
                        c.addStatement(CodeBlock.of("$T.checkArgument($L.length == $L, $S, $L)", Preconditions.class, i.getName(), ((Exactly) count).getCount(), errorMessage, i.getName() + ".length"));
                    }else if(count instanceof AtLeast){
                        c.addStatement(CodeBlock.of("$T.checkArgument($L.length >= $L, $S, $L)", Preconditions.class, i.getName(), ((AtLeast) count).getMin(), errorMessage, i.getName() + ".length"));
                    }else if(count instanceof AtMost){
                        c.addStatement(CodeBlock.of("$T.checkArgument($L.length <= $L, $S, $L)", Preconditions.class, i.getName(), ((AtMost) count).getMax(), errorMessage, i.getName() + ".length"));
                    }else if(count instanceof Range){
                        c.addStatement(CodeBlock.of("$T.checkArgument($L.length >= $L && $L.length <= $L, $S, $L)", Preconditions.class, i.getName(), ((Range) count).getFrom(), i.getName(), ((Range) count).getTo(), errorMessage, i.getName() + ".length"));
                    }
                }
            }
        }

        //Args:
        List<Arg> args = op.getArgs();
        Map<DataType, Class> typeMapping = new HashMap<>();
        typeMapping.put(DataType.BOOL, boolean.class);
        typeMapping.put(DataType.FLOATING_POINT, double.class);
        typeMapping.put(DataType.NUMERIC, double.class);
        typeMapping.put(DataType.INT, long.class);

        Map<DataType, Class> arrayTypeMapping = new HashMap<>();
        arrayTypeMapping.put(DataType.BOOL, boolean[].class);
        arrayTypeMapping.put(DataType.FLOATING_POINT, double[].class);
        arrayTypeMapping.put(DataType.NUMERIC, double[].class);
        arrayTypeMapping.put(DataType.INT, long[].class);

        if(args != null && !args.isEmpty()){
            final int size = args.size();
            for (int i = 0; i < size; i++) {
                Arg arg = args.get(i);
                inNames.add(arg.getName());
                if (arg.getCount() == null || (arg.getCount() instanceof Exactly && ((Exactly) arg.getCount()).getCount() == 1)) {
                    c.addParameter(typeMapping.get(arg.getType()), arg.getName());
                    c.addJavadoc("@param " + arg.getName() + " " + (arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(), op)) + "\n");
                } else {
                    final Count count = arg.getCount();
                    c.addJavadoc("@param " + arg.getName() + " " + (arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(), op)) + " (Size: " + count.toString() + ")\n");
                    c.addParameter(arrayTypeMapping.get(arg.getType()), arg.getName());

                    final String errorMessage = arg.getName() + " has incorrect size/length. Expected: " + count.toString() + ", got %s";
                    if (count instanceof Exactly) {
                        c.addStatement(CodeBlock.of("$T.checkArgument($L.length == $L, $S, $L)", Preconditions.class, arg.getName(), ((Exactly) count).getCount(), errorMessage, arg.getName()));
                    } else if (count instanceof AtLeast) {
                        c.addStatement(CodeBlock.of("$T.checkArgument($L.length >= $L, $S, $L)", Preconditions.class, arg.getName(), ((AtLeast) count).getMin(), errorMessage, arg.getName()));
                    } else if (count instanceof AtMost) {
                        c.addStatement(CodeBlock.of("$T.checkArgument($L.length <= $L, $S, $L)", Preconditions.class, arg.getName(), ((AtMost) count).getMax(), errorMessage, arg.getName()));
                    } else if (count instanceof Range) {
                        c.addStatement(CodeBlock.of("$T.checkArgument($L.length >= $L && $L.length <= $L, $S, $L)", Preconditions.class, arg.getName(), ((Range) count).getFrom(), arg.getName(), ((Range) count).getTo(), errorMessage, arg.getName()));
                    }
                }
            }
        }

        //Outputs:
        List<Output> outputs = op.getOutputs();
        if(outputs != null && !outputs.isEmpty()){
            if(outputs.size() == 1){
                Output o = outputs.get(0);
                c.addJavadoc("@return " + o.getName() + " " + (o.getDescription() == null ? "" : DocTokens.processDocText(o.getDescription(), op)) + " (" + o.getType() + " type)\n");
            } else {
                throw new UnsupportedOperationException("Javadoc for multi-output ops not yet implemented");
            }
        }

        // Constraints:
        if(constraints != null && !constraints.isEmpty()){
            // Don't materialize the Backend Constraints
            for (Constraint constraint : constraints.stream().filter(it -> !(it instanceof BackendConstraint)).collect(Collectors.toList())) {
                c.addStatement(CodeBlock.of("$T.checkArgument($L, $S)", Preconditions.class, constraintCodeGenerator.generateExpression(constraint.getCheck()), constraint.getMessage()));
            }
        }

        //Op execution:
        StringBuilder sb = new StringBuilder();
        sb.append("return $T.exec(new ")
                .append(op.getJavaPackage())
                .append(".")
                .append(GenUtil.ensureFirstIsCap(op.getOpName())).append("Op")
                .append("(")
                .append(String.join(", ", inNames))
                .append("))");
        if(singleOut)
            sb.append("[0]");

        c.addStatement(sb.toString(), Nd4j.class);

        return c.build();
    }
}

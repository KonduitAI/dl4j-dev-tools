package org.nd4j.codegen.impl.java;

import com.squareup.javapoet.CodeBlock;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeSpec;
import org.jetbrains.annotations.NotNull;
import org.nd4j.base.Preconditions;
import org.nd4j.codegen.api.*;
import org.nd4j.codegen.api.doc.DocSection;
import org.nd4j.codegen.api.doc.DocTokens;
import org.nd4j.codegen.api.generator.ConstraintCodeGenerator;
import org.nd4j.codegen.api.generator.GeneratorConfig;
import org.nd4j.codegen.util.GenUtil;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.lang.model.element.Modifier;
import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class Nd4jNamespaceGenerator {
    private static Map<DataType, Class> typeMapping = new HashMap<>();
    private static Map<DataType, Class> arrayTypeMapping = new HashMap<>();
    private static Map<DataType, String> validationMapping = new HashMap<>();
    private static Count exactlyOne = new Exactly(1);

    static {
        typeMapping.put(DataType.BOOL, boolean.class);
        typeMapping.put(DataType.FLOATING_POINT, double.class);
        typeMapping.put(DataType.NUMERIC, double.class);
        typeMapping.put(DataType.INT, long.class);

        arrayTypeMapping.put(DataType.BOOL, boolean[].class);
        arrayTypeMapping.put(DataType.FLOATING_POINT, double[].class);
        arrayTypeMapping.put(DataType.NUMERIC, double[].class);
        arrayTypeMapping.put(DataType.INT, long[].class);

        validationMapping.put(DataType.BOOL, "validateBool");
        validationMapping.put(DataType.FLOATING_POINT, "validateFloatingPoint");
        validationMapping.put(DataType.NUMERIC, "validateNumerical");
        validationMapping.put(DataType.INT, "validateInteger");
    }

    private static ConstraintCodeGenerator constraintCodeGenerator = new JavaConstraintCodeGenerator();

    private Nd4jNamespaceGenerator() { }

    public static void generate(NamespaceOps namespace, GeneratorConfig config, File directory, String className) throws IOException {

        TypeSpec.Builder builder = TypeSpec.classBuilder(className)
                .addModifiers(Modifier.PUBLIC);

        addDefaultConstructor(builder);

        //Add ops
        namespace.getOps()
                .stream()
                .filter(it -> !it.isAbstract())
                .sorted(Comparator.comparing(Op::getOpName))
                .forEachOrdered(o -> builder.addMethod(creatorMethod(o)));


        TypeSpec ts = builder.build();

        JavaFile jf = JavaFile.builder("org.nd4j.linalg.factory.ops", ts)
                .addFileComment("********** GENERATED CODE - DO NOT MODIFY THIS FILE **********")
                .addStaticImport(NDValidation.class, "isSameType")
                .build();

        jf.writeTo(directory);

        // Copy Utilities that haven't yet landed in ND4J too
//        final InputStream resource = Nd4jNamespaceGenerator.class.getClassLoader().getResourceAsStream("java/org/nd4j/linalg/api/NDValidation.java");
//        final InputStream resource = Nd4jNamespaceGenerator.class.getClassLoader().getResourceAsStream("org/nd4j/linalg/api/NDValidation.java");
//        File f = new File(directory, "NDValidation.java");
//        final BufferedOutputStream outputStream = new BufferedOutputStream(new FileOutputStream(f));
//        IOUtils.copy(resource, outputStream);
//        outputStream.close();
//        resource.close();
    }

    private static void addDefaultConstructor(TypeSpec.Builder builder) {
        //Add private no-arg constructor
        MethodSpec noArg = MethodSpec.constructorBuilder()
                .addModifiers(Modifier.PUBLIC)
                .build();

        builder.addMethod(noArg);
    }

    private static MethodSpec creatorMethod(Op op){
        MethodSpec.Builder c = MethodSpec.methodBuilder(GenUtil.ensureFirstIsNotCap(op.getOpName()))
                .addModifiers(Modifier.PUBLIC);

        enableVarargsOnLastArg(c, op);

        buildJavaDoc(op, c);
        List<String> inNames = buildParameters(c, op);
        buildConstraints(c, op);
        buildExecution(c, op, inNames);

        return c.build();
    }



    private static void  buildJavaDoc(Op op, MethodSpec.Builder c) {
        //Method javadoc:
        List<DocSection> doc = op.getDoc();
        if(doc != null && !doc.isEmpty()){
            for(DocSection ds : doc){
                if(ds.applies(Language.JAVA, CodeComponent.OP_CREATOR)){
                    String text = DocTokens.processDocText(ds.getText(), op, DocTokens.GenerationType.ND4J);
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

        List<Input> in = op.getInputs();
        if(in != null && !in.isEmpty()){
            for(Input i : in){
                c.addJavadoc("@param " + i.getName() + " " + (i.getDescription() == null ? "" : DocTokens.processDocText(i.getDescription(), op, DocTokens.GenerationType.ND4J)) + " (" + i.getType() + " type)\n");
            }
        }

        List<Arg> args = op.getArgs();
        if(args != null && !args.isEmpty()){
            for (Arg arg : args) {
                final Count count = arg.getCount();
                if (count == null || count.equals(exactlyOne)) {
                    c.addJavadoc("@param " + arg.getName() + " " + (arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(), op, DocTokens.GenerationType.ND4J)) + "\n");
                } else {
                    c.addJavadoc("@param " + arg.getName() + " " + (arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(), op, DocTokens.GenerationType.ND4J)) + " (Size: " + count.toString() + ")\n");
                }
            }
        }

        //Outputs:
        List<Output> outputs = op.getOutputs();
        if(outputs != null && !outputs.isEmpty()){
            if(outputs.size() == 1){
                Output o = outputs.get(0);
                c.addJavadoc("@return " + o.getName() + " " + (o.getDescription() == null ? "" : DocTokens.processDocText(o.getDescription(), op, DocTokens.GenerationType.ND4J)) + " (" + o.getType() + " type)\n");
            } else {
                throw new UnsupportedOperationException("Javadoc for multi-output ops not yet implemented");
            }
        }
    }

    @NotNull
    private static List<String> buildParameters(MethodSpec.Builder c, Op op) {
        //Inputs:
        List<Input> in = op.getInputs();
        List<String> inNames = new ArrayList<>();
        if(in != null && !in.isEmpty()){
            for(Input i : in){
                final String inputName = i.getName();
                inNames.add(inputName);

                final Count count = i.getCount();
                if(count == null || count.equals(exactlyOne)) {
                    //Single input
                    c.addParameter(INDArray.class, inputName);
                } else {
                    //Array input
                    c.addParameter(INDArray[].class, inputName);
                }
                // Check for parameter types
                c.addStatement(CodeBlock.of("$T.$L($S, $S, $L)", NDValidation.class, validationMapping.get(i.getType()), op.getOpName(), inputName, inputName));
                checkParameterCount(c, count, inputName);
            }
        }

        //Args:
        List<Arg> args = op.getArgs();

        if(args != null && !args.isEmpty()){
            for (Arg arg : args) {
                final String argName = arg.getName();
                inNames.add(argName);

                final Count count = arg.getCount();
                if (count == null || count.equals(exactlyOne)) {
                    // single arg
                    c.addParameter(typeMapping.get(arg.getType()), argName);
                } else {
                    // array Arg
                    c.addParameter(arrayTypeMapping.get(arg.getType()), argName);
                }

                checkParameterCount(c, count, argName);
            }
        }
        return inNames;
    }

    private static void buildConstraints(MethodSpec.Builder c, Op op) {
        // Constraints:
        final List<Constraint> constraints = op.getConstraints();
        if(constraints != null && !constraints.isEmpty()){
            // Don't materialize the Backend Constraints
            for (Constraint constraint : constraints.stream().filter(it -> !(it instanceof BackendConstraint)).collect(Collectors.toList())) {
                c.addStatement(CodeBlock.of("$T.checkArgument($L, $S)", Preconditions.class, constraintCodeGenerator.generateExpression(constraint.getCheck()), constraint.getMessage()));
            }
        }
    }

    private static void buildExecution(MethodSpec.Builder c, Op op, List<String> inNames) {
        boolean singleOut = op.getOutputs().size() == 1;
        if(singleOut){
            c.returns(INDArray.class);
        } else {
            c.returns(INDArray[].class);
        }

        //Op execution:
        StringBuilder sb = new StringBuilder();
        sb.append("return $T.exec(new ")
                .append(op.getJavaPackage())
                .append(".")
                .append(op.getJavaOpClass() == null ? GenUtil.ensureFirstIsCap(op.getOpName()) + "Op" : op.getJavaOpClass())
                .append("(")
                .append(String.join(", ", inNames))
                .append("))");
        if(singleOut)
            sb.append("[0]");

        c.addStatement(sb.toString(), Nd4j.class);
    }

    private static void enableVarargsOnLastArg(MethodSpec.Builder c, Op op) {
        if(op.getArgs() != null && !op.getArgs().isEmpty()){
            final Arg lastArg = op.getArgs().get(op.getArgs().size() - 1);
            final Count count = lastArg.getCount();
            if(count != null && !count.equals(exactlyOne)){
                c.varargs(true);
            }
        }
    }

    private static String countToJava(Count count,String paramName) {
        final String paramLength = paramName + ".length";
        if(count instanceof Exactly){
            return paramLength + " == " + ((Exactly) count).getCount();
        }else if(count instanceof AtLeast){
            return paramLength + " >= " + ((AtLeast) count).getMin();
        }else if(count instanceof AtMost){
            return paramLength + " <= "+ ((AtMost) count).getMax();
        }else if(count instanceof Range){
            return ((Range) count).getFrom() + " <= " + paramLength + " && " + paramLength + " <= " + ((Range) count).getTo();
        }else{
            throw new IllegalArgumentException("Can not deal with Count of type " + count.getClass().getName());
        }
    }

    private static void checkParameterCount(MethodSpec.Builder c, Count count, String paramName) {
        // Check for parameter counts
        if(count != null && !count.equals(exactlyOne)){
            final String errorMessage = paramName + " has incorrect size/length. Expected: " + countToJava(count, paramName) + ", got %s";
            if(count instanceof Exactly){
                c.addStatement(CodeBlock.of("$T.checkArgument($L.length == $L, $S, $L)", Preconditions.class, paramName, ((Exactly) count).getCount(), errorMessage, paramName + ".length"));
            }else if(count instanceof AtLeast){
                c.addStatement(CodeBlock.of("$T.checkArgument($L.length >= $L, $S, $L)", Preconditions.class, paramName, ((AtLeast) count).getMin(), errorMessage, paramName + ".length"));
            }else if(count instanceof AtMost){
                c.addStatement(CodeBlock.of("$T.checkArgument($L.length <= $L, $S, $L)", Preconditions.class, paramName, ((AtMost) count).getMax(), errorMessage, paramName + ".length"));
            }else if(count instanceof Range){
                c.addStatement(CodeBlock.of("$T.checkArgument($L.length >= $L && $L.length <= $L, $S, $L)", Preconditions.class, paramName, ((Range) count).getFrom(), paramName, ((Range) count).getTo(), errorMessage, paramName + ".length"));
            }
        }
    }
}

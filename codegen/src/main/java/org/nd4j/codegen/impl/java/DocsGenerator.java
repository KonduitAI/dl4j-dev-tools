package org.nd4j.codegen.impl.java;

import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeName;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.codegen.api.*;
import org.nd4j.codegen.api.doc.DocSection;
import org.nd4j.codegen.api.doc.DocTokens;
import org.nd4j.codegen.util.GenUtil;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static org.nd4j.codegen.impl.java.Nd4jNamespaceGenerator.exactlyOne;

public class DocsGenerator {

    // Markdown marker for start-end of code section
    private static final String MD_CODE = "```";
    // Javadoc constants which should be dropped or replaced for markdown generation
    private static final String JD_CODE = "{@code ";
    private static final String JD_CODE_END = "}";
    private static final String JD_INPUT_TYPE = "%INPUT_TYPE%";

    public static class JavaDocToMDAdapter {
        private String current;

        public JavaDocToMDAdapter(String original) {
            this.current = original;
        }

        public JavaDocToMDAdapter filter(String pattern, String replaceWith) {
            String result =  StringUtils.replace(current, pattern, replaceWith);
            this.current = result;
            return this;
        }

        @Override
        public String toString() {
            return current;
        }
    }

    private static String generateMethodText(Op op, Signature s, boolean isSameDiff, boolean isLoss, boolean withName) {
        StringBuilder sb = new StringBuilder();
        MethodSpec.Builder c = MethodSpec.methodBuilder(GenUtil.ensureFirstIsNotCap(op.getOpName()));
        List<Parameter> params = s.getParameters();
        List<Output> outs = op.getOutputs();
        String retType = "void";

        if (outs.size() == 1) {
            retType = isSameDiff ? "SDVariable" : "INDArray";
        }
        else if (outs.size() >= 1) {
            retType = isSameDiff ? "SDVariable[]" : "INDArray[]";
        }
        sb.append(retType).append(" ").append(op.getOpName()).append("(");
        boolean first = true;
        for (Parameter param : params) {
            if (param instanceof Arg) {
                Arg arg = (Arg) param;
                if (!first)
                    sb.append(", ");
                else if (withName)
                    sb.append("String name, ");
                String className;
                if(arg.getType() == DataType.ENUM) {
                    className = GenUtil.ensureFirstIsCap(arg.name());
                } else {
                    TypeName tu = Nd4jNamespaceGenerator.getArgType(arg);
                    className = tu.toString();
                }
                if(className.contains(".")){
                    className = className.substring(className.lastIndexOf('.')+1);
                }
                sb.append(className).append(" ").append(arg.name());
                first = false;
            }
            else if (param instanceof Input) {
                Input arg = (Input) param;
                if (!first)
                    sb.append(", ");
                else if (withName)
                    sb.append("String name, ");
                sb.append(isSameDiff ? "SDVariable " : "INDArray ").append(arg.name());
                first = false;
            }
        }
        sb.append(")");
        return sb.toString();
    }

    private static StringBuilder buildDocSectionText(List<DocSection> docSections) {
        StringBuilder sb = new StringBuilder();
        for (DocSection ds : docSections) {
            //if(ds.applies(Language.JAVA, CodeComponent.OP_CREATOR)){
            String text = ds.getText();
            String[] lines = text.split("\n");
            for (int i = 0; i < lines.length; i++) {
                if (!lines[i].endsWith("<br>")) {
                    String filteredLine = new JavaDocToMDAdapter(lines[i])
                            .filter(JD_CODE, "`")
                            .filter(JD_CODE_END, "`")
                            .filter(JD_INPUT_TYPE, "INDArray").toString();

                    lines[i] = filteredLine + System.lineSeparator();
                }
            }
            text = String.join("\n", lines);
            sb.append(text).append(System.lineSeparator());
            //}
        }
        return sb;
    }

    public static void generateDocs(NamespaceOps namespace, String docsDirectory, String basePackage) throws IOException {
        File outputDirectory = new File(docsDirectory);
        StringBuilder sb = new StringBuilder();
        sb.append("#  Namespace ").append(namespace.getName()).append(System.lineSeparator());
        List<Op> ops = namespace.getOps();

        ops.sort(Comparator.comparing(Op::getOpName));

        if (ops.size() > 0)
            sb.append("# Operation classes <ops>").append(System.lineSeparator());
        for (Op op : ops) {
            sb.append("## <a name=" + "\"").append(op.name()).append("\"></a>").append(op.name()).append(System.lineSeparator());
            List<DocSection> doc = op.getDoc();
            if(!doc.isEmpty()) {
                boolean first = true;
                for(Signature s : op.getSignatures()) {
                    if (first) {
                        Language lang = doc.get(0).getLanguage();
                        sb.append(MD_CODE).append(lang.equals(Language.ANY) ? Language.JAVA : lang).append(System.lineSeparator());
                        first = false;
                    }
                    String ndCode = generateMethodText(op, s, false, false, false);
                    sb.append(ndCode).append(System.lineSeparator());
                    String sdCode = generateMethodText(op, s, true, false, false);
                    sb.append(sdCode).append(System.lineSeparator());
                    String withNameCode = generateMethodText(op, s, true, false, true);
                    sb.append(withNameCode).append(System.lineSeparator());
                }
                sb.append(MD_CODE).append(System.lineSeparator());
                StringBuilder tsb = buildDocSectionText(doc);
                sb.append(tsb.toString());
                List<Signature> l = op.getSignatures();
                for(Signature s : l) {
                    List<Parameter> params = s.getParameters();
                    for (Parameter p : params) {
                        if(p instanceof Input){
                            Input i = (Input)p;
                            sb.append("* **").append(i.getName()).append("** - ").append(i.getDescription() == null ? "" : DocTokens.processDocText(i.getDescription(),
                                    op, DocTokens.GenerationType.ND4J)).append(" (").append(i.getType()).append(" type)").append(System.lineSeparator());
                        } else if(p instanceof Arg) {
                            Arg arg = (Arg) p;
                            final Count count = arg.getCount();
                            if (count == null || count.equals(exactlyOne)) {
                                sb.append("* **").append(arg.getName()).append("** - ").append(arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(),
                                        op, DocTokens.GenerationType.ND4J));    //.append(System.lineSeparator());
                            } else {
                                sb.append("* **").append(arg.getName()).append("** - ").append(arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(),
                                        op, DocTokens.GenerationType.ND4J)).append(" (Size: ").append(count.toString());    //.append(System.lineSeparator());
                            }

                            Object defaultValue = arg.defaultValue();
                            if(defaultValue != null){
                                sb.append(" - default = ").append(formatDefaultValue(defaultValue));
                            }

                            sb.append(System.lineSeparator());
                        }
                    }
                }
                sb.append(System.lineSeparator());
            }
        }


        if (namespace.getConfigs().size() > 0)
            sb.append("# Configuration Classes <configs>").append(System.lineSeparator());
        for (Config config : namespace.getConfigs()) {
            sb.append("## ").append(config.getName()).append(System.lineSeparator());
            for (Input i : config.getInputs()) {
                sb.append("* **").append(i.getName()).append("**- ").append(i.getDescription()).append(" (").append(i.getType()).append(" type)");
                if (i.hasDefaultValue() && (i.defaultValue() != null))
                    sb.append(" Default value:").append(formatDefaultValue(i.defaultValue())).append(System.lineSeparator());
                else
                    sb.append(System.lineSeparator());
            }
            for (Arg arg : config.getArgs()) {
                sb.append("* **").append(arg.getName()).append("** - ").append(arg.getDescription()).append(" (").append(arg.getType()).append(" type)");
                if (arg.hasDefaultValue() && (arg.defaultValue() != null))
                    sb.append(" - default = ").append(formatDefaultValue(arg.defaultValue())).append(System.lineSeparator());
                else
                    sb.append(System.lineSeparator());
            }
            StringBuilder tsb = buildDocSectionText(config.getDoc());
            sb.append(tsb.toString());
            sb.append(System.lineSeparator());
            ops.stream().filter(op -> op.getConfigs().contains(config)).forEach(op ->
                    sb.append("[").append(op.getOpName()).append("]").append("(#").append(op.getOpName()).append(")").append(System.lineSeparator()));

        }
        File outFile = new File(outputDirectory + "/ops", "/" + namespace.getName() + ".md");
        FileUtils.writeStringToFile(outFile, sb.toString(), StandardCharsets.UTF_8);
    }

    private static String formatDefaultValue(Object v){
        if(v == null){ return "null"; }
        else if(v instanceof int[]){ return Arrays.toString((int[]) v); }
        else if(v instanceof long[]){ return Arrays.toString((long[]) v); }
        else if(v instanceof float[]){ return Arrays.toString((float[]) v); }
        else if(v instanceof double[]){ return Arrays.toString((double[]) v); }
        else if(v instanceof boolean[]){ return Arrays.toString((boolean[]) v); }
        else if(v instanceof Input){ return ((Input)v).getName(); }
        else if(v instanceof org.nd4j.linalg.api.buffer.DataType){ return "DataType." + v; }
        else if(v instanceof LossReduce || v instanceof org.nd4j.autodiff.loss.LossReduce){ return "LossReduce." + v; }
        else return v.toString();
    }
}

package org.nd4j.codegen.impl.java;

import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeName;
import org.apache.commons.io.FileUtils;
import org.nd4j.codegen.api.*;
import org.nd4j.codegen.api.doc.DocSection;
import org.nd4j.codegen.api.doc.DocTokens;
import org.nd4j.codegen.util.GenUtil;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.nd4j.codegen.impl.java.Nd4jNamespaceGenerator.exactlyOne;

public class DocsGenerator {

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
        sb.append(retType + " " + op.getOpName() + "(");
        boolean first = true;
        for (Parameter param : params) {
            if (param instanceof Arg) {
                Arg arg = (Arg) param;
                if (!first)
                    sb.append(",");
                else if (withName)
                    sb.append("String name,");
                TypeName tu = Nd4jNamespaceGenerator.getArgType(arg);
                sb.append(tu.toString() + " " + arg.name());
                first = false;
            }
            else if (param instanceof Input) {
                Input arg = (Input) param;
                if (!first)
                    sb.append(",");
                else if (withName)
                    sb.append("String name,");
                sb.append((isSameDiff ? "SDVariable " : "INDArray ") + arg.name());
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
                    lines[i] = lines[i] + System.lineSeparator();
                }
            }
            text = String.join("\n", lines);
            sb.append(text + System.lineSeparator());
            //}
        }
        return sb;
    }

    public static void generateDocs(NamespaceOps namespace, String docsDirectory, String basePackage) throws IOException {
        File outputDirectory = new File(docsDirectory);
        StringBuilder sb = new StringBuilder();
        sb.append("#  Namespace " + namespace.getName() + System.lineSeparator());
        List<Op> ops = namespace.getOps();
        for (Op op : ops) {
            sb.append("## <a name=" + "\"" + op.name() + "\"></a>" + op.name()  + System.lineSeparator());
            List<DocSection> doc = op.getDoc();
            if(!doc.isEmpty()) {
                boolean first = true;
                for(Signature s : op.getSignatures()) {
                    if (first) {
                        Language lang = doc.get(0).getLanguage();
                        sb.append("````" + (lang.equals(Language.ANY) ? Language.JAVA : lang ) + System.lineSeparator());
                        first = false;
                    }
                    String ndCode = generateMethodText(op, s, false, false, false);
                    sb.append(ndCode + System.lineSeparator());
                    String sdCode = generateMethodText(op, s, true, false, false);
                    sb.append(sdCode + System.lineSeparator());
                    String withNameCode = generateMethodText(op, s, true, false, true);
                    sb.append(withNameCode + System.lineSeparator());
                }
                sb.append("````" + System.lineSeparator());
                StringBuilder tsb = buildDocSectionText(doc);
                sb.append(tsb.toString());
                List<Signature> l = op.getSignatures();
                for(Signature s : l) {
                    List<Parameter> params = s.getParameters();
                    for (Parameter p : params) {
                        if(p instanceof Input){
                            Input i = (Input)p;
                            sb.append("* " + i.getName() + " " + (i.getDescription() == null ? "" : DocTokens.processDocText(i.getDescription(),
                                    op, DocTokens.GenerationType.ND4J)) + " (" + i.getType() + " type)" + System.lineSeparator());
                        } else if(p instanceof Arg) {
                            Arg arg = (Arg) p;
                            final Count count = arg.getCount();
                            if (count == null || count.equals(exactlyOne)) {
                                sb.append("* " + arg.getName() + " " + (arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(),
                                        op, DocTokens.GenerationType.ND4J)) +  System.lineSeparator());
                            } else {
                                sb.append("* " + arg.getName() + " " + (arg.getDescription() == null ? "" : DocTokens.processDocText(arg.getDescription(),
                                        op, DocTokens.GenerationType.ND4J)) + " (Size: " + count.toString() +  System.lineSeparator());
                            }
                        }
                    }
                }
                sb.append(System.lineSeparator());
                tsb = buildDocSectionText(doc);
                sb.append(tsb.toString());
            }
        }

        for (Config config : Registry.INSTANCE.configs()) {
            sb.append("## " + config.getName()  + System.lineSeparator());
            boolean first = true;
            for (Input i : config.getInputs()) {
                if (first) {
                    sb.append("````" + System.lineSeparator());
                    first = false;
                }
                sb.append("* " + i.getName() + " " + i.getDescription() + " (" + i.getType() + " type)" + System.lineSeparator());
            }
            for (Arg arg : config.getArgs()) {
                if (first) {
                    sb.append("````" + System.lineSeparator());
                    first = false;
                }
                sb.append("* " + arg.getName() + " " + " (" + arg.getType() + " type)" + System.lineSeparator());
            }
            StringBuilder tsb = buildDocSectionText(config.getDoc());
            sb.append(tsb.toString());
            sb.append("````" + System.lineSeparator());
            ops.stream().filter(op -> op.getConfigs().contains(config)).forEach(op ->
                    sb.append("[" + op.getOpName() + "]" + "(#" + op.getOpName() + ")" + System.lineSeparator()));
        }
        File outFile = new File(outputDirectory + "/ops", "/" + namespace.getName() + ".md");
        FileUtils.writeStringToFile(outFile, sb.toString(), StandardCharsets.UTF_8);
    }
}

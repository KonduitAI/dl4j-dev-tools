package org.nd4j.codegen.cli;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.codegen.Namespace;
import org.nd4j.codegen.SameDiffNamespace;
import org.nd4j.codegen.api.NamespaceOps;
import org.nd4j.codegen.impl.java.Nd4jNamespaceGenerator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Planned CLI for generating classes
 */
@Slf4j
public class CLI {
    private static final String relativePath = "nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/";

    @Parameter(names = "-dir", description = "Root directory of deeplearning4j mono repo", required = true)
    private String repoRootDir;

    @Parameter(names = "-namespaces", description = "List of namespaces to generate, or 'ALL' to generate all namespaces", required = true)
    private List<String> namespaces;

    enum NS_PROJECT {
        ND4J,
        SAMEDIFF;
    }

    private void generateNamespaces(NS_PROJECT project, File outputDir, String basePackage) throws IOException {

        List<Namespace> nd4jNamespaces = new ArrayList<>();
        List<SameDiffNamespace> sdNamespaces = new ArrayList<>();

        for(String s : namespaces) {
            if ("all".equalsIgnoreCase(s)) {
                if (project == NS_PROJECT.ND4J) {
                    Collections.addAll(nd4jNamespaces, Namespace.values());
                } else {
                    Collections.addAll(sdNamespaces, SameDiffNamespace.values());
                }
                break;
            }
            Object ns = null;
            if (project == NS_PROJECT.ND4J) {
                ns = Namespace.fromString(s);
                nd4jNamespaces.add((Namespace)ns);
            }
            else {
                ns = SameDiffNamespace.fromString(s);
                sdNamespaces.add((SameDiffNamespace) ns);
            }

            if (ns == null) {
                throw new IllegalStateException("Invalid/unknown SD namespace provided: " + s);
            }
        }

        int cnt = 0;
        for (int i = 0; i < (NS_PROJECT.ND4J == project ? nd4jNamespaces.size() : sdNamespaces.size()); ++i) {
            Object ns = NS_PROJECT.ND4J == project ? nd4jNamespaces.get(i) : sdNamespaces.get(i);
            log.info("Starting generation of namespace: {}", ns);

            String javaClassName = project == NS_PROJECT.ND4J ? ((Namespace)ns).javaClassName() : ((SameDiffNamespace)ns).javaClassName();
            NamespaceOps ops = project == NS_PROJECT.ND4J ? ((Namespace)ns).getNamespace() : ((SameDiffNamespace)ns).getNamespace();

            String basePackagePath = basePackage.replace(".", "/") + "/ops/";
            File outputPath = new File(outputDir,  basePackagePath + javaClassName + ".java");
            log.info("Output path: {}", outputPath.getAbsolutePath());

            Nd4jNamespaceGenerator.generate(ops, null, outputDir, javaClassName, basePackage);
            ++cnt;
        }
        log.info("Complete - generated {} namespaces", cnt);
    }


    public static void main(String[] args) throws Exception {
        new CLI().runMain(args);
    }

    public void runMain(String[] args) throws Exception {
        JCommander.newBuilder()
                .addObject(this)
                .build()
                .parse(args);

        //First: Check root directory.
        File dir = new File(repoRootDir);
        if(!dir.exists() || !dir.isDirectory()){
            throw new IllegalStateException("Provided root directory does not exist (or not a directory): " + dir.getAbsolutePath());
        }

        File outputDir =  new File(dir, relativePath);
        if(!outputDir.exists() || !dir.isDirectory()){
            throw new IllegalStateException("Expected output directory does not exist: " + outputDir.getAbsolutePath());
        }

        if(namespaces == null || namespaces.isEmpty()){
            throw new IllegalStateException("No namespaces were provided");
        }

        try {
            generateNamespaces(NS_PROJECT.ND4J, outputDir, "org.nd4j.linalg.factory");
            generateNamespaces(NS_PROJECT.SAMEDIFF, outputDir, "org.nd4j.autodiff.samediff");
        } catch (Exception e) {
            log.error(e.toString());
        }
    }
}

package org.nd4j.codegen.cli;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.codegen.Namespace;
import org.nd4j.codegen.api.NamespaceOps;
import org.nd4j.codegen.impl.java.Nd4jNamespaceGenerator;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Planned CLI for generating classes
 */
@Slf4j
public class CLI {
    private static final String relativePath = "nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/ops";


    @Parameter(names = "-dir", description = "Root directory of deeplearning4j mono repo", required = true)
    private String repoRootDir;

    @Parameter(names = "-namespaces", description = "List of namespaces to generate, or 'ALL' to generate all namespaces", required = true)
    private List<String> namespaces;



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

        boolean all = false;
        List<Namespace> n = new ArrayList<>();
        for(String s : namespaces){
            if("all".equalsIgnoreCase(s)){
                all = true;
                Collections.addAll(n, Namespace.values());
                break;
            }

            Namespace ns = Namespace.fromString(s);
            if(ns == null){
                throw new IllegalStateException("Invalid/unknown namespace provided: " + s);
            }
        }

        for(Namespace ns : n){
            log.info("Starting generation of namespace: {}", ns);

            File outputPath = new File(dir, ns.javaClassName() + ".java");
            log.info("Output path: {}", outputPath.getAbsolutePath());

            NamespaceOps ops = ns.getNamespace();

            Nd4jNamespaceGenerator.generate(ops, null, dir, ns.javaClassName());
        }

        log.info("Complete - generated {} namespaces", n.size());
    }
}

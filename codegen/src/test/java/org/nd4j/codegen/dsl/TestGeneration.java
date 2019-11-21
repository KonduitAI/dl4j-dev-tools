package org.nd4j.codegen.dsl;

import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.codegen.api.NamespaceOps;
import org.nd4j.codegen.impl.java.Nd4jNamespaceGenerator;
import org.nd4j.codegen.ops.BitwiseKt;
import org.nd4j.codegen.ops.RandomKt;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class TestGeneration {

    @TempDir
    public File testDir;

    @Test
    public void test() throws Exception {
        File f = testDir;

        List<NamespaceOps> list = Arrays.asList(BitwiseKt.Bitwise(), RandomKt.Random());

        for(NamespaceOps ops : list) {
            Nd4jNamespaceGenerator.generate(ops, null, f);
        }

        File[] files = f.listFiles();
        Iterator<File> iter = FileUtils.iterateFiles(f, null, true);
        if(files != null) {
            while(iter.hasNext()){
                File file = iter.next();
                if(file.isDirectory())
                    continue;
                System.out.println(FileUtils.readFileToString(file, StandardCharsets.UTF_8));
                System.out.println("\n\n================\n\n");
            }
        }
    }

}

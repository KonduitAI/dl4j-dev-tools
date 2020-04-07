package ai.konduit;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Strings;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.io.FileUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.*;

public class ConvertTF2ONNX {

    public static final String UNSUPPORTED_OPS = "Unsupported ops";

    @Parameter
    private String baseDir = "/dl4j-test-resources/src/main/resources/tf_graphs";
//    private String baseDir = "C:/DL4J/git/dl4j-test-resources/src/main/resources/tf_graphs";

    public static void main(String[] args) throws Exception {
        new ConvertTF2ONNX().run(args);
    }

    public void run(String[] args) throws Exception {
        JCommander.newBuilder()
                .addObject(this)
                .args(args)
                .build();

        long startTime = System.currentTimeMillis();
        Collection<File> files = FileUtils.listFiles(new File(baseDir), new String[]{"pb"}, true);

        System.out.println("Found " + files.size() + " .pb files");

        int countSuccess8 = 0;
        int countSuccess11 = 0;


        for(File f : files){
            System.out.println("===============================================================================================================");
            System.out.println(f);

            File dir = f.getParentFile();
            String dirAbs = dir.getAbsolutePath();
            Collection<File> dirFiles = FileUtils.listFiles(dir, null, true);
            List<String> inputNames = new ArrayList<>();
            List<String> outputNames = new ArrayList<>();
            for(File f2 : dirFiles){
                String fileAbs = f2.getAbsolutePath();
                if(fileAbs.contains("prediction.csv")){
                    int from = dirAbs.length()+1;
                    int to = fileAbs.indexOf("prediction.csv")-1;
                    String outVarName = f2.getAbsolutePath().substring(from, to).replaceAll("\\\\", "/");       //Get relative path
                    if(outVarName.matches(".*\\.\\d+")){
                        int idx = outVarName.lastIndexOf('.');
                        String start = outVarName.substring(0, idx);
                        outVarName = start + ":" + outVarName.substring(idx+1);
                    } else {
                        outVarName = outVarName + ":0";
                    }
                    outputNames.add(outVarName);
                }
                //TODO placeholder inputs... but: vast majority of our "single op" test models don't have placeholders, however...
            }

            Collections.sort(outputNames);

            String inPath = f.getAbsolutePath();
            String outPath8 = inPath.substring(0, inPath.length()-3) + "_opset8.onnx";
            String outPath11 = inPath.substring(0, inPath.length()-3) + "_opset11.onnx";

            String[] cmd8 = getCmd(inPath, outPath8, outputNames, 8);
            String[] cmd11 = getCmd(inPath, outPath11, outputNames, 11);

            System.out.println(String.join(" ", cmd8));
            ProcessBuilder pb = new ProcessBuilder(cmd8);

            Pair p = launchAndGetOutput(pb);
            System.out.println(p.getOutput());

            File out8 = new File(outPath8);
            File out11 = new File(outPath11);

            boolean ok8 = p.getRc() == 0;
            if(p.getRc() > 0 && out8.exists()){
                out8.delete();
                System.out.println("Exit code > 0 and file exists - deleted file: " + out8.getAbsolutePath());
            }

            if(p.getOutput().contains(UNSUPPORTED_OPS)){
                out8.delete();
                System.out.println("Deleting converted file - Unsupported ops detected: " + out8.getAbsolutePath());
                ok8 = false;
            }

            if(ok8)
                countSuccess8++;


            System.out.println(String.join(" ", cmd11));
            pb = new ProcessBuilder(cmd11);
            p = launchAndGetOutput(pb);
            System.out.println(p.getOutput());

            boolean ok11 = p.getRc() == 0;
            if(p.getRc() > 0 && out11.exists()){
                out11.delete();
                System.out.println("Exit code > 0 and file exists - deleted file: " + out11.getAbsolutePath());
            }

            if(p.getOutput().contains(UNSUPPORTED_OPS)){
                out11.delete();
                System.out.println("Deleting converted file - Unsupported ops detected: " + out8.getAbsolutePath());
                ok11 = false;
            }

            if(ok11)
                countSuccess11++;
        }

        long end = System.currentTimeMillis();

        System.out.println("===== CONVERSION COMPLETE IN " + (end-startTime) + " MS =====");
        System.out.println("Files: " + files.size() + " - opset 8 converted successfully: " + countSuccess8 + "; opset 11 converted successfully: " + countSuccess11);
    }

    private Pair launchAndGetOutput(ProcessBuilder pb) throws Exception {
        Process process = pb.start();
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder sb = new StringBuilder();
        int rc = process.waitFor();
        String line = null;
        while ( (line = reader.readLine()) != null) {
            sb.append(line).append("\n");
        }
        reader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
        while ( (line = reader.readLine()) != null) {
            sb.append(line).append("\n");
        }
        return new Pair(sb.toString(), rc);
    }

    @Data
    @AllArgsConstructor
    private static class Pair {
        private String output;
        private int rc;
    }

    private static String[] getCmd(String inPath, String outPath, List<String> outputNames, int opset){
        String[] cmd = new String[]{
                "python3", "-m", "tf2onnx.convert", "--graphdef", inPath, "--output", outPath, "--inputs", "", "--outputs", Strings.join(",", outputNames), "--opset", String.valueOf(opset)};
        return cmd;
    }
}

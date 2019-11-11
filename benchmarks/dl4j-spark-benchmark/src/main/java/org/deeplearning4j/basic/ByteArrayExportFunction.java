package org.deeplearning4j.basic;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.VoidFunction;
import org.deeplearning4j.util.UIDProvider;

import java.net.URI;
import java.util.Iterator;

public class ByteArrayExportFunction implements VoidFunction<Iterator<byte[]>> {
    private static final Configuration conf = new Configuration();

    private final URI outputDir;
    private String uid = null;

    private int outputCount;

    public ByteArrayExportFunction(URI outputDir) {
        this.outputDir = outputDir;
    }

    @Override
    public void call(Iterator<byte[]> iter) throws Exception {
        String jvmuid = UIDProvider.getJVMUID();
        uid = Thread.currentThread().getId() + jvmuid.substring(0,Math.min(8,jvmuid.length()));


        while(iter.hasNext()){
            byte[] next = iter.next();

            String filename = "bytearray_" + uid + "_" + (outputCount++) + ".bin";

            URI uri = new URI(outputDir.getPath() + "/" + filename);
            FileSystem file = FileSystem.get(uri, conf);
            try(FSDataOutputStream out = file.create(new Path(uri))){
                out.write(next);
            }
        }
    }
}

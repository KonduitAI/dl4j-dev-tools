# How to generate a .csv file with information on the ignored tests in the various projects


For dl4j:
```
./summarize-ignores deeplearning4j results
```

This will write two files into results/deeplearning4j:
ignored_classes.csv
ignored_methods.csv

Both files can be opened up to be viewed in excel/numbers

Sample information from files:

```
head -3 java-testing/ignores/results/nd4j/ignored_classes.csv 
IGNORED/ALL IGNORED: PROJECT, PACKAGE NAME, TEST CLASS, TEST METHOD, IGNORED COMMENT, LOCAL PATH TO TEST CLASS
ALL IGNORED: nd4j, org.nd4j.jita.allocator, AllocatorTest, *,"AB 2019/05/23 - Getting stuck (tests never finishing) on CI - see issue #7657", /Users/susaneraly/SKYMIND/deeplearning4j/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/test/java/org/nd4j/jita/allocator/AllocatorTest.java
ALL IGNORED: nd4j, org.nd4j.autodiff.samediff, FailingSameDiffTests, *,"AB 2019/05/21 - JVM Crash on ppc64 - Issue #7657", /Users/susaneraly/SKYMIND/deeplearning4j/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/autodiff/samediff/FailingSameDiffTests.java

head -3 java-testing/ignores/results/nd4j/ignored_methods.csv
IGNORED/ALL IGNORED: PROJECT, PACKAGE NAME, TEST CLASS, TEST METHOD, IGNORED COMMENT, LOCAL PATH TO TEST CLASS
IGNORED: nd4j, org.nd4j.autodiff.execution, GraphExecutionerTest, testConversion,, /Users/susaneraly/SKYMIND/deeplearning4j/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/autodiff/execution/GraphExecutionerTest.java
IGNORED: nd4j, org.nd4j.autodiff.execution, GraphExecutionerTest, testSums1,, /Users/susaneraly/SKYMIND/deeplearning4j/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/autodiff/execution/GraphExecutionerTest.java
```

# Generate a .csv file with information on the ignored tests in the various projects


For arbiter:
```
./summarize-ignores arbiter results
....
INFO:Checked test file
/Users/susaneraly/SKYMIND/deeplearning4j/arbiter/arbiter-ui/src/test/java/org/deeplearning4j/arbiter/optimize/TestBasic.java...
INFO:Finished running subdir
/Users/susaneraly/SKYMIND/deeplearning4j/arbiter/arbiter-ui/src/test


INFO: Run completed.
====================================================================
Ignored classes count:1. Written to: results/arbiter/ignored_classes.csv
Ignored methods count: 10. Written to: results/arbiter/ignored_methods.csv
Log file written to summarize-ignores-arbiter.log
```
 ignored_classes.csv, contains the list of test classes with a @Ignore for the whole class with other pertinent information. 
 ignored_methods.csv, contains the list of test method with a @Ignore for the whole class with other pertinent information. 

Refer sample first three lines below. Note the csv includes the string inside @Ignore("XX") if present

To run other projects repeat as follows:
```
./summarize-ignores nd4j results
./summarize-ignores deeplearning4j results
./summarize-ignores datavec results
#./summarize-ignores arbiter results
```


Both files can be opened up to be viewed in excel/numbers. Sample information from files:
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

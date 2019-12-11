# Generate csv files with information on ignored tests

Usage: 
  run-summarize-ignores.sh <Relative Path to check> <Path for results> [DL4J
repo path]
                                                                                
  If the optional third argument in not provided, script will check the parent
dir of the dl4j-dev-tools repo for the deeplearning4j repo.                       
                                                                                
  Example usage:                                                                
  run-summarize-ignores.sh nd4j results                                         
  run-summarize-ignores.sh datavec/datavec-spark results_only_datavec_spark     
  run-summarize-ignores.sh nd4j results path_to_my_dl4j_repo 

For nd4j:
```
./run-summarize-ignores.sh nd4j ~/Desktop/ignore-results/.
....
INFO: Run completed.
========================================================================================================================================
Ignored classes count:29. Written to:
/Users/susaneraly/Desktop/ignore-results/./nd4j/ignored_classes.csv
Ignored methods count: 60. Written to:
/Users/susaneraly/Desktop/ignore-results/./nd4j/ignored_methods.csv
Ignore failing tests count:36. Written to:
/Users/susaneraly/Desktop/ignore-results/./nd4j/ignore_failing_methods.csv

	Log file written to
/Users/susaneraly/Desktop/ignore-results/./nd4j/summarize-ignores.log
========================================================================================================================================

```
 ignored_classes.csv, contains the list of test classes with a @Ignore for the whole class with other pertinent information. 
 ignored_methods.csv, contains the list of test method with a @Ignore for the whole class with other pertinent information. 
 ignore_failing_methods.csv, contains the list of test methods that have a OpValidationSuite.ignoreFailing() or a check for OpValidationSuite.IGNORE_FAILING in it

All files can be opened up to be viewed in excel/numbers and  includes the string inside @Ignore("XX") if present
Sample information from files:

```
head -3 ~/Desktop/ignore-results/nd4j/ignored_classes.csv 
IGNORED/ALL IGNORED/IGNORE FAILING, PACKAGE NAME, TEST CLASS, TEST METHOD,IGNORED COMMENT, LOCAL PATH TO TEST CLASS
ALL IGNORED, org.nd4j.jita.allocator, AllocatorTest, *,"AB 2019/05/23 - Getting stuck (tests never finishing) on CI - see issue #7657", /Users/susaneraly/SKYMIND/deeplearning4j/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/test/java/org/nd4j/jita/allocator/AllocatorTest.java
ALL IGNORED, org.nd4j.autodiff.samediff, FailingSameDiffTests, *,"AB 2019/05/21 - JVM Crash on ppc64 - Issue #7657", /Users/susaneraly/SKYMIND/deeplearning4j/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/autodiff/samediff/FailingSameDiffTests.java

head -3 ~/Desktop/ignore-results/nd4j/ignored_methods.csv 
IGNORED/ALL IGNORED/IGNORE FAILING, PACKAGE NAME, TEST CLASS, TEST METHOD,IGNORED COMMENT, LOCAL PATH TO TEST CLASS
IGNORED, org.nd4j.autodiff.execution, GraphExecutionerTest, testConversion,, /Users/susaneraly/SKYMIND/deeplearning4j/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/autodiff/execution/GraphExecutionerTest.java
IGNORED, org.nd4j.autodiff.execution, GraphExecutionerTest, testSums1,, /Users/susaneraly/SKYMIND/deeplearning4j/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/autodiff/execution/GraphExecutionerTest.java

head -3 ~/Desktop/ignore-results/nd4j/ignore_failing_methods.csv 
IGNORED/ALL IGNORED/IGNORE FAILING, PACKAGE NAME, TEST CLASS, TEST METHOD,IGNORED COMMENT, LOCAL PATH TO TEST CLASS
IGNORE FAILING, org.nd4j.autodiff.execution, GraphExecutionerTest, testEquality2,, /Users/susaneraly/SKYMIND/deeplearning4j/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/autodiff/execution/GraphExecutionerTest.java
IGNORE FAILING, org.nd4j.autodiff.execution, GraphExecutionerTest, testEquality1,, /Users/susaneraly/SKYMIND/deeplearning4j/nd4j/nd4j-backends/nd4j-tests/src/test/java/org/nd4j/autodiff/execution/GraphExecutionerTest.java
```

                                                                                
To run other projects repeat as follows:                                        
```                                                                             
./run-summarize-ignores.sh nd4j ~/Desktop/ignore-results/.                      
./run-summarize-ignores.sh deeplearning4j ~/Desktop/ignore-results/.            
./run-summarize-ignores.sh datavec ~/Desktop/ignore-results/.                   
./run-summarize-ignores.sh arbiter ~/Desktop/ignore-results/.                   
```     

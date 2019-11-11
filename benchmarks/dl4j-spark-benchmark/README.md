# deeplearning4j-spark-benchmark

### Deeplearning4j Spark Benchmarks

The Spark network training tests in this module are based on synthetic data,
and are fully parameterized with respect to configuration options such as
network and data sizes, etc.

Can be run either on Spark local or via spark-submit, depending on the launch configuration

### Available Tests

Currently the following tests (all using synthetic data) are available:

- MLPTest: A simple Multi-Layer Perceptron, with 2 hidden layers and an output layer
- RNNTest: A LSTM RNN test, with 2 LSTM layers and a RNN output layer

A CNN test is partially complete.

All tests can be configured with options such as:

- Number of network parameters
- Data input vector size
- Number of DataSet objects
- Number of examples (minibatch size) in each DataSet object

For full details, see the launch parameters in RunTrainingTests.java

### Building Uber-Jars

For building uber-jar for Spark local:
mvn package -DskipTests -P sparklocal

For building uber-jar for Spark submit:
mvn package -DskipTests

### Launching

Example launch shell scripts are available under /scripts.

The main entry point is org.deeplearning4j.train.RunTrainingTests.java
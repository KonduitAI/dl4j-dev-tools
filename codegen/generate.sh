#!/bin/bash

mvn clean package -DskipTests
java -cp target/codegen-1.0.0-SNAPSHOT-shaded.jar org.nd4j.codegen.cli.CLI -dir ../../deeplearning4j -namespaces "$@"
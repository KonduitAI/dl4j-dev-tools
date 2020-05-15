FROM continuumio/miniconda3:latest
RUN conda install tensorflow=1.15.0
#RUN conda install tensorflow=2.0
RUN conda install psutil
ENV DL4J_TEST_RESOURCES=/dl4j-test-resources

#Usage:
#docker build --tag tfops:latest .
#docker run -v C:\DL4J\Git\dl4j-dev-tools\import-tests:/ImportTests -v C:/DL4J/Git/dl4j-test-resources:/dl4j-test-resources -v C:/Temp/TF_Graphs/:/TF_Graphs/ -it tfops:latest
#cd ImportTests
#python setup.py develop
#python tests/test_ops.py
#python model_zoo/util/zoo_evaluation.py
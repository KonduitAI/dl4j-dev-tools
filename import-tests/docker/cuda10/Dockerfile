#nvida-docker: https://github.com/NVIDIA/nvidia-docker

FROM tensorflow/tensorflow:1.15.0-gpu-py3
RUN apt-get install -y wget
# RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
# RUN bash Anaconda3-2019.10-Linux-x86_64.sh -b
# RUN rm Anaconda3-2019.10-Linux-x86_64.sh
# RUN conda install tensorflow=1.15.0
# RUN ~/anaconda3/bin/conda install psutil
RUN pip install psutil
ENV DL4J_TEST_RESOURCES=/dl4j-test-resources

#Usage:
#docker build --tag tfops:cuda .
#docker run -v C:\DL4J\Git\dl4j-dev-tools\import-tests:/ImportTests -v C:/DL4J/Git/dl4j-test-resources:/dl4j-test-resources -v C:/Temp/TF_Graphs/:/TF_Graphs/ -it tfops:cuda
#docker run --gpus 1 -v ~/DL4J/dl4j-dev-tools/import-tests:/ImportTests -v ~/DL4J/dl4j-test-resources:/dl4j-test-resources -v ~/TF_Graphs/:/TF_Graphs/ -it tfops:cuda
#cd ImportTests
#python setup.py develop
#python tests/test_ops.py
#python model_zoo/util/zoo_evaluation.py
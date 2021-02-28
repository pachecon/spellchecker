#python:3.6-stretch
#FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04
#pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime 
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
LABEL maintainer="Arlette Perez <arlettepe892@gmail.com>"

# install build utilities
RUN apt-get update 
# 	&& \ 
#	apt-get install -y gcc 
#make apt-transport-https ca-certificates build-essential

#RUN apt-get -y install git
# check our python environment
RUN python3 --version
RUN pip --version

# Installing python dependencies
COPY requirements.txt .
RUN pip install pyxDamerauLevenshtein
RUN pip install --requirement requirements.txt

RUN mkdir -p /spellckr/src
RUN ls -la ./
WORKDIR  /spellckr/src
# Copy all the files from the project’s root to the working directory
COPY src/ /spellckr/src
RUN ls -la ./

RUN ls -ltr
#RUN ls -la /src/*
#RUN git clone https://github.com/pachecon/spellchecker.git
# set the working directory for containers
#RUN ls -la ./
# Running Python Application
CMD ["python3", "./main_koehler.py"]
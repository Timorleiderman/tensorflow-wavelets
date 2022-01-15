FROM python:3.8

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --upgrade pip
RUN pip3 install opencv-python
RUN pip3 install tensorflow
RUN pip3 install tensorflow-probability
RUN pip3 install tensorflow_addons
RUN pip3 install PyWavelets
RUN pip3 install psnr-hvsm
RUN pip3 install imageio
ENV PYTHONPATH /workspace/tensorflow_wavelets/src
RUN pip3 install numpy scipy
RUN pip3 install tensorflow-compression
RUN pip3 install matplotlib
RUN pip3 install ipykernel

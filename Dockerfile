FROM tensorflow/tensorflow:2.6.1-gpu

# RUN apt-get install -y --no-install-recommends wget
# RUN apt-key del 7fa2af80
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN apt-get update

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub 88
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub 20


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

RUN useradd -ms /bin/bash ubu-admin


RUN apt-get install git -y
RUN pip3 install jupyter
RUN pip3 install jupyterlab
RUN pip3 install notebook

EXPOSE 8888
# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]

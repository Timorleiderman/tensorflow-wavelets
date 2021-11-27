FROM nvcr.io/nvidia/tensorflow:21.10-tf2-py3

RUN apt-get update

RUN mkdir -p ~/Workspace cd ~/Workspace && \
	git clone https://github.com/Timorleiderman/tensorflow-wavelets.git && \
	cd tensorflow-wavelets
	
RUN apt-get install libgl1 -y

COPY requirments.txt requirments.txt

RUN pip3 install -r requirments.txt

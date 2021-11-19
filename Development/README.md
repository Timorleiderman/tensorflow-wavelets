// Prepare Enviroment
# Docker Tensorflow wavelets linux ubuntu20.04

sudo apt-get update

sudo apt-get install ca-certificates curl gnupg lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io

// verify:
sudo docker run hello-world

Now for the Nvidia docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update

sudo apt-get install -y nvidia-docker2

sudo systemctl restart docker

// verify:
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

// for tensorflow 1.15 to work with tensorflow compression:

sudo docker pull nvcr.io/nvidia/tensorflow:21.10-tf1-py3

// for tensorflow 2.5
sudo docker pull nvcr.io/nvidia/tensorflow:21.10-tf2-py3



docker build -t tensorflow-wavelets:1.0 .

docker run --privileged=true -v /mnt/:/mnt/ --gpus all --user 1000:1000 -p 6006:6006 -p 8080:8080 tensorflow-wavelets:1.0

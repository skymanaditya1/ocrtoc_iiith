docker rm -f ocrtoc1

docker run -i -d --gpus all --name ocrtoc1 --network host \
        --privileged -v /dev:/dev -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $HOME/OCRTOC_software_package:/root/ocrtoc_ws/src \
        -v $HOME/Desktop/docker:/root/upload \
        registry.cn-hangzhou.aliyuncs.com/ocrtoc2021/release:2.1
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ocrtoc1`

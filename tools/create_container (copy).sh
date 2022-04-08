docker rm -f sub30_6dof

docker run -i -d --gpus all --name sub30_6dof --network host \
        --privileged -v /dev:/dev -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $HOME/ocrtoc_iiith:/root/ocrtoc_ws/src \
        -v $HOME/Desktop/docker:/root/upload \
        registry.cn-hangzhou.aliyuncs.com/ocrtoc2021/release:2.1
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' sub30_6dof`

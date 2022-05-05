docker rm -f ocrtoc

docker run -i -d --gpus all --name ocrtoc --network host \
        --privileged -v /dev:/dev -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /home/vishal_mandadi/submissions_OCRTOC/30jan/ocrtoc_iiith:/root/ocrtoc_ws/src \
        -v $HOME/Desktop/docker:/root/upload \
        sapien_apr12:latest
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ocrtoc`

docker start ocrtoc1
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ocrtoc1`

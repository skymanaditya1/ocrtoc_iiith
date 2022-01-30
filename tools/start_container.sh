docker start sub30_6dof
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' sub30_6dof`

#!/bin/bash

# rm ~/ocrtoc_ws/src/result.txt

# tasks=('0-0' '1-1-1' '1-1-2' '1-1-3' '1-2-1' '1-2-2' '1-3-1' '1-4-1' '1-4-2' '1-5-1' '1-5-2' '1-5-3' '2-1-1' '2-1-2' '2-2-1' '2-2-2' '2-2-3' '2-2-4' '3-1-1' '3-1-2' '3-2-1' '3-2-2' '3-3-1' '3-3-2' '4-1-1' '4-1-2' '4-2-1' '4-2-2' '4-2-3' '4-2-4' '4-3-1' '4-3-2' '4-3-3' '4-3-4' '5-1-1' '5-1-2' '5-1-3' '5-2-1' '5-2-2' '5-3-1' '5-3-2' '5-3-3' '6-1-1' '6-1-2' '6-2-1' '6-2-2' '6-2-3' '6-3-1' '6-3-2' '6-3-3')

tasks=( '1-4-1' '1-5-1' '1-1-1' '1-1-2' '1-4-2' '1-5-1' '1-5-2' '1-5-3' '2-1-1' '2-2-1' '3-1-1' '3-2-1' '3-3-1' '4-1-1' '4-2-1' '4-3-1' '5-1-1' '5-2-1' '5-2-2' '5-3-1' '5-3-2' '5-3-3' '6-1-1' '6-1-2' '6-2-1' '6-2-2' '6-2-3' '6-3-1' '6-3-2' '6-3-3')

source ~/ocrtoc_ws/devel/setup.bash

for t in ${tasks[@]}; do
	echo $t starts now!
	roslaunch ocrtoc_task bringup_simulator_sapien.launch task_index:=$t &
  	process1=$!

	sleep 60

	source ~/ocrtoc_ws/devel/setup.bash
	roslaunch ocrtoc_task solution.launch &
	process2=$!

	sleep 60
	
	roslaunch ocrtoc_task trigger_and_evaluation.launch task_index:=$t

	kill -SIGINT $process2
	kill -SIGINT $process1
	sleep 120
	
	echo $t over!! Yooooo
done




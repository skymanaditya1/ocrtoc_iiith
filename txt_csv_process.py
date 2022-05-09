#! /usr/bin/env python3

import pandas as pd
df = pd.read_csv("/root/ocrtoc_ws/src/process_result.txt",delimiter='|')
df.to_csv('/root/ocrtoc_ws/src/process_result.csv')

text = open("/root/ocrtoc_ws/src/process_result.csv", "r")
text = ''.join([i for i in text]) \
    .replace("Processed:", " ")
text = ''.join([i for i in text]) \
    .replace("Task name:", "Task :")

x = open("/root/ocrtoc_ws/src/process_result.csv","w")
x.writelines(text)
x.close()


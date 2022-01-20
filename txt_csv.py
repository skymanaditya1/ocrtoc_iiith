#! /usr/bin/env python3

import pandas as pd
df = pd.read_csv("/root/ocrtoc_ws/src/result.txt",delimiter='|')
df.to_csv('/root/ocrtoc_ws/src/result.csv')

text = open("/root/ocrtoc_ws/src/result.csv", "r")
text = ''.join([i for i in text]) \
    .replace("Scene mean score:", " ")
text = ''.join([i for i in text]) \
    .replace("Task name:", "Task :")
text = ''.join([i for i in text]) \
    .replace("Time cost:", " ")
x = open("/root/ocrtoc_ws/src/result.csv","w")
x.writelines(text)
x.close()


import os,os.path
import pandas as pd
df = pd.DataFrame(columns=["lineitem","relevance"])

for root, dirs, files in os.walk(os.path.join(os.getcwd(),'data')):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(root,file),"r") as f:
                for line in f:
                    print(line)
                    df = df.append({"lineitem":line.strip()},ignore_index=True)
df.to_csv("nlp_data.csv")            
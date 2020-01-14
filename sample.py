#%%
import os


def walk(path, f, num):
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(file):
            walk(file, f, num)
        else:
            f.write(num+"/"+file+" "+num+"\n")


def record(cat):
    path0 = f"data1/{cat}/0"
    path1 = f"data1/{cat}/1"

    with open(f"data1/{cat}/{cat}.txt", "w") as f:
        walk(path0, f, "0")
        walk(path1, f, "1")


record("test")
record("train")
record("val")

# %%

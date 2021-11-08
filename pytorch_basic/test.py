import torch
import os

print(torch.__version__)
current_dir = os.getcwd()
print(current_dir)
print(os.path.join(current_dir, "a/b/c"))
print(os.path.join(current_dir, "./a"))
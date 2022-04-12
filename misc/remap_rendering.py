import os
import numpy as np
import shutil

import sys
in_dir = sys.argv[1]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

all_items = [x for x in os.listdir(in_dir) if 'raycolor.png' in x]
print(all_items)

initial = [str(i)+'.png' for i in range(len(all_items))]
initial = sorted(initial)

for i in range(len(initial)):
    shutil.copy2(os.path.join(in_dir, str(i)+'_raycolor.png'),
                 os.path.join(out_dir, initial[i][:-4]+'_raycolor.png'))



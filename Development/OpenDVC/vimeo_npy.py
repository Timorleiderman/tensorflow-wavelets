import os
import numpy as np
import fnmatch

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(root)
    return result

folder = find('im1.png', 'vimeo_septuplet/sequences/')
np.save('folder.npy', folder)
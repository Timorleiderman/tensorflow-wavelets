import sys
import os
import numpy as np
import fnmatch

from subprocess import Popen, PIPE



def encode_decode_bpg(path):
    
    in_img1 = path + "/im1.png"
    enc_img1 = path + "/im1_QP27.bpg"
    dec_img1 = path + "/im1_bpg444_QP27.png"
    process_enc = Popen([
                        "bpgenc",
                        "-f", "444",
                        "-m", "9",
                        in_img1,
                        "-o" , enc_img1,
                        "-q", "27"],
                        stdout=PIPE, stderr=PIPE)
    
    stdout, stderr = process_enc.communicate()
    if (stderr):
        print(stderr)
    
    process_dec = Popen([
                        "bpgdec",
                        enc_img1, 
                        "-o", dec_img1 
                        ],
                        stdout=PIPE, stderr=PIPE)
    
    stdout, stderr = process_dec.communicate()
    if (stderr):
        print(stderr)
        
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(root)
                encode_decode_bpg(root)
                sys.stdout.write('\r'+root + "decoded and encoded")
    return result

folder = find('im1.png', '/mnt/WindowsDev/DataSets/vimeo_septuplet/sequences/')
np.save('folder_cloud.npy', folder)

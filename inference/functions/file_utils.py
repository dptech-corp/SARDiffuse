import os

import warnings
warnings.filterwarnings("ignore")

type2extensions = {
    'tif':['tif','tiff'],
    'img':['png','jpeg','jpg'],
    'npy':['npy'],
    'dm':['dm4', 'dm3']
}

def get_extensions_by_type(file_type):
    return type2extensions[file_type]

def find_all_files(base_dir, extensions):
    files = []
    for root, ds, fs in os.walk(base_dir):
        for f in fs:
            if f.endswith(tuple(extensions)):
                fullname = os.path.join(root, f)
                files.append(fullname)
    return files
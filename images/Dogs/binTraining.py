from __future__ import print_function
import pandas as pd
import shutil
import os
import sys

labels = pd.read_csv('labels.csv')

# Create `train_sep` directory
train_dir = 'train/'
train_sep_dir = 'train_sep/'
if not os.path.exists(train_sep_dir):
    os.mkdir(train_sep_dir)

for filename, class_name in labels.values:
    # Create subdirectory with `class_name`
    if not os.path.exists(train_sep_dir + class_name):
        os.mkdir(train_sep_dir + class_name)
    src_path = train_dir + filename + '.jpg'
    dst_path = train_sep_dir + class_name + '/' + filename + '.jpg'
    try:
        shutil.copy(src_path, dst_path)
    except IOError as e:
        print('Unable to copy file {} to {}'
              .format(src_path, dst_path))
    except:
        print('When try copy file {} to {}, unexpected error: {}'
              .format(src_path, dst_path, sys.exc_info()))

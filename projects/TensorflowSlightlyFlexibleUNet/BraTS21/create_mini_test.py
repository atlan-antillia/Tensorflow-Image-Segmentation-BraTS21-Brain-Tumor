# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import glob
import random
import shutil
import traceback

def create_mini_test():
  image_datapath = "../../../dataset/BraTS21/test/images/"
  masks_datapath = "../../../dataset/BraTS21/test/masks/"

  mini_test= "./mini_test"
  if os.path.exists(mini_test):
    shutil.rmtree(mini_test)
  if not os.path.exists(mini_test):
    os.makedirs(mini_test)
  mini_test_images_dir = os.path.join(mini_test, "images")
  mini_test_masks_dir  = os.path.join(mini_test, "masks")

  if not os.path.exists(mini_test_images_dir):
    os.makedirs(mini_test_images_dir)

  if not os.path.exists(mini_test_masks_dir):
    os.makedirs(mini_test_masks_dir)

  image_files = glob.glob(image_datapath + "/*.jpg")
  random.seed(137)

  image_files = random.sample(image_files, 20)
  for image_file in image_files:
    basename = os.path.basename(image_file)
    mask_file = os.path.join(masks_datapath, basename)
    shutil.copy2(image_file, mini_test_images_dir)
    shutil.copy2(mask_file,  mini_test_masks_dir)

if __name__ == "__main__":
  try:
    create_mini_test()
  except:
    traceback.print_exc()

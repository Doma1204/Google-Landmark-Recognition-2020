from __future__ import print_function
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from argparse import ArgumentParser

_size = (224, 224)
try:
    from torchvision import transforms
    _preprocess = transforms.Compose([transforms.Resize(_size)])
    def resize_image(img_path, save_path):
        img = Image.open(img_path)
        img = _preprocess(img)

        new_img_folder = os.path.dirname(save_path)
        if not os.path.exists(new_img_folder):
            os.makedirs(new_img_folder)

        img.save(save_path)

    _torch = True
except:
    def resize_image(img_path, save_path):
        img = Image.open(img_path)
        img = img.resize(_size)

        new_img_folder = os.path.dirname(save_path)
        if not os.path.exists(new_img_folder):
            os.makedirs(new_img_folder)

        img.save(save_path)

    _torch = False

def resize_images(data_id, data_dir, output_dir, size=(224, 224), parallel=True, max_workers=None):
    global _size, _preprocess
    _size = size
    if _torch: _preprocess = transforms.Compose([transforms.Resize(_size)])

    if parallel:
        img_paths  = [os.path.join(data_dir  , id[0], id[1], id[2], id+".jpg") for id in data_id]
        save_paths = [os.path.join(output_dir, id[0], id[1], id[2], id+".jpg") for id in data_id]
        with ProcessPoolExecutor(max_workers) as executor:
            list(tqdm(executor.map(resize_image, img_paths, save_paths), total=len(data_id)))
    else:
        for id in data_id:
            resize_image(
                os.path.join(data_dir  , id[0], id[1], id[2], id+".jpg"),
                os.path.join(output_dir, id[0], id[1], id[2], id+".jpg")
            )

def _build_parser():
    parser = ArgumentParser(description='Resize images')
    parser.add_argument("csv_dir", help="directory of data csv file", type=str)
    parser.add_argument("input_dir", help="directory of input data", type=str)
    parser.add_argument("output_dir", help="directory of output data", type=str)
    parser.add_argument("-s", "--size", help="width and height of resized image, default 224", default=224, type=int, dest="size")
    return parser

def _resize_images(args):
    print("Reading CSV file")
    import pandas as pd
    ids = pd.read_csv(args.csv_dir)["id"]
    resize_images(ids, args.input_dir, args.output_dir, size=(args.size, args.size))

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    _resize_images(args)

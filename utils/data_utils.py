import os

from torch.utils.data import DataLoader, Dataset

import pandas as pd
from PIL import Image

def get_dataloader(csv_file, root_dir="./data/train", transform=None, **kwargs):
    with open(csv_file, "r") as file:
        data_df = pd.read_csv(file)
    
    train_df = data_df[data_df["dataset"] == 0].reset_index(drop=True)
    val_df   = data_df[data_df["dataset"] == 1].reset_index(drop=True)
    test_df  = data_df[data_df["dataset"] == 2].reset_index(drop=True)

    print("No. of train sample:", len(train_df))
    print("No. of val sample:", len(val_df))
    print("No. of test sample:", len(test_df))

    dataloaders = {
        "train": DataLoader(GoogleLandmarkDataset(
            train_df,
            root_dir=root_dir,
            transform=transform["train"] if transform is not None else None
        ), **kwargs),
        "val": DataLoader(GoogleLandmarkDataset(
            val_df,
            root_dir=root_dir,
            transform=transform["val"] if transform is not None else None
        ), **kwargs),
        "test": DataLoader(GoogleLandmarkDataset(
            test_df,
            root_dir=root_dir,
            transform=transform["test"] if transform is not None else None
        ), **kwargs),
    }

    distributions = {
        "train": train_df["landmark_id"].value_counts().to_numpy(),
        "val": val_df["landmark_id"].value_counts().to_numpy(),
        "test": test_df["landmark_id"].value_counts().to_numpy(),
    }

    return dataloaders, distributions

def get_dataloader_2(csv_file, root_dir="./data/train", transform=None, **kwargs):
    with open(csv_file, "r") as file:
        data_df = pd.read_csv(file)
    
    train_df = data_df[data_df["dataset"] == 0].reset_index(drop=True)
    val_df   = data_df[data_df["dataset"] == 1].reset_index(drop=True)
    test_df  = data_df[data_df["dataset"] == 2].reset_index(drop=True)

    try:
        num_classes = {
            "train": len(train_df["relabel"].unique()),
            "val": len(val_df["relabel"].unique()),
            "test": len(test_df["relabel"].unique()),
        }
    except:
        print("Cannot find column \"relabel\"")
        num_classes = {
            "train": len(train_df["landmark_id"].unique()),
            "val": len(val_df["landmark_id"].unique()),
            "test": len(test_df["landmark_id"].unique()),
        }
    
    dataset_size = {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
    }

    print("No. of train sample:", dataset_size["train"], "(", num_classes["train"], "classes)")
    print("No. of val sample:", dataset_size["val"], "(", num_classes["val"], "classes)")
    print("No. of test sample:", dataset_size["test"], "(", num_classes["test"], "classes)")

    dataloaders = {
        "train": DataLoader(GoogleLandmarkDataset(
            train_df,
            root_dir=root_dir,
            transform=transform["train"] if transform is not None else None
        ), **kwargs),
        "val": DataLoader(GoogleLandmarkDataset(
            val_df,
            root_dir=root_dir,
            transform=transform["val"] if transform is not None else None
        ), **kwargs),
        "test": DataLoader(GoogleLandmarkDataset(
            test_df,
            root_dir=root_dir,
            transform=transform["test"] if transform is not None else None
        ), **kwargs),
    }

    distributions = {
        "train": train_df["landmark_id"].value_counts().to_numpy(),
        "val": val_df["landmark_id"].value_counts().to_numpy(),
        "test": test_df["landmark_id"].value_counts().to_numpy(),
    }

    return dataloaders, dataset_size, num_classes, distributions

class GoogleLandmarkDataset(Dataset):
    def __init__(self, data, root_dir="./data/train", transform=None):
        self.paths = data["id"].apply(lambda id: os.path.join(root_dir, id[0], id[1], id[2], id+".jpg"))
        try:
            self.landmark_id = data["relabel"]
        except:
            print("Cannot find column \"relabel\"")
            self.landmark_id = data["landmark_id"]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, id):
        try:
            img = Image.open(self.paths[id])
        except Exception as e:
            print(f"Error occur when loading image\n{e}")

        if self.transform is not None:
            img = self.transform(img)

        return img, self.landmark_id[id]

def read_csv(name):
    with open(name, "r") as file:
        return pd.read_csv(file)

def merge_csv(train_csv, val_csv, test_csv, output_csv):
    train_df = read_csv(train_csv)
    val_df = read_csv(val_csv)
    test_df = read_csv(test_csv)

    train_df.insert(1, "dataset", 0, True)
    val_df.insert(1, "dataset", 1, True)
    test_df.insert(1, "dataset", 2, True)

    pd.concat((train_df, val_df, test_df)).to_csv(output_csv, index=False)

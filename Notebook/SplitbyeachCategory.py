import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset_by_category(
    dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
):
    """
    Splits a dataset into training, validation, and test sets by each category (class).

    Parameters:
    - dataset_dir: str, the directory containing class subfolders.
    - output_dir: str, the directory where the split data will be stored.
    - train_ratio: float, proportion of the dataset used for training.
    - val_ratio: float, proportion of the dataset used for validation.
    - test_ratio: float, proportion of the dataset used for testing.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in os.listdir(dataset_dir):
        category_dir = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_dir):
            continue

        # Get all files in the category
        files = os.listdir(category_dir)
        files = [f for f in files if os.path.isfile(os.path.join(category_dir, f))]

        # Split the files into train, validation, and test sets
        train_files, temp_files = train_test_split(
            files, test_size=(1 - train_ratio), random_state=42
        )
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            random_state=42,
        )

        # Create category subfolders in output_dir for train, val, and test
        for split, split_files in zip(
            ["train", "val", "test"], [train_files, val_files, test_files]
        ):
            split_dir = os.path.join(output_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)

            # Copy files to the respective split directory
            for file in split_files:
                src_path = os.path.join(category_dir, file)
                dest_path = os.path.join(split_dir, file)
                shutil.copy2(src_path, dest_path)

        print(f"Processed category '{category}':")
        print(f"  Training: {len(train_files)} images")
        print(f"  Validation: {len(val_files)} images")
        print(f"  Testing: {len(test_files)} images")


# Example usage
dataset_dir = "C:/Users/Desktop/Desktop/Thesis/Data/CacaoDataset"
output_dir = "C:/Users/Desktop/Desktop/Thesis/Data/Dataset"
split_dataset_by_category(dataset_dir, output_dir)

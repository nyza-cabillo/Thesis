import os

dataset_dir = "C:/Users/Desktop/Desktop/Thesis/Data/CacaoDataset"
for class_dir in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_dir)
    if os.path.isdir(class_path):
        print(f"{class_dir}: {len(os.listdir(class_path))} images")

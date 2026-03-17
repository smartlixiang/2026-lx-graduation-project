from dataset.dataset import BaseDataLoader

loader = BaseDataLoader(dataset="tiny-imagenet", data_path="./data", batch_size=4)
train_loader, val_loader, test_loader = loader.load()

train_ds = train_loader.dataset.dataset if hasattr(train_loader.dataset, "dataset") else train_loader.dataset
test_ds = test_loader.dataset

print("train dataset type:", type(train_ds))
print("test dataset type:", type(test_ds))

print("train classes num:", len(train_ds.classes))
print("test classes num:", len(test_ds.classes))

print("train first 10 classes:", train_ds.classes[:10])
print("test first 10 classes:", test_ds.classes[:10])

print("train class_to_idx sample:", list(train_ds.class_to_idx.items())[:5])
print("test class_to_idx sample:", list(test_ds.class_to_idx.items())[:5])

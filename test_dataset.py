from datasets.dataset_loader import load_config, verify_dataset

cfg = load_config("configs/data_config.yaml")
verify_dataset(cfg)
# ds = FoodDataset('data_config.yaml', split='train')
print(f'Classes: {ds.num_classes}')
print(f'Samples: {len(ds)}')
print(f'Sample 0: {ds[0][0].shape}')
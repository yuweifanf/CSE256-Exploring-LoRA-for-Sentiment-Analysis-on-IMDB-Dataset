from torch.utils.data import DataLoader, Dataset

class IMDBDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]["input_ids"], self.partition[index]["attention_mask"], self.partition[index]["label"]

    def __len__(self):
        return self.partition.num_rows
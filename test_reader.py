import prefetch_generator
from torch.utils import data
from tqdm import tqdm


class dataset():
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self):
        return 1


def collate_fn(batch):
    return batch


datasets = dataset()

data_loader = data.DataLoader(datasets, batch_size=2, collate_fn=collate_fn, shuffle=True, drop_last=True,
                              num_workers=2)

pbar = tqdm(enumerate(prefetch_generator.BackgroundGenerator(data_loader)), total=len(data_loader))

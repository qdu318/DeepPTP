# encoding utf-8

from torch.utils.data import Dataset, DataLoader
import ujson
import numpy as np
import torch
import utils


class MySet(Dataset):
    def __init__(self, input_file):
        self.content = open('./data/' + input_file, 'r').readlines()
        self.content = list(map(lambda x: ujson.loads(x), self.content))
        self.lengths = list(map(lambda x: len(x['X']), self.content))

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)


def collate_fn(data):
    keys = ['time_gap', 'X', 'Y', 'direction', 'distance', 'dist']

    parameters = {}
    lens = np.asarray([len(item['X']) for item in data])

    for key in keys:
        if key in ['time_gap', 'X', 'Y', 'distance']:
            seqs = np.asarray([item[key] for item in data])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.zeros(mask.shape, dtype=np.float32)
            padded[mask] = np.concatenate(seqs)
            padded = utils.Z_Score(padded)

            padded = torch.from_numpy(padded).float()
            parameters[key] = padded

        elif key == 'direction':
            parameters[key] = torch.from_numpy(np.asarray([item[key] for item in data])).type(torch.long)

        elif key == 'dist':
            x = np.asarray([item[key] for item in data])
            x = utils.Z_Score(x)
            parameters[key] = torch.from_numpy(x).type(torch.long)

    lens = lens.tolist()
    parameters['lens'] = lens

    return parameters


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = list(range(self.count))

    def __iter__(self):
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


def get_loader(file, batch_size):
    dataset = MySet(input_file=file)
    batch_sampler = BatchSampler(dataset, batch_size)
    data_loader = DataLoader(dataset=dataset, batch_size=1, collate_fn=lambda x: collate_fn(x), num_workers=0,
                             batch_sampler=batch_sampler, pin_memory=True)
    return data_loader
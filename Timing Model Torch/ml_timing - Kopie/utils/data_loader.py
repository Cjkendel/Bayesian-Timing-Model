import torch
from .dataset import Dataset
from .packed_sequence import pack_sequence

def pack_times(times):
    lengths = [v.size(0) for v in times]

def simple_collate_fn(batch):
    features, times = zip(*batch)
    return torch.stack(features), pack_sequence(times)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):

        if isinstance(dataset, str):
            dataset = Dataset(dataset)

        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                         batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=simple_collate_fn,
                         pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context)

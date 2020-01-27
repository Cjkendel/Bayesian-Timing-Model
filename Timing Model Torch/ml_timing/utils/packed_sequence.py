from collections import namedtuple
import torch

#PackedSequence_ = namedtuple('PackedSequence', ['data', 'indices'])

# type annotation for PackedSequence_ to make it compatible with TorchScript
#PackedSequence   _.__annotations__ = {'data': torch.Tensor, 'indices': torch.Tensor}


class PackedSequence:
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def pin_memory(self):
        return type(self)(self.data.pin_memory(), self.indices)

    def cuda(self, *args, **kwargs):
        """Returns a GPU copy if `self.data` not already on the GPU"""
        if self.is_cuda:
            return self
        else:
            return type(self)(self.data.cuda(*args, **kwargs), self.indices)

    def cpu(self):
        """Returns a CPU copy if `self.data` not already on the CPU"""
        if self.is_cuda:
            return type(self)(self.data.cpu(), self.indices)
        else:
            return self

    def to(self, *args, **kwargs):
        data = self.data.to(*args, **kwargs)
        if data is self.data:
            return self
        else:
            return type(self)(data, self.indices)

    def __getitem__(self, i):
        return self.data[self.indices[i]:self.indices[i+1]]

    def __len__(self):
        return len(self.indices) - 1

    def seq_len(self):
        return self.indices[-1]

    @property
    def is_cuda(self):
        r"""Returns true if `self.data` stored on a gpu"""
        return self.data.is_cuda

    def is_pinned(self):
        r"""Returns true if `self.data` stored on in pinned memory"""
        return self.data.is_pinned()


def pack_sequence(sequences):
    data = torch.cat(sequences)

    lengths = [0] + [v.size(0) for v in sequences]
    indices = torch.tensor(lengths)
    torch.cumsum(indices, 0, out=indices)

    return PackedSequence(data, indices)

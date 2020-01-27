import torch


def mean_along_packed_seq(packed_sequence, data):
    means = []
    for i in range(len(packed_sequence)):
        data_temp = data[packed_sequence.indices[i]:packed_sequence.indices[i + 1], :]
        data_temp = torch.mean(data_temp, dim=0, keepdim=True)
        means.append(data_temp)
    return torch.cat(means, dim=0)


def expand_along_packed_seq(packed_sequence, data):
    expanded = torch.empty(len(packed_sequence.data), 4)
    for i in range(len(packed_sequence)):
        expanded[packed_sequence.indices[i]:packed_sequence.indices[i + 1], :] = data[i, :]
    return expanded

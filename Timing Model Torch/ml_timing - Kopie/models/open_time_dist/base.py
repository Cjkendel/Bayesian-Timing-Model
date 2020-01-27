import torch
from utils.packed_sequence import PackedSequence


class Base:
    def seq_sum_log_prob(self, packed_seq, out=None):
        if out is None:
            out = torch.empty(len(packed_seq))

        log_prob_packed_seq = PackedSequence(self.log_prob(packed_seq.data), packed_seq.indices)

        for i in range(len(packed_seq)):
            out[i] = log_prob_packed_seq[i].sum()

        return out


from utils import DataLoader, mean_along_packed_seq, expand_along_packed_seq
from .open_time_clusters import OpenTimeClusters
from cluster_update_torch import ClusterUpdates
import torch
import time

class LatentTimingCluster:
    def __init__(self, training_dataset,
                 nn_hidden_layers=[10, 10],
                 clusters=[('exp', 1.5), ('norm', 8., 3.), ('norm', 14., 3.), ('norm', 20., 3.)],
                 batch_size=10, dropout=0.5):
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pin = bool(self.device != 'cpu')
        self.training_data_loader = DataLoader(training_dataset, batch_size=batch_size, pin_memory=self.pin)
        self.total_data_len = self.training_data_loader.dataset.total_data_len
        self.batch_number = self.total_data_len / self.batch_size
        self.clusters_list = clusters
        self.clusters = OpenTimeClusters(self.clusters_list)
        self.nn = self._build_nn(nn_hidden_layers).to(self.device)
        self.nn_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.nn_optimizer = torch.optim.SGD(self.nn.parameters(), lr=0.01)
        self.q = torch.empty((batch_size, len(self.clusters)))

    def fit(self, epochs=1000):
        self.nn.train()
        for i in range(epochs):
            losses = []
            start = time.time()
            for x, t in self.training_data_loader:
                x, t = x.to(self.device, non_blocking=True), t.to(self.device, non_blocking=True)
                self._e_step(x, t)
                loss, clusters = self._m_step(x, t)
                losses.append(loss)
            total = time.time() - start
            print(f' Epoch: {i} | Mean Loss: {sum(losses)/self.batch_number} | Time Elapsed: {total} \n '
                  f'Updated Cluster Params: {clusters} \n')
        return losses, clusters

    def _e_step(self, x, t):
        self.log_prob_nn = self.nn(x)
        with torch.no_grad():
            self.log_prob_nn_expanded = expand_along_packed_seq(t, self.log_prob_nn.detach())
            log_prob_evidence = self.clusters.log_likelihoods(t) + self.log_prob_nn_expanded
            log_sum_prob_evidence = torch.logsumexp(log_prob_evidence, dim=1, keepdim=True)
            log_prob = log_prob_evidence - log_sum_prob_evidence
            torch.exp(log_prob, out=self.q)
            self.q_hist_mean = mean_along_packed_seq(t, self.q)

    def _m_step(self, x, t):
        # update nn weights
        self.nn_optimizer.zero_grad()
        loss = self.nn_loss(self.log_prob_nn, self.q_hist_mean)
        loss.backward()
        self.nn_optimizer.step()

        with torch.no_grad():
            # update clusters
            self.clusters_list = ClusterUpdates(t,
                                                self.clusters_list,
                                                self.q,
                                                self.total_data_len,
                                                self.batch_size).update_clusters_from_list()
            self.clusters = OpenTimeClusters(self.clusters_list)

        return loss.item(), self.clusters_list

    def _build_nn(self, nn_hidden_layers):
        layers = [self.training_data_loader.dataset.static_features_number()] + nn_hidden_layers

        nn = torch.nn.Sequential()
        for i in range(1, len(layers)):
            nn.add_module("dense%d" % i, torch.nn.Linear(layers[i-1], layers[i]))
            nn.add_module("relu%d" % i, torch.nn.ReLU())
            nn.add_module("dropout%d" % i, torch.nn.Dropout(self.dropout))

        nn.add_module("dense%d" % len(layers), torch.nn.Linear(layers[-1], len(self.clusters)))

        nn.add_module("logsoftmax", torch.nn.LogSoftmax(dim=1))

        return nn

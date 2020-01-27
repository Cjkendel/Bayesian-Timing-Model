import torch


class ClusterUpdates:
    def __init__(self, t, cluster_list, q, total_data_size, batch_size):
        self.times = t.data
        self.clusters = cluster_list
        self.q = q
        self.total_data_size = total_data_size
        self.batch_size = batch_size
        self.alpha = self.batch_size / self.total_data_size

    def normal_mu_update(self, idx, convex_combo=True):
        """
        """
        q = self.q[:, idx]
        opens = self.times[:, 1]

        numerator = torch.dot(q.t(), opens)
        denominator = q.sum()
        new_param = numerator / denominator

        if convex_combo:
            convex_val = self.convex_combination_update(self.clusters[idx][1], new_param)
            return convex_val

        else:
            return new_param

    def normal_sigma_update(self, mu, idx, convex_combo=True):
        """
        """
        qq = self.q[:, idx]
        opens = self.times[:, 1]
        centred_open_sq = torch.mul((opens - mu), (opens - mu))

        numerator = torch.dot(qq.t(), centred_open_sq)
        denominator = qq.sum()
        new_param = float(numerator / denominator)
        new_param **= 0.5

        if convex_combo:
            convex_val = self.convex_combination_update(self.clusters[idx][2], new_param)
            return convex_val

        else:
            return new_param

    def exponential_rate_update(self, idx, convex_combo=True):
        """
        """
        q = self.q[:, idx]

        opens_sub_sents = self.times[:, 0]
        denominator = torch.dot(q.t(), opens_sub_sents)
        numerator = q.sum()
        new_param = float(numerator / denominator)

        if convex_combo:
            convex_val = self.convex_combination_update(self.clusters[idx][1], new_param)
            return ('exp', convex_val)

        else:
            return ('exp', new_param)

    def convex_combination_update(self, parameter_old, parameter_new):
        """Creates a convex-combination between old and new param
           with alpha = batch_size/total_data_size
        """
        convex = (1 - self.alpha) * parameter_old + self.alpha * parameter_new
        return convex

    def normal_update(self, idx):
        mu = float(self.normal_mu_update(idx))
        sigma = self.normal_sigma_update(mu, idx)
        return ('norm', mu, sigma)

    def update_cluster(self, item, idx):
        if item[0] == 'exp':
            return self.exponential_rate_update(idx)
        elif item[0] == 'norm':
            return self.normal_update(idx)

    def update_clusters_from_list(self):
        return [self.update_cluster(item, idx) for idx, item in enumerate(self.clusters)]

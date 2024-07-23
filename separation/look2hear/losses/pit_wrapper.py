###
# Author: Kai Li
# Date: 2021-06-09 09:51:16
# LastEditors: Please set LastEditors
# LastEditTime: 2024-01-24 00:16:04
###

from itertools import permutations
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


class PITLossWrapper(nn.Module):
    def __init__(
        self, loss_func, pit_from="pw_mtx", perm_reduce=None, threshold_byloss=True
    ):
        super().__init__()
        self.loss_func = loss_func
        self.pit_from = pit_from
        self.perm_reduce = perm_reduce
        self.threshold_byloss = threshold_byloss
        if self.pit_from not in ["pw_mtx", "pw_pt", "perm_avg"]:
            raise ValueError(
                "Unsupported loss function type {} for now. Expected"
                "one of [`pw_mtx`, `pw_pt`, `perm_avg`]".format(self.pit_from)
            )

    def forward(self, ests, targets, return_ests=False, reduce_kwargs=None, **kwargs):
        n_src = targets.shape[1]
        if self.pit_from == "pw_mtx":
            pw_loss = self.loss_func(ests, targets, **kwargs)
        elif self.pit_from == "pw_pt":
            pw_loss = self.get_pw_losses(self.loss_func, ests, targets, **kwargs)
        elif self.pit_from == "perm_avg":
            min_loss, batch_indices = self.best_perm_from_perm_avg_loss(
                self.loss_func, ests, targets, **kwargs
            )
            # print(batch_indices)
            mean_loss = torch.mean(min_loss)
            if not return_ests:
                return mean_loss
            reordered = self.reordered_sources(ests, batch_indices)
            return mean_loss, reordered
        else:
            return
        # import pdb; pdb.set_trace()
        assert pw_loss.ndim == 3, (
            "Something went wrong with the loss " "function, please read the docs."
        )
        assert (
            pw_loss.shape[0] == targets.shape[0]
        ), "PIT loss needs same batch dim as input"

        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
        min_loss, batch_indices = self.find_best_perm(
            pw_loss, perm_reduce=self.perm_reduce, **reduce_kwargs
        )
        if self.threshold_byloss:
            if min_loss[min_loss > -30].nelement() > 0:
                min_loss = min_loss[min_loss > -30]
        mean_loss = torch.mean(min_loss)
        if not return_ests:
            return mean_loss
        reordered = self.reordered_sources(ests, batch_indices)
        # import pdb; pdb.set_trace()
        return mean_loss, reordered

    def get_pw_losses(self, loss_func, ests, targets, **kwargs):
        B, n_src, _ = targets.shape
        pair_wise_losses = targets.new_empty(B, n_src, n_src)
        for est_idx, est_src in enumerate(ests.transpose(0, 1)):
            for target_idx, target_src in enumerate(targets.transpose(0, 1)):
                pair_wise_losses[:, est_idx, target_idx] = loss_func(
                    est_src, target_src, **kwargs
                )
        return pair_wise_losses

    def best_perm_from_perm_avg_loss(self, loss_func, ests, targets, **kwargs):
        n_src = targets.shape[1]
        perms = torch.tensor(list(permutations(range(n_src))), dtype=torch.long)
        # import pdb; pdb.set_trace()
        loss_set = torch.stack(
            [loss_func(ests[:, perm], targets) for perm in perms], dim=1
        )
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)
        batch_indices = torch.stack([perms[m] for m in min_loss_idx], dim=0)
        return min_loss, batch_indices

    def reordered_sources(self, sources, batch_indices):
        reordered_sources = torch.stack(
            [torch.index_select(s, 0, b) for s, b in zip(sources, batch_indices)]
        )
        return reordered_sources

    def find_best_perm(self, pair_wise_losses, perm_reduce=None, **kwargs):
        n_src = pair_wise_losses.shape[-1]
        if perm_reduce is not None or n_src <= 3:
            min_loss, batch_indices = self.find_best_perm_factorial(
                pair_wise_losses, perm_reduce=perm_reduce, **kwargs
            )
        else:
            min_loss, batch_indices = self.find_best_perm_hungarian(pair_wise_losses)
        return min_loss, batch_indices

    def find_best_perm_factorial(self, pair_wise_losses, perm_reduce=None, **kwargs):
        n_src = pair_wise_losses.shape[-1]
        # After transposition, dim 1 corresp. to sources and dim 2 to estimates
        pwl = pair_wise_losses.transpose(-1, -2)
        perms = pwl.new_tensor(list(permutations(range(n_src))), dtype=torch.long)
        # Column permutation indices
        idx = torch.unsqueeze(perms, 2)
        # Loss mean of each permutation
        if perm_reduce is None:
            # one-hot, [n_src!, n_src, n_src]
            # import pdb; pdb.set_trace()
            perms_one_hot = pwl.new_zeros((*perms.size(), n_src)).scatter_(2, idx, 1)
            loss_set = torch.einsum("bij,pij->bp", [pwl, perms_one_hot])
            loss_set /= n_src
        else:
            # batch = pwl.shape[0]; n_perm = idx.shape[0]
            # [batch, n_src!, n_src] : Pairwise losses for each permutation.
            pwl_set = pwl[:, torch.arange(n_src), idx.squeeze(-1)]
            # Apply reduce [batch, n_src!, n_src] --> [batch, n_src!]
            loss_set = perm_reduce(pwl_set, **kwargs)
        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)

        # Permutation indices for each batch.
        batch_indices = torch.stack([perms[m] for m in min_loss_idx], dim=0)
        return min_loss, batch_indices

    def find_best_perm_hungarian(self, pair_wise_losses: torch.Tensor):
        pwl = pair_wise_losses.transpose(-1, -2)
        # Just bring the numbers to cpu(), not the graph
        pwl_copy = pwl.detach().cpu()
        # Loop over batch + row indices are always ordered for square matrices.
        batch_indices = torch.tensor(
            [linear_sum_assignment(pwl)[1] for pwl in pwl_copy]
        ).to(pwl.device)
        min_loss = torch.gather(pwl, 2, batch_indices[..., None]).mean([-1, -2])
        return min_loss, batch_indices


if __name__ == "__main__":
    import torch
    from matrix import pairwise_neg_sisdr, pairwise_neg_sisdr

    ests = torch.randn(10, 2, 32000)
    targets = torch.randn(10, 2, 32000)

    pit_wrapper_1 = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    pit_wrapper_2 = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    print(pit_wrapper_1(ests, targets))
    print(pit_wrapper_2(ests, targets))

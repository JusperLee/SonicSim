import torch
from itertools import permutations

class SISNRi(object):
    def __init__(self):
        super(Loss, self).__init__()
    
    def sisnr(self, mix, est, ref, eps = 1e-8):
        """
        input: 
            mix: B x L
            est: B x L
        output: B
        """
        est = est - torch.mean(est, dim = -1, keepdim = True)
        ref = ref - torch.mean(ref, dim = -1, keepdim = True)
        mix = mix - torch.mean(mix, dim = -1, keepdim = True)
        est_p = (torch.sum(est * ref, dim = -1, keepdim = True) * ref) / torch.sum(ref * ref, dim = -1, keepdim = True)
        est_v = est - est_p
        mix_p = (torch.sum(mix * ref, dim = -1, keepdim = True) * ref) / torch.sum(ref * ref, dim = -1, keepdim = True)
        mix_v = mix - mix_p
        est_sisnr = 10 * torch.log10((torch.sum(est_p * est_p, dim = -1, keepdim = True) + eps) / (torch.sum(est_v * est_v, dim = -1, keepdim = True) + eps))
        mix_sisnr = 10 * torch.log10((torch.sum(mix_p * mix_p, dim = -1, keepdim = True) + eps) / (torch.sum(mix_v * mix_v, dim = -1, keepdim = True) + eps))
        return est_sisnr - mix_sisnr

    def compute_loss(self, mix, ests, refs):
        """
        input: 
            mix: B x L
            est: num_spk x B x L
        output: 1
        """

        def sisnr_loss(permute):
            # B
            return torch.mean(torch.stack([self.sisnr(mix, ests[s], refs[t]) for s, t in enumerate(permute)]), dim = 0, keepdim = True)
        num_spks = len(ests)
        # pmt_num x B
        sisnr_mat = torch.stack([sisnr_loss(p) for p in permutations(range(num_spks))])
        # B
        max_pmt, _ = torch.max(sisnr_mat, dim=0)
        return -torch.mean(max_pmt)
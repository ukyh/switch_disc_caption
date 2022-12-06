import torch
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward, get_btw_scst_reward
import numpy as np

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, sim_gts=None, gts_btw=None):
        opt = self.opt

        assert (sim_gts is None and gts_btw is None) or (sim_gts is not None and gts_btw is not None)
        if gts_btw is not None:
            ref_btw_scores = np.expand_dims(gts_btw, axis=1).repeat(gts[0].shape[1], axis=1)    # np(batch * gts_per_img, 16)
            ref_btw_scores = opt.btw_lw - opt.btw_aw * ref_btw_scores   # already normalized
        else:
            ref_btw_scores = None
        
        out = {}
        if struc_flag:
            if sim_gts is not None:
                raise NotImplementedError('Currently, CIDErBtw is not supported for StructureLosses')
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                            or not 'margin' in opt.structure_loss_type,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                            'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag:
            loss = self.crit(
                self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:],
                btw_score = torch.from_numpy(ref_btw_scores) if ref_btw_scores is not None else None
            )
        else:
            if sim_gts is not None: # CIDErBtw
                self.model.eval()
                with torch.no_grad():
                    greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                        mode='sample',
                        opt={'sample_method': opt.sc_sample_method,
                            'beam_size': opt.sc_beam_size})
                self.model.train()
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method,
                            'beam_size':opt.train_beam_size,
                            'sample_n': opt.train_sample_n},
                        mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                reward = get_btw_scst_reward(gen_result, greedy_res, gts, sim_gts, ref_btw_scores, self.opt)
                reward = torch.from_numpy(reward).to(sample_logprobs)
                loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
                out['reward'] = reward[:,0].mean()
            else:   # SCST
                self.model.eval()
                with torch.no_grad():
                    greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                        mode='sample',
                        opt={'sample_method': opt.sc_sample_method,
                            'beam_size': opt.sc_beam_size})
                self.model.train()
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method,
                            'beam_size':opt.train_beam_size,
                            'sample_n': opt.train_sample_n},
                        mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
                reward = torch.from_numpy(reward).to(sample_logprobs)
                loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
                out['reward'] = reward[:,0].mean()

        out['loss'] = loss
        return out

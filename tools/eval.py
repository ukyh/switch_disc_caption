from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
from captioning.data.dataloaderraw import *
import captioning.utils.eval_utils as eval_utils
import argparse
import captioning.utils.misc as utils
import captioning.modules.losses as losses
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
parser.add_argument('--only_lang_eval', type=int, default=0,
                help='lang eval on saved results')
parser.add_argument('--force', type=int, default=1,
                help='force to evaluate no matter if there are results available')
parser.add_argument('--device', type=str, default='cuda',
                help='cpu or cuda')
parser.add_argument('--tau_norm', type=float, default=0,
                help='tau to normalize classifier norms')
parser.add_argument('--tau_logit', type=float, default=1,
                help='tau to normalize classifier logits')
parser.add_argument('--poe_decode', type=float, default=-1,
                help='PoE decoding during evaluation with a specified poe_temp value')
parser.add_argument('--model_suffix', type=str, default='best', 
                    help='suffix of the model to evaluate')
parser.add_argument('--sim_img_json', type=str, default='data/similar_set_id/method_retrieval_05',
                    help='path to the json files containing similar image ids')
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
opt = parser.parse_args()
opt.sim_img_train = os.path.join(opt.sim_img_json, 'similar_train.json')
opt.sim_img_val = os.path.join(opt.sim_img_json, 'similar_val.json')
opt.sim_img_test = os.path.join(opt.sim_img_json, 'similar_test.json')

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = [
    'input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 
    'batch_size', 'id', 'tau_norm', 'tau_logit', 'model_suffix', 'sample_method', 
]
ignore = [
    'start_from', 'dump_images', 'dump_json', 
]

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

if opt.model_suffix != 'best':
    opt.id = opt.id + '-' + opt.model_suffix

if opt.poe_decode != -1:    # PoE decoding
    assert opt.simple_ft == False  # only for PoE models
    opt.poe_temp = opt.poe_decode
else:
    opt.simple_ft = True    # Default: Not to use PoE during evaluation
opt.focal_loss = ''         # Not to use Focal during evaluation

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
state_dict = torch.load(opt.model, map_location='cpu')
if opt.tau_norm > 0:
    print('normalizing classifier weights with tau={}'.format(opt.tau_norm))
    for k, param in state_dict.items():
        if k.startswith('model.generator') or k.startswith('logit'):
            print('size of {}: {}'.format(k, param.size()))
            if not k.endswith('bias'):
                print('before norm of {}: {}'.format(k, param.norm(dim=-1).mean()))
                tnorm = param.norm(dim=-1).pow(opt.tau_norm)
                tnorm = tnorm.unsqueeze(-1).expand_as(param)
                state_dict[k] = param / tnorm
                print('after norm of {}: {}'.format(k, state_dict[k].norm(dim=-1).mean()))
            else:
                print('before norm of {}: {}'.format(k, param.abs().mean()))
                tmp_bias = torch.zeros_like(param)
                for idx, val in enumerate(param):
                    if val.item() != 0:
                        tmp_bias[idx] = val / val.abs().pow(opt.tau_norm)
                state_dict[k] = tmp_bias
                print('after norm of {}: {}'.format(k, state_dict[k].abs().mean()))

model.load_state_dict(state_dict)
model.to(opt.device)
model.eval()
crit = losses.LanguageModelCriterion()
print("opt", opt)

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.dataset.ix_to_word = infos['vocab']


# Set sample options
opt.dataset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
        vars(opt))

print('loss: ', loss)
if lang_stats:
    print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import os
import numpy as np
import numpy.random as npr
import random
from functools import partial

import torch
import torch.utils.data as data

import multiprocessing
import six
import copy

class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """
    def __init__(self, db_path, ext, in_memory=False):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                return x['feat'] if 'feat' in x else x['z']  # normally it should be 'feat', but under cocotest_bu, the key is saved to be 'z' mistakenly.
            self.loader = load_npz
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.lmdb = lmdbdict(db_path, unsafe=True)
            self.lmdb._key_dumps = DUMPS_FUNC['ascii']
            self.lmdb._value_loads = LOADS_FUNC['identity']
        elif db_path.endswith('.pth'): # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            self.db_type = 'dir'

        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}
    
    def get(self, key):

        if self.in_memory and key in self.features:
            # We save f_input because we want to save the
            # compressed bytes to save memory
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            f_input = self.lmdb[key]
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
        else:
            f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read()

        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # load image
        feat = self.loader(f_input)

        return feat

class Dataset(data.Dataset):
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.seq_per_img = opt.seq_per_img
        
        # feature related options
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        print("#### fc:{} att:{} box:{}".format(self.use_fc, self.use_att, self.use_box))
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']
            self.vocab_size = len(self.ix_to_word)
            print('vocab size is ', self.vocab_size)
        
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        """
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
            # load in the sequence data
            seq_size = self.h5_label_file['labels'].shape
            self.label = self.h5_label_file['labels'][:]    # np(616767, 16)
            self.seq_length = seq_size[1]
            print('max sequence length in data is', self.seq_length)
            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        else:
            self.seq_length = 1

        self.data_in_memory = getattr(opt, 'data_in_memory', False)
        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy', in_memory=self.data_in_memory)
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz', in_memory=self.data_in_memory)
        self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy', in_memory=self.data_in_memory)

        self.num_images = len(self.info['images']) # self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        self.ix_to_id = {}
        self.id_to_ix = {}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if not 'split' in img:
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)
            self.ix_to_id[ix] = img['id']   # {ix:id, ...}
            self.id_to_ix[img['id']] = ix   # {id:ix, ...}
        
        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

        # preprocess similar image dict
        if os.path.isdir(opt.sim_img_json):
            sim_img_train = json.load(open(opt.sim_img_train))    # {'id':[sim_id, ...], ...}
            sim_img_val = json.load(open(opt.sim_img_val))
            sim_img_test = json.load(open(opt.sim_img_test))
            assert len(sim_img_train) == len(self.split_ix['train'])
            assert len(sim_img_val) == len(self.split_ix['val'])
            assert len(sim_img_test) == len(self.split_ix['test'])
            sim_img_ids = copy.deepcopy(sim_img_train)
            sim_img_ids.update(sim_img_val)
            sim_img_ids.update(sim_img_test)
            assert len(sim_img_ids) == len(sim_img_train) + len(sim_img_val) + len(sim_img_test)
            self.sim_img_ixs = {self.id_to_ix[int(k)]: [self.id_to_ix[v] for v in vl] for k, vl in sim_img_ids.items()}   # {ix:[sim_ix, ...], ...}
            sim_per_img = len(self.sim_img_ixs[0])
            assert len(self.sim_img_ixs) == len(sim_img_ids)
            for k, v in self.sim_img_ixs.items():
                assert len(v) == sim_per_img
            print('{} similar image ids for each image'.format(sim_per_img))
        else:
            self.sim_img_ixs = None

        # load gts_btw weights
        if os.path.exists(opt.gts_btw_json):
            gts_btw_scores = json.load(open(opt.gts_btw_json))    # {'id':[btw, ...], ...}
            self.gts_btw_scores = {int(k): v for k, v in gts_btw_scores.items()} # {id:[btw, ...], ...}
            assert len(self.gts_btw_scores) == len(self.split_ix['train'])
            for k, v in self.gts_btw_scores.items():
                assert len(v) == self.seq_per_img
            print('read CiderBtw weights for training reference captions')
        else:
            self.gts_btw_scores = None

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        assert ncap >= seq_per_img, 'Currently, dataloader_btw requires ncap >= seq_per_img'
        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            # ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            ixl = ix1   # NOTE: fix gts start for btw scores
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq
    
    def collate_func(self, batch, split):
        seq_per_img = self.seq_per_img

        fc_batch = []
        att_batch = []
        label_batch = []

        wrapped = False

        infos = []
        gts = []
        sim_gts = []
        gts_btw = []

        for sample in batch:
            # fetch image
            tmp_fc, tmp_att, tmp_seq, \
                ix, it_pos_now, tmp_wrapped = sample
            if tmp_wrapped:
                wrapped = True

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
            
            tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype = 'int')
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                tmp_label[:, 1 : self.seq_length + 1] = tmp_seq
            label_batch.append(tmp_label)

            # Used for reward evaluation
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                # gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
                ixl = self.label_start_ix[ix] - 1
                gts.append(self.label[ixl: ixl + seq_per_img])
            else:
                gts.append([])
            
            # Used for CIDErBtw reward evaluation
            if hasattr(self, 'h5_label_file') and split == 'train': # NOTE: Currently, btw is not used in eval
                sim_ixs = self.sim_img_ixs[ix]
                tmp_sim_gts = []
                for sim_ix in sim_ixs:
                    # tmp_sim_gts.append(self.label[self.label_start_ix[sim_ix] - 1: self.label_end_ix[sim_ix]])  # [np(seq_per_img, 16), ...]
                    ixl = self.label_start_ix[sim_ix] - 1
                    tmp_sim_gts.append(self.label[ixl: ixl + seq_per_img])  # [np(seq_per_img, 16), ...]
                tmp_sim_gts = np.vstack(tmp_sim_gts)    # np(sim_per_img * seq_per_img, 16)
                sim_gts.append(tmp_sim_gts)
            else:
                sim_gts.append([])
            
            if self.gts_btw_scores is not None and split == 'train':
                gts_btw.append(self.gts_btw_scores[ix]) # [[btw, ...] * batch]
            else:
                gts_btw.append([])
        
            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
            infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        fc_batch, att_batch, label_batch, gts, sim_gts, gts_btw, infos = \
            zip(*sorted(zip(fc_batch, att_batch, label_batch, gts, sim_gts, gts_btw, infos), key=lambda x: 0, reverse=True))
        data = {}
        data['fc_feats'] = np.stack(fc_batch)
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype = 'float32')
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype = 'float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch
        data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
        data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)

        data['gts'] = gts # all ground truth captions of each images
        data['sim_gts'] = sim_gts
        data['gts_btw'] = sum(gts_btw, [])  # [gts_per_img * batch]
        data['bounds'] = {'it_pos_now': it_pos_now, # the it_pos_now of the last sample
                          'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

        return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix, it_pos_now, wrapped = index #self.split_ix[index]
        if self.use_att:
            att_feat = self.att_loader.get(str(self.info['images'][ix]['id']))
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = self.box_loader.get(str(self.info['images'][ix]['id']))
                # devided by image width and height
                x1,y1,x2,y2 = np.hsplit(box_feat, 4)
                h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
                if self.norm_box_feat:
                    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
        else:
            att_feat = np.zeros((0,0), dtype='float32')
        if self.use_fc:
            try:
                fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
            except:
                # Use average of attention when there is no fc provided (For bottomup feature)
                fc_feat = att_feat.mean(0)
        else:
            fc_feat = np.zeros((0), dtype='float32')
        if hasattr(self, 'h5_label_file'):
            seq = self.get_captions(ix, self.seq_per_img)
        else:
            seq = None
        return (fc_feat,
                att_feat, seq,
                ix, it_pos_now, wrapped)

    def __len__(self):
        return len(self.info['images'])

class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.dataset = Dataset(opt)

        # Initialize loaders and iters
        self.loaders, self.iters = {}, {}
        for split in ['train', 'val', 'test']:
            if split == 'train':
                sampler = MySampler(self.dataset.split_ix[split], shuffle=True, wrap=True)
            else:
                sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)
            self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                                  batch_size=self.batch_size,
                                                  sampler=sampler,
                                                  pin_memory=True,
                                                  num_workers=2, # 4 is usually enough -> change 2 may solve the deadlock?
                                                  collate_fn=partial(self.dataset.collate_func, split=split),
                                                  drop_last=False)
            self.iters[split] = iter(self.loaders[split])

    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0
        return {split: loader.sampler.state_dict(get_prefetch_num(split)) \
                    for split, loader in self.loaders.items()}

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])


class MySampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        # if wrap, there will be not stop iteration called
        # wrap True used during training, and wrap False used during test.
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):
        wrapped = False
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        if len(self._index_list) == 0: # overflow when 0 samples
            return None
        elem = (self._index_list[self.iter_counter], self.iter_counter+1, wrapped)
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }

    
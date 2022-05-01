
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import truncate, concat_sequences
from models.siamese.siamese_network import SiameseNetwork
from models.interfaces.base_text_model import BaseTextModel
from custom_architectures.transformers_arch import get_pretrained_transformer_encoder

class TextSiamese(BaseTextModel, SiameseNetwork):
    def __init__(self,
                 lang,
                 
                 truncate   = None,
                 max_input_length   = 512,
                 use_fixed_length_input = False,
                 
                 ** kwargs
                ):
        assert truncate in (None, False, 'random', 'start', 'end')
        
        self._init_text(lang = lang, ** kwargs)
        
        self.truncate   = truncate
        self.max_input_length   = max_input_length
        self.use_fixed_length_input = use_fixed_length_input
        
        super().__init__(** kwargs)

    def build_encoder(self, embedding_dim = 512, pretrained = 'bert-base-uncased', ** kwargs):
        """ Create a simple cnn architecture with default config fitted for MNIST """
        return get_pretrained_transformer_encoder(
            pretrained, output_dim = embedding_dim, return_attention = False, ** kwargs
        )
        
    @property
    def training_hparams(self):
        return super().training_hparams(max_input_length = None, ** self.training_hparams_text)

    @property
    def encoder_input_signature(self):
        return self.text_signature
    
    def __str__(self):
        return super().__str__() + self._str_text()
    
    def get_input(self, data):
        if isinstance(data, pd.DataFrame):
            return [self.get_input(row) for idx, row in data.iterrows()]
        elif isinstance(data, list):
            return [self.get_input(data_i) for data_i in data]
        
        encoded_text = self.tf_encode_text(data)
        
        if self.truncate not in (None, False):
            encoded_text = truncate(encoded_text, self.max_input_length, keep_mode = self.truncate)
        
        return encoded_text, len(encoded_text)
    
    def filter_input(self, inp):
        return inp[1] <= self.max_input_length
    
    def augment_input(self, inp):
        tokens, length = inp
        return self.augment_text(tokens, length)
    
    def concat(self, x_same, x_not_same):
        x_same_txt, x_same_length = x_same
        x_not_same_txt, x_not_same_length = x_not_same
        
        return (
            concat_sequences(x_same_txt, x_not_same_txt, pad_value = self.blank_token_idx),
            tf.concat([x_same_length, x_not_same_length], axis = 0)
        )
    
    def get_dataset_config(self, ** kwargs):
        kwargs['padded_batch']  = True
        kwargs['pad_kwargs']    = {
            'padding_values' : (
                (((self.blank_token_idx, 0), (self.blank_token_idx, 0)), 0),
                (((self.blank_token_idx, 0), (self.blank_token_idx, 0)), 0)
            )
        }
        if self.use_fixed_length_input:
            input_shape = self.encoder_input_signature[0].shape
            kwargs['pad_kwargs'] = {
                'padded_shapes' : (
                    (((input_shape[1:], ()), (input_shape[1:], ())), ()),
                    (((input_shape[1:], ()), (input_shape[1:], ())), ()),
                )
            }
        
        return super().get_dataset_config(** kwargs)
        
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_text(),
            'truncate'  : self.truncate,
            'max_input_length'  : self.max_input_length,
            'use_fixed_length_input'    : self.use_fixed_length_input
        })
        
        return config

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import truncate, concat_sequences
from utils.text import get_encoder, random_mask
from models.siamese.siamese_network import SiameseNetwork
from custom_architectures.transformers_arch import get_pretrained_transformer_encoder

class TextSiamese(SiameseNetwork):
    def __init__(self,
                 lang,
                 
                 truncate   = None,
                 max_input_length   = 512,
                 use_fixed_length_input = False,
                 
                 text_encoder       = None,
                 
                 ** kwargs
                ):
        assert truncate in (None, False, 'random', 'start', 'end')
        
        self.lang   = lang
        self.truncate   = truncate
        self.max_input_length   = max_input_length
        self.use_fixed_length_input = use_fixed_length_input
        
        # Initialization of Text Encoder
        self.text_encoder = get_encoder(text_encoder = text_encoder, lang = lang)
        
        
        super().__init__(** kwargs)
                
        # Saving text encoder
        if not os.path.exists(self.text_encoder_file):
            self.text_encoder.save_to_file(self.text_encoder_file)
    
    def init_train_config(self,
                          max_input_length  = None,
                          nb_mask   = 1,
                          min_mask_length   = 1,
                          max_mask_length   = 1,
                          ** kwargs
                         ):
        if max_input_length: self.max_input_length   = max_input_length
        
        self.nb_mask = nb_mask
        self.min_mask_length = min_mask_length
        self.max_mask_length = max_mask_length
        
        super().init_train_config(** kwargs)

    def build_encoder(self, embedding_dim = 512, pretrained = 'bert-base-uncased', ** kwargs):
        """ Create a simple cnn architecture with default config fitted for MNIST """
        return get_pretrained_transformer_encoder(
            pretrained, output_dim = embedding_dim, return_attention = False, ** kwargs
        )
        
    @property
    def text_encoder_file(self):
        return os.path.join(self.save_dir, 'text_encoder.json')
    
    @property
    def training_hparams(self):
        return super().training_hparams(
            max_input_length = None,
            nb_mask = 1,
            min_mask_length = 1,
            max_mask_length = 1
        )

    @property
    def encoder_input_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # text (tokens)
            tf.TensorSpec(shape = (None,), dtype = tf.int32)        # text length
        )
    
    @property
    def vocab(self):
        return self.text_encoder.vocab

    @property
    def vocab_size(self):
        return self.text_encoder.vocab_size
                
    @property
    def blank_token_idx(self):
        return self.text_encoder.blank_token_idx

    @property
    def mask_token_idx(self):
        return self.text_encoder[self.text_encoder.mask_token]
    
    def __str__(self):
        des = super().__str__()
        des += "Input language : {}\n".format(self.lang)
        des += "Input vocab (size = {}) : {}\n".format(self.vocab_size, self.vocab[:50])
        return des
    
    def encode_text(self, text):
        return self.text_encoder.encode(text)
    
    def decode_text(self, encoded):
        if isinstance(encoded, tf.Tensor): encoded = encoded.numpy()
        return self.text_encoder.decode(encoded)
        
    def get_input(self, data):
        if isinstance(data, pd.DataFrame):
            return [self.get_input(row) for idx, row in data.iterrows()]
        elif isinstance(data, list):
            return [self.get_input(data_i) for data_i in data]
        
        encoded_text = tf.py_function(self.encode_text, [data['text']], Tout = tf.int32)
        encoded_text.set_shape([None])
        
        if self.truncate not in (None, False):
            encoded_text = truncate(encoded_text, self.max_input_length, keep_mode = self.truncate)
        
        return encoded_text, len(encoded_text)
    
    def filter_input(self, inp):
        return inp[1] <= self.max_input_length
    
    def augment_input(self, inp):
        tokens, length = inp
        tokens = tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: random_mask(
                tokens, self.mask_token_idx, min_idx = 1, max_idx = len(inp) - 1, nb_mask = self.nb_mask,
                min_mask_length = self.min_mask_length, max_mask_length = self.max_mask_length
            ),
            lambda: tokens
        )
        return tokens, len(tokens)
    
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
        config['lang']      = self.lang
        
        config['truncate']  = self.truncate
        config['max_input_length']  = self.max_input_length
        config['use_fixed_length_input']    = self.use_fixed_length_input
        
        config['text_encoder']      = self.text_encoder_file
        
        return config

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.text import TextEncoder
from models.siamese.siamese_network import SiameseNetwork
from custom_architectures.transformers_arch.bert_arch import BertEmbedding

class TextSiamese(SiameseNetwork):
    def __init__(self,
                 lang,
                 
                 max_input_length   = 512,
                 use_fixed_length_input = False,
                 
                 text_encoder       = None,
                 
                 ** kwargs
                ):
        self.lang   = lang
        self.max_input_length   = max_input_length
        self.use_fixed_length_input = use_fixed_length_input
        
        # Initialization of Text Encoder
        if text_encoder is None: text_encoder = {}
        if isinstance(text_encoder, dict):
            text_encoder['use_sos_and_eos'] = False
            if 'vocab' not in text_encoder:
                text_encoder['vocab'] = get_symbols(lang, arpabet = False)
                text_encoder['level'] = 'char'
            else:
                text_encoder.setdefault('level', 'char')
            text_encoder.setdefault('cleaners', ['french_cleaners'] if lang == 'fr' else ['english_cleaners'])
            self.text_encoder = TextEncoder(** text_encoder)
        
        elif isinstance(text_encoder, str):
            self.text_encoder = TextEncoder.load_from_file(text_encoder)
        elif isinstance(text_encoder, TextEncoder):
            self.text_encoder = text_encoder
        else:
            raise ValueError("input encoder de type inconnu : {}".format(text_encoder))
        
        
        super().__init__(**kwargs)
                
        # Saving text encoder
        if not os.path.exists(self.text_encoder_file):
            self.text_encoder.save_to_file(self.text_encoder_file)
        
        if hasattr(self.siamese, '_build'): self.siamese._build()
    
    def build_encoder(self, embedding_dim = 512, pretrained = 'bert-base-uncased', ** kwargs):
        """ Create a simple cnn architecture with default config fitted for MNIST """
        return BertEmbedding.from_pretrained(
            pretrained, output_dim = embedding_dim, return_attention = False, ** kwargs
        )
        
    @property
    def text_encoder_file(self):
        return os.path.join(self.save_dir, 'text_encoder.json')
    
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
    def go_frame(self):
        return tf.zeros((1, self.n_mel_channels), dtype = tf.float32)
    
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
        
        if tf.shape(encoded_text)[0] > self.max_input_length:
            start = tf.random.uniform(
                (), minval = 0, 
                maxval = tf.shape(encoded_text)[0] - self.max_input_length,
                dtype = tf.int32
            )
            encoded_text = encoded_text[start : start + self.max_input_length]
        
        return encoded_text, len(encoded_text)
    
    def concat(self, x_same, x_not_same):
        x_same_txt, x_same_length = x_same
        x_not_same_txt, x_not_same_length = x_not_same
        
        seq_1, seq_2 = tf.shape(x_same_txt)[1], tf.shape(x_not_same_txt)[1]
        
        if seq_1 != seq_2:
            padding = [(0,0), (0, tf.abs(seq_1 - seq_2))]
            if seq_1 > seq_2:
                x_not_same_txt = tf.pad(x_not_same_txt, padding, constant_values = self.blank_token_idx)
            else:
                x_same_txt = tf.pad(x_same_txt, padding, constant_values = self.blank_token_idx)
                
        return (
            tf.concat([x_same_txt, x_not_same_txt], axis = 0),
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
        
        config['max_input_length']  = self.max_input_length
        config['use_fixed_length_input']    = self.use_fixed_length_input
        
        config['text_encoder']      = self.text_encoder_file
        
        return config

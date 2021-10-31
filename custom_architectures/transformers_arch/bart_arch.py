# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TF 2.0 BART model. """

import tensorflow as tf

from hparams.hparams import HParams
from custom_layers import FasterEmbedding, get_activation
from custom_architectures.transformers_arch.transformer_arch import *
from custom_architectures.transformers_arch.embedding_head import EmbeddingHead, HParamsEmbeddingHead

__base_bart_config = {
    'vocab_size'    : -1,
    'embedding_dim' : -1,
    'max_input_length'  : 1024,
    'scale_embedding'   : False,
    'epsilon'   : 1e-5,
    'positional_offset' : 2
}
_shared_keys   = list(__base_bart_config.keys()) + ['return_attention', 'return_states', 'return_mask']

HParamsBartEncoder  = HParamsTransformerEncoder(
    ** __base_bart_config,
    subsampling_step    = -1,
    subsampling_offset  = 1,
    subsampling_mode    = 'select',
    subsample_after     = True
)
HParamsBartEmbedding    = HParamsBartEncoder(** HParamsEmbeddingHead)

HParamsBartDecoder  = HParamsTransformerDecoder(
    ** __base_bart_config,
    final_activation    = 'softmax',
    sos_token   = None,
    eos_token   = None
)

HParamsBart         = HParamsTransformer(
    ** HParamsBartEncoder.get_config(add_prefix = 'encoder'),
    ** HParamsBartDecoder.get_config(add_prefix = 'decoder'),
    ** __base_bart_config,
    sos_token   = None,
    eos_token   = None
)

class BartEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, token_embedding = None, name = None, ** kwargs):
        super().__init__(name = name)

        self.hparams = HParamsBartEncoder.extract(kwargs)
        self.hparams = self.hparams(vocab_size = vocab_size, embedding_dim = embedding_dim)
        
        
        self.embedding_factor = tf.math.sqrt(float(self.embedding_dim)) if self.hparams.scale_embedding else 1.
        
        if token_embedding is None:
            token_embedding = FasterEmbedding(self.vocab_size, self.embedding_dim, name = 'token_embedding')
        
        self.token_embedding_layer = token_embedding
        self.pos_embedding_layer   = FasterEmbedding(
            self.max_input_length, self.embedding_dim, name = "pos_embeddings"
        )
        self.encoder    = TransformerEncoder(** self.hparams, name = "encoder")
        
        self.norm       = tf.keras.layers.LayerNormalization(epsilon = self.hparams.epsilon)
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate)
        
        self.subsampling_layer  = None
        if self.hparams.subsampling_step > 1:
            assert self.hparams.subsampling_mode in ('select', 'conv', 'min' 'max', 'mean'), "Unknown subsampling mode : {}".format(self.hparams.subsampling_mode)
            if self.hparams.subsampling_mode == 'conv':
                self.subsampling_layer = tf.keras.layers.Conv1D(
                    filters = self.embedding_dim, kernel_size = self.hparams.subsampling_step,
                    strides = self.hparams.subsampling_step, padding = 'valid', name = 'subsampling_layer'
                )
    
    @property
    def encoder_layers(self):
        return self.encoder.encoder_layers
    
    @property
    def embedding_dim(self):
        return self.hparams.embedding_dim
    
    @property
    def vocab_size(self):
        return self.hparams.vocab_size
    
    @property
    def max_input_length(self):
        return self.hparams.max_input_length + self.hparams.positional_offset
    
    def _build(self):
        batch_size, seq_len = 2, 32
        text = tf.ones([batch_size, seq_len], dtype = tf.int32)
        text_length = tf.fill([batch_size, 1], seq_len)
        
        self([text, text_length], training = False)
        
    def freeze(self, trainable = False):
        self.token_embedding_layer.trainable    = trainable
        self.pos_embedding_layer.trainable      = trainable
        self.encoder.trainable  = trainable 
        self.norm.trainable     = trainable 

    def embed_tokens(self, text, training = False, positional_offset = -1):
        if positional_offset == -1: positional_offset = self.hparams.positional_offset
        seq_len = tf.shape(text)[1]
        
        pos_ids = tf.expand_dims(tf.range(seq_len) + positional_offset, axis = 0)
        
        if len(tf.shape(text)) == 3:
            token_embedded = text * self.embedding_factor
        else:
            token_embedded = self.token_embedding_layer(text) * self.embedding_factor
        pos_embedded   = self.pos_embedding_layer(pos_ids)
        
        embedded = self.norm(token_embedded + pos_embedded)
        embedded = self.dropout(embedded, training = training)
        
        return embedded
    
    def subsample(self, output, mask = None, training = False):
        if self.hparams.subsampling_step <= 1: return output, mask
        
        if self.hparams.subsampling_mode == 'select':
            indices = tf.range(self.hparams.subsampling_offset, tf.shape(output)[1], self.hparams.subsampling_step)
            indices = tf.tile(tf.expand_dims(indices, axis = 0), [tf.shape(output)[0], 1])

            output = tf.gather(output, indices, batch_dims = 1)

            if mask is not None:
                mask = tf.gather(tf.squeeze(mask, [1, 2]), indices, batch_dims = 1)
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1])
        elif self.hparams.subsampling_mode == 'conv':
            output = self.subsampling_layer(output, training = training)
            

            if mask is not None:
                indices = tf.range(0, tf.shape(output)[1]) * self.hparams.subsampling_step
                indices = tf.tile(tf.expand_dims(indices, axis = 0), [tf.shape(output)[0], 1])

                mask = tf.gather(tf.squeeze(mask, [1, 2]), indices, batch_dims = 1)
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1])
        else:
            rest = tf.shape(output)[1] % self.hparams.subsampling_step
            if rest != 0:
                output = tf.pad(output, [(0, 0), (0, self.hparams.subsampling_step - rest), (0, 0)])
                
                if mask is not None:
                    mask = tf.pad(
                        mask, [(0, 0), (0, 0), (0, 0), (0, self.hparams.subsampling_step - rest)],
                        constant_values = 1
                    )
            
            output = tf.reshape(
                output, [tf.shape(output)[0], -1, self.hparams.subsampling_step, tf.shape(output)[-1]]
            )
            
            if mask is not None:
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1, self.hparams.subsampling_step])
                mask = tf.reduce_min(mask, axis = -1)
            
            if self.hparams.subsampling_mode == 'min':
                output = tf.reduce_min(output, axis = 2)
            elif self.hparams.subsampling_mode == 'max':
                output = tf.reduce_max(output, axis = 2)
            else:
                output = tf.reduce_mean(output, axis = 2)
        
        return output, mask

    def call(self, inputs, mask = None, training = False, positional_offset = -1, force_not_subsampling = False,
             return_attention = None, return_mask = None, return_states = None, ** kwargs):
        text, text_lengths = inputs
        batch_size, seq_len = tf.shape(text)[0], tf.shape(text)[1]
        
        if mask is None:
            mask = create_padding_mask(text, seq_len = text_lengths)
        
        embedded = self.embed_tokens(text, training = training, positional_offset = positional_offset)
        
        if not self.hparams.subsample_after and not force_not_subsampling:
            encoder_outputs, mask = self.subsample(encoder_outputs, mask = mask, training = training)
        
        encoder_outputs, attn_weights, states, mask = self.encoder(
            embedded, seq_length = text_lengths, mask = mask, training = training,
            return_attention = True, return_states = True, return_mask = True
        )
        
        if self.hparams.subsample_after and not force_not_subsampling:
            encoder_outputs, mask = self.subsample(encoder_outputs, mask = mask, training = training)
        
        return self.encoder.format_output(
            encoder_outputs, attn_weights = attn_weights, states = states, mask = mask,
            return_attention = return_attention, return_states = return_states, return_mask = return_mask
        )

    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'facebook/bart-large',
                        pretrained_task = 'generation',
                        pretrained      = None,
                        ** kwargs
                       ):
        from models.weights_converter import partial_transfer_learning
        
        if pretrained is None:
            with tf.device('cpu') as d:
                pretrained = transformers_bart(pretrained_name, pretrained_task)

        config = HParamsBartEncoder(
            vocab_size      = pretrained.config.vocab_size,
            embedding_dim   = pretrained.config.d_model,
            max_input_length    = pretrained.config.max_position_embeddings,

            num_layers  = pretrained.config.encoder_layers,
            ffn_dim     = pretrained.config.encoder_ffn_dim,
            ffn_activation  = pretrained.config.activation_function,
            mha_num_heads   = pretrained.config.encoder_attention_heads,
            mha_epsilon     = 1e-5,
            epsilon     = 1e-5,
        )
        
        instance = cls(** config(** kwargs))
        instance._build()
        
        offset, n_enc_layer_weights = 1, 16
        
        weights = pretrained.model.encoder.get_weights()
        # Invert `key` and `value` weights for each MHA layer
        for i in range(pretrained.config.encoder_layers):
            weights[i * n_enc_layer_weights + offset], weights[i * n_enc_layer_weights + offset + 2] = (
                weights[i * n_enc_layer_weights + offset + 2], weights[i * n_enc_layer_weights + offset]
            )
            weights[i * n_enc_layer_weights + offset + 1], weights[i * n_enc_layer_weights + offset + 3] = (
                weights[i * n_enc_layer_weights + offset + 3], weights[i * n_enc_layer_weights + offset + 1]
            )
        # Add shared embeddings weights to the list
        weights = [pretrained.get_weights()[0]] + weights
        
        partial_transfer_learning(instance, weights)
        
        return instance

class BartEmbedding(BartEncoder):
    def __init__(self, output_dim, vocab_size, embedding_dim, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
        
        self.hparams = HParamsBartEmbedding.extract(kwargs)
        self.hparams = self.hparams(
            vocab_size = vocab_size, output_dim = output_dim, embedding_dim = embedding_dim
        )
        
        self.embedding_head = EmbeddingHead(** self.hparams)

    def call(self, inputs, mask = None, training = False,
             return_attention = None, return_mask = False, ** kwargs):
        hidden, attn, mask = super().call(
            inputs, mask = mask, training = training, return_attention = True, return_mask = True, ** kwargs
        )
        
        output = self.embedding_head(hidden, mask = mask, training = training)

        return format_output(
            self.hparams, output, attn, mask = mask,
            return_attention = return_attention, return_mask = return_mask
        )

class BartDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, token_embedding = None, name = None, ** kwargs):
        super().__init__(name = name)

        self.hparams = HParamsBartDecoder.extract(kwargs)
        self.hparams = self.hparams(vocab_size = vocab_size, embedding_dim = embedding_dim)
        
        self.embedding_factor = tf.math.sqrt(float(embedding_dim)) if self.hparams.scale_embedding else 1.
        
        if token_embedding is None:
            token_embedding = FasterEmbedding(self.vocab_size, self.embedding_dim, name = 'token_embedding')
        
        self.token_embedding_layer = token_embedding
        self.pos_embedding_layer   = FasterEmbedding(
            self.max_input_length, self.embedding_dim, name = "pos_embeddings"
        )
        self.decoder    = TransformerDecoder(** self.hparams, name = "decoder")
        
        self.norm       = tf.keras.layers.LayerNormalization(epsilon = self.hparams.epsilon)
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate)

        self.final_bias = self.add_weight(
            shape = [1, vocab_size], name = "final_bias", trainable = False, initializer = "zeros"
        )
        self.final_act_layer    = get_activation(self.hparams.final_activation)
        
    @property
    def decoder_layers(self):
        return self.decoder.decoder_layers
    
    @property
    def embedding_dim(self):
        return self.hparams.embedding_dim
    
    @property
    def vocab_size(self):
        return self.hparams.vocab_size
    
    @property
    def max_input_length(self):
        return self.hparams.max_input_length + self.hparams.positional_offset

    def _build(self):
        batch_size, in_seq_len, out_seq_len = 2, 8, 16
        
        encoder_out = tf.random.normal((batch_size, in_seq_len, self.embedding_dim))
        text    = tf.ones([batch_size, out_seq_len], dtype = tf.int32)
        
        self([encoder_out, text], training = False)
        
    def freeze(self, trainable = False):
        self.token_embedding_layer.trainable    = trainable
        self.pos_embedding_layer.trainable      = trainable
        self.decoder.trainable  = trainable 
        self.norm.trainable     = trainable 
        
    def set_tokens(self, sos_token, eos_token):
        self.hparams = self.hparams(sos_token = sos_token, eos_token = eos_token)

    def embed_tokens(self, text, training = False, positional_offset = -1):
        if positional_offset == -1: positional_offset = self.hparams.positional_offset
        seq_len = tf.shape(text)[1]
        
        pos_ids = tf.expand_dims(tf.range(seq_len) + positional_offset, axis = 0)
        
        if len(tf.shape(text)) == 3:
            token_embedded = text * self.embedding_factor
        else:
            token_embedded = self.token_embedding_layer(text) * self.embedding_factor
        pos_embedded   = self.pos_embedding_layer(pos_ids)
        
        embedded = self.norm(token_embedded + pos_embedded)
        embedded = self.dropout(embedded, training = training)
        
        return embedded

    def call(self,
             inputs,
             mask   = None,
             training   = False,
             enc_padding_mask   = None,
             dec_padding_mask   = None,
             look_ahead_mask    = None,
             positional_offset  = -1,
             
             return_attention   = None,
             return_states      = None,
             return_mask        = None,
             return_logits      = None,
             
             ** kwargs
            ):
        encoder_out, text = inputs[:2]
        dec_text_length = inputs[2] if len(inputs) == 3 else None
        
        batch_size  = tf.shape(text)[0]
        out_seq_len = tf.shape(text)[1]
        
        
        embedded = self.embed_tokens(text, training = training, positional_offset = positional_offset)

        decoder_outputs, attn_weights, states, mask = self.decoder(
            [encoder_out, embedded],
            mask    = mask,
            training    = training,
            dec_seq_length  = dec_text_length,
            enc_padding_mask    = enc_padding_mask,
            dec_padding_mask    = dec_padding_mask,
            look_ahead_mask     = look_ahead_mask,
            return_attention    = True,
            return_states       = True,
            return_mask         = True,
            return_logits       = False
        )

        logits = tf.reshape(decoder_outputs, [-1, self.embedding_dim])
        logits = tf.matmul(logits, self.token_embedding_layer.embeddings, transpose_b = True)
        logits = tf.reshape(logits, [batch_size, out_seq_len, self.vocab_size])
        
        logits = logits + self.final_bias
        
        output = self.final_act_layer(logits) if self.final_act_layer is not None else logits
        
        return self.decoder.format_output(
            output, logits = logits, attn_weights = attn_weights, states = states, mask = mask,
            return_attention = return_attention, return_states = return_states, return_mask = return_mask,
            return_logits = return_logits
        )

    def infer(self,
              encoder_output,
              enc_padding_mask  = None,
              training  = False,

              max_length    = None,
              early_stopping    = True,
              
              return_attention   = None,
              return_states      = None,
              return_mask        = None,
              return_logits      = None,
             
              ** kwargs
             ):
        assert self.hparams.sos_token is not None and self.hparams.eos_token is not None
        if max_length is None: max_length = self.hparams.max_input_length

        batch_size  = tf.shape(encoder_output)[0]
        
        tokens      = tf.fill((batch_size, 1), self.hparams.sos_token)
        output      = tf.zeros((batch_size, 1, self.vocab_size), dtype = tf.float32)
        logits      = output
        finished    = tf.zeros((batch_size,), dtype = tf.int32)
        
        while tf.shape(tokens)[1] < max_length:
            output, logits, attn_weights, states, mask = self(
                [encoder_output, tokens],
                training    = training,
                enc_padding_mask    = enc_padding_mask,
                return_attention    = True,
                return_logits       = True,
                return_states       = True,
                return_mask         = True,
                ** kwargs
            )
            
            next_token = tf.argmax(output[:, -1], axis = -1, output_type = tokens.dtype)
            
            tokens      = tf.concat([tokens, tf.reshape(next_token, [batch_size, 1])], axis = -1)
            finished    = tf.maximum(
                finished, tf.cast(tf.math.equal(next_token, self.hparams.eos_token), tf.int32)
            )
            if early_stopping and tf.reduce_sum(finished) == batch_size:
                break
        
        return self.decoder.format_output(
            output, logits = logits, attn_weights = attn_weights, states = states, mask = mask,
            return_attention = return_attention, return_states = return_states, return_mask = return_mask,
            return_logits = return_logits
        )
        
    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'facebook/bart-large',
                        pretrained_task = 'generation',
                        pretrained      = None,
                        ** kwargs
                       ):
        from models.weights_converter import partial_transfer_learning
        
        if pretrained is None:
            with tf.device('cpu') as d:
                pretrained = transformers_bart(pretrained_name, pretrained_task)

        config = HParamsBartDecoder(
            vocab_size      = pretrained.config.vocab_size,
            embedding_dim   = pretrained.config.d_model,
            max_input_length    = pretrained.config.max_position_embeddings,
            sos_token   = 0,
            eos_token   = 2,

            num_layers  = pretrained.config.decoder_layers,
            ffn_dim     = pretrained.config.decoder_ffn_dim,
            ffn_activation  = pretrained.config.activation_function,
            mha_num_heads   = pretrained.config.decoder_attention_heads,
            mha_epsilon     = 1e-5,
            enc_mha_num_heads   = pretrained.config.decoder_attention_heads,
            enc_mha_epsilon     = 1e-5,
            epsilon     = 1e-5
        )
        
        instance = cls(** config(** kwargs))
        instance._build()
        
        offset, n_enc_layer_weights = 2, 16
        
        weights = pretrained.model.decoder.get_weights()
        # Invert `key` and `value` weights for each MHA layer
        offset, n_mha_weights, n_dec_layer_weights = 1, 10, 26
        for i in range(pretrained.config.decoder_layers):
            weights[i * n_dec_layer_weights + offset], weights[i * n_dec_layer_weights + offset + 2] = (
                weights[i * n_dec_layer_weights + offset + 2], weights[i * n_dec_layer_weights + offset]
            )
            weights[i * n_dec_layer_weights + offset + 1], weights[i * n_dec_layer_weights + offset + 3] = (
                weights[i * n_dec_layer_weights + offset + 3], weights[i * n_dec_layer_weights + offset + 1]
            )
            
            weights[i * n_dec_layer_weights + n_mha_weights + offset], weights[i * n_dec_layer_weights + n_mha_weights + offset + 2] = (
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 2],
                weights[i * n_dec_layer_weights + n_mha_weights + offset]
            )
            weights[i * n_dec_layer_weights + n_mha_weights + offset + 1], weights[i * n_dec_layer_weights + n_mha_weights + offset + 3] = (
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 3],
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 1]
            )
        # Add shared embeddings weights to the list
        weights = [pretrained.get_weights()[0]] + weights + [pretrained.get_weights()[-1]]
        
        partial_transfer_learning(instance, weights)
        
        return instance

class Bart(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, max_input_length,
                 sos_token = None, eos_token = None, name = None, ** kwargs):
        super().__init__(name = name)
        
        tokens = {'sos_token' : sos_token, 'eos_token' : eos_token}
        if sos_token is not None:
            tokens.update({'decoder_sos_token' : sos_token, 'decoder_eos_token' : eos_token})
        
        kwargs.update({
            'embedding_dim' : embedding_dim, 'vocab_size' : vocab_size, 'max_input_length' : max_input_length
        })
        self.hparams = HParamsBart.extract(kwargs)
        self.hparams = self.hparams(
            ** {'encoder_{}'.format(k) : self.hparams[k] for k in _shared_keys},
            ** {'decoder_{}'.format(k) : self.hparams[k] for k in _shared_keys},
            ** tokens
        )
        
        
        self.shared_embedding = FasterEmbedding(vocab_size, embedding_dim, name = "token_embedding")
        
        self.encoder    = BartEncoder(
            token_embedding = self.shared_embedding,
            name = "encoder",
            ** self.hparams.get_config(prefix = 'encoder')
        )
        self.decoder    = BartDecoder(
            token_embedding = self.shared_embedding,
            name = "decoder",
            ** self.hparams.get_config(prefix = 'decoder')
        )

    @property
    def embedding_dim(self):
        return self.hparams.embedding_dim
    
    @property
    def vocab_size(self):
        return self.hparams.vocab_size
    
    @property
    def max_input_length(self):
        return self.hparams.max_input_length + self.hparams.positional_offset

    def _build(self):
        batch_size, in_seq_len, out_seq_len = 2, 16, 32
        text_in = tf.ones([batch_size, in_seq_len], dtype = tf.int32)
        text_in_length = tf.fill([batch_size, 1], in_seq_len)
        text_out = tf.ones([batch_size, out_seq_len], dtype = tf.int32)
        text_out_length = tf.fill([batch_size, 1], out_seq_len)
        
        self([text_in, text_in_length, text_out, text_out_length], training = False)
        
    def freeze(self, trainable = False):
        self.shared_embedding.trainable = trainable 
        self.encoder.freeze(trainable = trainable )
        self.decoder.freeze(trainable = trainable)

    def set_tokens(self, sos_token, eos_token):
        self.hparams = self.hparams(sos_token = sos_token, eos_token = eos_token)
        self.decoder.set_tokens(sos_token, eos_token)
    
    def call(self,
             inputs,
             training   = False,
             encoder_mask   = None,
             decoder_mask   = None,
             return_attention   = None,
             positional_offset  = -1,
             ** kwargs
            ):
        text_in, text_in_length, text_out, text_out_length = inputs
        
        if encoder_mask is None:
            encoder_mask = create_padding_mask(text_in, seq_len = text_in_length)
        
        encoder_out, encoder_attn_weights, encoder_mask = self.encoder(
            [text_in, text_in_length], mask = encoder_mask, training = training,
            positional_offset = positional_offset, return_attention = True, return_mask = True
        )
        
        decoder_outputs, decoder_attn_weights = self.decoder(
            [encoder_out, text_out, text_out_length],
            mask    = decoder_mask,
            training    = training,
            enc_padding_mask    = encoder_mask,
            return_attention    = True
        )
        
        return format_output(
            self.hparams, decoder_outputs, decoder_attn_weights, return_attention = return_attention
        )

    def infer(self, inputs, training = False, encoder_mask = None, ** kwargs):
        kwargs.setdefault('return_attention', self.hparams.return_attention)
        
        text_in, text_in_length = inputs
        
        if encoder_mask is None:
            encoder_mask = create_padding_mask(text_in, seq_len = text_in_length)
        
        encoder_out, encoder_attn_weights, encoder_mask = self.encoder(
            [text_in, text_in_length], mask = encoder_mask, training = training,
            return_attention = True, return_mask = True
        )

        return self.decoder.infer(
            encoder_out, training = training, enc_padding_mask = encoder_mask, ** kwargs
        )
    
    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'facebook/bart-large',
                        pretrained_task = 'generation', 
                        ** kwargs
                       ):
        from models.weights_converter import partial_transfer_learning
        
        with tf.device('cpu') as d:
            pretrained = transformers_bart(pretrained_name, pretrained_task)

        config = HParamsBart(
            vocab_size      = pretrained.config.vocab_size,
            embedding_dim   = pretrained.config.d_model,
            max_input_length    = pretrained.config.max_position_embeddings,
            sos_token   = 0,
            eos_token   = 2,

            encoder_num_layers  = pretrained.config.encoder_layers,
            encoder_ffn_dim     = pretrained.config.encoder_ffn_dim,
            encoder_ffn_activation  = pretrained.config.activation_function,
            encoder_mha_num_heads   = pretrained.config.encoder_attention_heads,
            encoder_mha_epsilon     = 1e-5,
            encoder_epsilon     = 1e-5,

            decoder_num_layers  = pretrained.config.decoder_layers,
            decoder_ffn_dim     = pretrained.config.decoder_ffn_dim,
            decoder_ffn_activation  = pretrained.config.activation_function,
            decoder_mha_num_heads   = pretrained.config.decoder_attention_heads,
            decoder_mha_epsilon     = 1e-5,
            decoder_enc_mha_num_heads   = pretrained.config.decoder_attention_heads,
            decoder_enc_mha_epsilon     = 1e-5,
            decoder_epsilon     = 1e-5
        )
        
        instance = cls(** config(** kwargs))
        instance._build()
        
        offset, n_enc_layer_weights = 2, 16
        
        weights = pretrained.get_weights()
        # Invert `key` and `value` weights for each MHA layer
        for i in range(pretrained.config.encoder_layers):
            weights[i * n_enc_layer_weights + offset], weights[i * n_enc_layer_weights + offset + 2] = (
                weights[i * n_enc_layer_weights + offset + 2], weights[i * n_enc_layer_weights + offset]
            )
            weights[i * n_enc_layer_weights + offset + 1], weights[i * n_enc_layer_weights + offset + 3] = (
                weights[i * n_enc_layer_weights + offset + 3], weights[i * n_enc_layer_weights + offset + 1]
            )


        offset = n_enc_layer_weights * pretrained.config.encoder_layers + offset + 3 # + 3
        n_mha_weights, n_dec_layer_weights = 10, 26
        for i in range(pretrained.config.decoder_layers):
            weights[i * n_dec_layer_weights + offset], weights[i * n_dec_layer_weights + offset + 2] = (
                weights[i * n_dec_layer_weights + offset + 2], weights[i * n_dec_layer_weights + offset]
            )
            weights[i * n_dec_layer_weights + offset + 1], weights[i * n_dec_layer_weights + offset + 3] = (
                weights[i * n_dec_layer_weights + offset + 3], weights[i * n_dec_layer_weights + offset + 1]
            )
            
            weights[i * n_dec_layer_weights + n_mha_weights + offset], weights[i * n_dec_layer_weights + n_mha_weights + offset + 2] = (
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 2],
                weights[i * n_dec_layer_weights + n_mha_weights + offset]
            )
            weights[i * n_dec_layer_weights + n_mha_weights + offset + 1], weights[i * n_dec_layer_weights + n_mha_weights + offset + 3] = (
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 3],
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 1]
            )
        
        partial_transfer_learning(instance, weights)
        
        return instance

def transformers_bart(name = 'facebook/bart-large', task = 'generation'):
    import transformers
    if task == 'generation':
        return transformers.TFBartForConditionalGeneration.from_pretrained(name)
    else:
        raise ValueError("Unknown task !\n  Accepted : {}\n  Got : {}".format(
            tuple(_transformers_pretrained_task.keys()), task
        ))

custom_functions    = {
    'transformers_bart' : transformers_bart,
    
    'BartEncoder'   : BartEncoder,
    'BartEmbedding' : BartEmbedding,
    'BartDecoder'   : BartDecoder,
    'Bart'          : Bart
}

custom_objects  = {
    'TransformerEncoder'    : TransformerEncoder,
    
    'BartEncoder'   : BartEncoder,
    'BartEmbedding' : BartEmbedding,
    'BartDecoder'   : BartDecoder,
    'Bart'          : Bart
}

_encoders   = {'Bart' : BartEmbedding}
_decoders   = {'Bart' : BartDecoder}

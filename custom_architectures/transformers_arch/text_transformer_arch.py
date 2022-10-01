
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

import tensorflow as tf

from loggers import timer
from hparams import HParams
from custom_layers import FasterEmbedding
from custom_architectures.transformers_arch.transformer_arch import *
from custom_architectures.transformers_arch.generation_utils import infer

HParamsTransformerTokenEmbedding = HParams(
    vocab_size  = None,
    embedding_dim   = None,
    max_input_length    = None,
    scale_embedding     = False,
    repeat_position     = -1,
    positional_offset   = 0,
    max_token_types     = 0,
    normalize_embeddings    = True,
    norm_training   = True,
    epsilon     = 1e-6,
    drop_rate   = 0.1
)

_shared_config = [
    'vocab_size', 'max_input_length', 'scale_embedding', 'positional_offset'
]

HParamsTextTransformerBlock = HParamsTransformerBlock(
    ** HParamsTransformerTokenEmbedding,
    sos_token   = None,
    eos_token   = None
)
HParamsTextTransformerEncoder = HParamsTextTransformerBlock(** HParamsTransformerEncoder)
HParamsTextTransformerDecoder = HParamsTextTransformerBlock(
    ** HParamsTransformerDecoder, ignore_kwargs = True
)

HParamsTextTransformer  = HParamsTransformer(
    ** HParamsTextTransformerEncoder.get_config(add_prefix = 'encoder'),
    ** HParamsTextTransformerDecoder.get_config(add_prefix = 'decoder'),
    ** {key : None for key in _shared_config},
    sos_token   = None,
    eos_token   = None
)

class TransformerTokenEmbedding(tf.keras.layers.Layer):
    _attr_to_set = [
        'embedding_dim', 'vocab_size', 'norm_training', 'positional_offset', 'repeat_position'
    ]
    
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 max_input_length,
                 
                 token_embedding    = None,
                 positional_embedding   = None,
                 
                 name = 'embeddings',
                 ** kwargs
                ):
        super().__init__(name = name)
        
        self.hparams = HParamsTransformerTokenEmbedding.extract(kwargs)
        self.hparams = self.hparams(
            vocab_size      = vocab_size,
            embedding_dim   = embedding_dim,
            max_input_length    = max_input_length
        )
        
        for key in self._attr_to_set:
            setattr(self, key, self.hparams[key])
        
        self.embedding_factor = tf.math.sqrt(
            float(embedding_dim) if self.hparams.scale_embedding else 1.
        )
        
        # Set token embedding layer
        if token_embedding is None:
            token_embedding = FasterEmbedding(
                self.vocab_size, self.embedding_dim, name = 'token_embedding'
            )
        
        self.token_embedding_layer = token_embedding
        
        # Set token type embedding layer (if required)
        self.token_type_embedding_layer = None
        if self.hparams.max_token_types > 1:
            self.token_type_embedding_layer = FasterEmbedding(
                self.hparams.max_token_types, self.embedding_dim, name = "token_type_embedding"
            )
        
        # Set positional embedding layer
        if positional_embedding is None:
            positional_embedding    = FasterEmbedding(
                self.max_input_length, self.embedding_dim, name = "pos_embeddings"
            )
        self.pos_embedding_layer    = positional_embedding
        
        # Set normalization layer
        self.norm       = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_embedding'
        ) if self.hparams.normalize_embeddings else None
        self.dropout    = tf.keras.layers.Dropout(
            self.hparams.drop_rate
        ) if self.hparams.drop_rate > 0. else None
    
    @property
    def max_input_length(self):
        return self.hparams.max_input_length + self.hparams.positional_offset

    @property
    def use_token_type(self):
        return self.token_type_embedding_layer is not None

    def linear(self, output):
        batch_size, seq_len = tf.shape(output)[0], tf.shape(output)[1]
        
        logits = tf.reshape(output, [-1, self.embedding_dim])
        logits = tf.matmul(logits, self.token_embedding_layer.embeddings, transpose_b = True)
        logits = tf.reshape(logits, [batch_size, seq_len, self.vocab_size])
        
        return logits

    @timer
    def embed_tokens(self, text):
        if len(tf.shape(text)) == 3 and text.dtype == tf.float32:
            token_embedded = text
        else:
            token_embedded = self.token_embedding_layer(text)
        
        return token_embedded * self.embedding_factor
    
    @timer
    def embed_token_types(self, text, token_types):
        token_type_embedded = 0.
        if self.token_type_embedding_layer is not None:
            if token_types is None:
                token_types = tf.fill(tf.shape(text), value = 0)
            elif len(tf.shape(token_types)) == 0:
                token_types = tf.fill(tf.shape(text), value = token_types)
            token_type_embedded = self.token_type_embedding_layer(token_types)
        return token_type_embedded
    
    @timer
    def embed_positions(self,
                        position_ids,
                        seq_len,
                        positional_offset,
                        repeat_position,
                        debug = False
                       ):
        if position_ids is None:
            if repeat_position > 1:
                position_ids = tf.repeat(tf.range(seq_len // repeat_position + 1), repeat_position)[:seq_len]
            else:
                position_ids = tf.range(seq_len)

            position_ids = tf.expand_dims(position_ids, axis = 0)
            if positional_offset > 0:
                position_ids = position_ids + positional_offset
        
        if debug:
            tf.print("Position ids :", position_ids)
        
        return self.pos_embedding_layer(position_ids)
    
    @timer(name = 'token embedding call')
    def call(self,
             inputs,
             input_length   = None,
             token_types    = None,
             position_ids   = None,
             training       = False,
             positional_offset  = -1,
             repeat_position    = -1,
             debug      = False,
             ** kwargs
            ):
        if positional_offset == -1: positional_offset = self.positional_offset
        if repeat_position == -1:   repeat_position = self.repeat_position
        
        text = inputs
        if isinstance(inputs, (list, tuple)):
            text, input_length = inputs[:2]
            if len(inputs) > 2: token_types  = inputs[2]
            if len(inputs) > 3: position_ids = inputs[3]
        
        if debug:
            tf.print("Tokens shape :", tf.shape(text))
            tf.print("Positional offset :", positional_offset)
        
        # Embed tokens (text)
        token_embedded = self.embed_tokens(text)
        
        # Embed token types (if necessary)
        token_type_embedded = self.embed_token_types(text, token_types)
        
        # Embed positions 
        pos_embedded = self.embed_positions(
            position_ids, tf.shape(text)[1], positional_offset, repeat_position, debug = debug
        )
        
        # Combine all embeddings
        embeddings  = token_embedded + pos_embedded + token_type_embedded

        if self.norm is not None:
            embeddings = self.norm(embeddings, training = training and self.norm_training)
        if self.dropout is not None:
            embeddings = self.dropout(embeddings, training = training)

        return embeddings

    def get_output_shape(self, inputs):
        return tuple(inputs) + (self.embedding_dim, )
    
    def get_config(self):
        config = super().get_config()
        return (self.hparams + config).get_config()

class TextTransformerBlock(TransformerBlock):
    """ Regular `TransformerBlock` with a `TransformerTokenEmbedding` layer applied on inputs """
    default_params  = HParamsTextTransformerBlock
    _attr_to_set    = TransformerBlock._attr_to_set + ['vocab_size', 'positional_offset']
    
    def __init__(self, vocab_size, embedding_dim, max_input_length, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim,
            max_input_length = max_input_length, ** kwargs
        )
    
    def _init_input_layers(self,
                           token_embedding = None,
                           positional_embedding = None,
                           ** kwargs
                          ):
        self.embeddings = TransformerTokenEmbedding(
            token_embedding         = token_embedding,
            positional_embedding    = positional_embedding,
            name    = 'embeddings',
            ** self.hparams
        )
    
    @property
    def sos_token(self):
        return self.hparams.sos_token
    
    @property
    def eos_token(self):
        return self.hparams.eos_token
    
    @property
    def max_input_length(self):
        return self.embeddings.max_input_length
    
    @property
    def use_token_type(self):
        return self.embeddings.use_token_type
    
    @property
    def dummy_inputs(self):
        batch_size, seq_len = 2, 32
        text = tf.ones([batch_size, seq_len], dtype = tf.int32)
        text_length = tf.fill([batch_size, 1], seq_len)
        
        return [text, text_length]
    
    def set_tokens(self, sos_token, eos_token):
        self.hparams = self.hparams(sos_token = sos_token, eos_token = eos_token)

    def embed_tokens(self, tokens):
        return self.embeddings.embed_tokens(tokens)
    
    def compute_output(self, output, training = False, mask = None, ** kwargs):
        return output
    
    @timer(name = 'Text Transformer call')
    def call(self,
             inputs,
             input_length   = None,
             token_types    = None,
             position_ids   = None,
             
             mask       = None,
             
             training   = False,
             
             first_layer_idx    = -1,
             last_layer_idx     = -1,
             
             positional_offset  = -1,
             repeat_position    = -1,
             ** kwargs
            ):
        text = inputs
        if first_layer_idx == -1:
            if isinstance(inputs, (list, tuple)):
                text, input_length = inputs[:2]
                if self.use_token_type:
                    if len(inputs) > 2: token_types  = inputs[2]
                    if len(inputs) > 3: position_ids = inputs[3]
                else:
                    if len(inputs) > 2: position_ids = inputs[2]

            embedded = self.embeddings(
                text,
                input_length    = input_length,
                token_types     = token_types,
                position_ids    = position_ids,

                repeat_position = repeat_position,
                positional_offset   = positional_offset,

                training    = training,
                mask    = mask,
                ** kwargs
            )
        else:
            embedded = inputs
        
        outputs = super().call(
            embedded, input_length = input_length, mask = mask, training = training,
            first_layer_idx = first_layer_idx, last_layer_idx = last_layer_idx, ** kwargs
        )
        if last_layer_idx != -1: return outputs
        
        if not isinstance(outputs, (list, tuple, TransformerOutput)): outputs = (outputs, )
        decoder_outputs = outputs[0]

        logits = self.compute_output(
            decoder_outputs, training = training, mask = mask, text = text, ** kwargs
        )
        
        if isinstance(outputs, TransformerOutput):
            return TransformerOutput(logits, * outputs[1:])
        elif len(outputs) > 1:
            return (logits, ) + outputs[1:]
        return logits
    
    @timer
    def infer(self, * args, ** kwargs):
        return infer(
            self,
            * args,
            vocab_size  = self.vocab_size,
            sos_token = self.sos_token,
            eos_token = self.eos_token,
            ** kwargs
        )
    
    def get_output_shape(self, inputs, * args, ** kwargs):
        return super().get_output_shape(
            self.embeddings.get_output_shape(inputs), * args, ** kwargs
        )
    
class TextTransformerEncoder(TextTransformerBlock):
    default_params = HParamsTextTransformerEncoder

class TextTransformerDecoder(TextTransformerBlock):
    default_params = HParamsTextTransformerDecoder

class TextTransformer(Transformer):
    encoder_class   = TextTransformerEncoder
    decoder_class   = TextTransformerDecoder
    default_params  = HParamsTextTransformer
    _shared_keys    = Transformer._shared_keys + _shared_config
    
    def __init__(self, vocab_size, embedding_dim, max_input_length,
                 sos_token = None, eos_token = None, ** kwargs):
        kwargs.update({'sos_token' : sos_token, 'eos_token' : eos_token})
        if sos_token is not None:
            kwargs.update({'decoder_sos_token' : sos_token, 'decoder_eos_token' : eos_token})

        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim,
            max_input_length = max_input_length, ** kwargs
        )

    @property
    def dummy_inputs(self):
        return self.encoder.dummy_inputs + self.decoder.dummy_inputs
        
    def set_tokens(self, sos_token, eos_token):
        self.hparams = self.hparams(sos_token = sos_token, eos_token = eos_token)
        self.decoder.set_tokens(sos_token, eos_token)

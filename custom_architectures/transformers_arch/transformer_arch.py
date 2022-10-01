
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

""" Tensorflow 2.x implementation of the main Transformers' blocks """

import logging
import collections
import tensorflow as tf

from loggers import timer
from hparams.hparams import HParams
from utils.text.text_processing import create_look_ahead_mask, create_padding_mask
from custom_layers import get_activation, MultiHeadAttention, HParamsMHA

time_logger = logging.getLogger('timer')

TransformerOutput = collections.namedtuple(
    "TransformerOutput", [
        "output",
        "state",
        "logits",
        "attention_weights",
        "hidden_states",
        "mask"
    ]
)


_base_enc_dec_kwargs    = {
    'num_layers'    : 4,
    'return_state'      : False,
    'return_logits'     : False,
    'return_attention'  : True,
    'return_hidden_states'  : False,
    'return_mask'       : False
}
_shared_config          = list(_base_enc_dec_kwargs.keys()) + [
    'embedding_dim', 'norm_training', 'epsilon', 'ffn_dim', 'ffn_activation', 'drop_rate'
]
_shared_config.remove('num_layers')

HParamsTransformerLayer = HParams(
    ** HParamsMHA.get_config(add_prefix = 'mha'),
    ** HParamsMHA.get_config(add_prefix = 'enc_mha'),
    embedding_dim   = 512,
    normalize   = 'after',
    epsilon     = 1e-12,
    drop_rate   = 0.1,
    use_encoder_attention   = False,
    use_causal_attention    = False,
    ffn_dim     = 1024,
    ffn_activation  = 'relu',
    norm_training   = True      # whether to allow `training = True` or not
)
HParamsTransformerBlock = HParamsTransformerLayer(** _base_enc_dec_kwargs)

HParamsTransformerEncoder   = HParamsTransformerBlock
HParamsTransformerDecoder   = HParamsTransformerBlock(
    use_encoder_attention = True, use_causal_attention = True
)


HParamsTransformer  = HParams(
    ** HParamsTransformerEncoder.get_config(add_prefix = 'encoder'),
    ** HParamsTransformerDecoder.get_config(add_prefix = 'decoder'),
    ** {
        k : (None if not k.startswith('return_') else _base_enc_dec_kwargs[k])
        for k in _shared_config
    }
)

@timer
def format_output(output,
                  state     = None,
                  logits    = None,
                  attn_weights  = None,
                  hidden_states = None,
                  mask      = None,
                  types     = None,
                  
                  return_state      = False,
                  return_logits     = False,
                  return_attention  = False,
                  return_hidden_states  = False,
                  return_mask       = False,
                  return_types      = False,
                  
                  as_dict       = False,
                  ** kwargs
                 ):
    def _maybe_add(out, key, value, should_return):
        return out if value is None or not should_return else (out + (value, ))
    
    if as_dict:
        return TransformerOutput(
            output  = output,
            state   = state if return_state else None,
            logits  = logits if return_logits else None,
            attention_weights   = attn_weights if return_attention else None,
            hidden_states   = hidden_states if return_hidden_states else None,
            mask    = mask if return_mask else None
        )
    
    out = (output, )
    
    out = _maybe_add(out, 'state',          state,        should_return = return_state)
    out = _maybe_add(out, 'logits',         logits,       should_return = return_logits)
    out = _maybe_add(out, 'attention',      attn_weights, should_return = return_attention)
    out = _maybe_add(out, 'hidden_states',  hidden_states,  should_return = return_hidden_states)
    out = _maybe_add(out, 'mask',           mask,         should_return = return_mask)
    out = _maybe_add(out, 'types',          types,        should_return = return_types)
    
    return out[0] if not as_dict and len(out) == 1 else out

@timer
def build_mask(inputs,
               use_causal_attention,
               input_length     = None,
               mask = None,
               padding_mask = None,
               look_ahead_mask  = None,
               initial_state    = None
              ):
    if mask is not None: return mask

    if padding_mask is None and input_length is not None:
        maxlen = tf.shape(inputs)[1]
        if initial_state is not None: maxlen += tf.shape(initial_state[0])[-2]
        padding_mask = create_padding_mask(
            inputs, seq_len = input_length, maxlen = maxlen, dtype = tf.float32
        )
        
    if not use_causal_attention or initial_state is not None: return padding_mask

    if look_ahead_mask is None:
        look_ahead_mask = create_look_ahead_mask(
            tf.shape(inputs)[0], tf.shape(inputs)[1], tf.float32
        )
    
    if padding_mask is None:
        return look_ahead_mask
    else:
        return tf.maximum(look_ahead_mask, padding_mask)


class FeedForwardNetwork(tf.keras.Model):
    def __init__(self, ffn_dim, ffn_activation, embedding_dim, name = 'ffn'):
        """
            Simple 2-`Dense` sequential network with an activation function between the 2 layers.
            
            Arguments :
                - ffn_dim   : the 1st layer's number of units
                - ffn_activation    : the activation function between the 2 layers
                - embedding_dim     : the Transformers' depth (the number of units for the 2nd layer)
        """
        super().__init__(name = name)
        
        self.ffn_dim    = ffn_dim
        self.ffn_activation = ffn_activation
        self.embedding_dim  = embedding_dim
        
        self.dense_1    = tf.keras.layers.Dense(ffn_dim, name = 'dense_1')
        self.act        = get_activation(ffn_activation)
        self.dense_2    = tf.keras.layers.Dense(embedding_dim, name = 'dense_2')
    
    @timer(name = 'FFN call')
    def call(self, inputs, training = False):
        x = self.dense_1(inputs)
        if self.act is not None: x = self.act(x)
        return self.dense_2(x)

    def get_config(self):
        return {
            'ffn_dim'           : self.ffn_dim,
            'ffn_activation'    : self.ffn_activation,
            'embedding_dim'     : self.embedding_dim
        }
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, name = None, ** kwargs):
        """
            A fully customizable Transformer layer.
            It handles:
                - self-attention    : when Q = K = V
                    The 1st MHA is by default a self-attention layer
                    - In Encoder-only       : there is only 1 self-MHA
                    - In Encoder-Decoder    : there is 1 self-MHA followed by a causal-MHA
                - causal attention  : when using the masking operator
                    Set `use_causal_attention = True` in the constructor
                    The 2nd attention (if `use_encoder_attention = True`) is by default causal
                - Encoder-Decoder mode  : uses 2 MHA (a self-MHA followed by a causal-MHA)
                    Set `use_encoder_attention = True` in the constructor.
                    Note that the 2nd MHA is not a self-MHA as K and V are the `encoder_output` call argument
            
            See the `HParamsTransformerLayer` class for an exhaustive list of configuration. 
                Those starting with `ffn_` are related to the feed-forward network
                Those starting with `mha_` are related to the 1st MHA
                Those starting with `enc_mha_` are related to the 2nd MHA (ignored if `use_encoder_attention = False`)
                
                - normalize : where to apply the `LayerNormalization`
                    - before    : directly on the layer's input
                    - middle    : just before the FFN call but it does not normalize the FFN's residual !
                    `ffn_out = mha_out + norm(ffn(mha_out))` (it is used by `GPT-2` models)
                    - after     : default case where the normalization is applied on the FFN's output
                - use_causal_attention  : whether to use the masking operator or not (on the 1st MHA)
                - use_encoder_attention : whether to use 1 or 2 MHA
            
            Note that the `epsilon` and `norm_training` are propagated to the MHA
        """
        super().__init__(name = name)
        
        self.hparams    = HParamsTransformerLayer.extract(kwargs)
        self.hparams    = self.hparams(
            embedding_dim   = embedding_dim,
            mha_epsilon     = self.hparams.epsilon,
            mha_attention_dim   = embedding_dim,
            mha_norm_training   = self.hparams.norm_training,
            enc_mha_epsilon     = self.hparams.epsilon,
            enc_mha_attention_dim   = embedding_dim,
            enc_mha_norm_training   = self.hparams.norm_training
        )
        
        self.normalize  = self.hparams.normalize
        self.norm_training  = self.hparams.norm_training
        self.use_causal_attention   = self.hparams.use_causal_attention
        
        self.attention  = MultiHeadAttention(
            ** self.hparams.get_config(prefix = 'mha'), name = 'mha'
        )
        self.enc_attention  = MultiHeadAttention(
            ** self.hparams.get_config(prefix = 'enc_mha'), name = 'enc_mha'
        ) if self.hparams.use_encoder_attention else None
        
        self.ffn = FeedForwardNetwork(
            self.hparams.ffn_dim, self.hparams.ffn_activation, embedding_dim, name = 'ffn'
        )
        
        self.norm   = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm'
        ) if self.hparams.normalize else None
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate)
    
    @timer(name = 'layer call')
    def call(self,
             inputs,
             input_length   = None,
             encoder_output = None,
             
             initial_state  = None,
             
             mask       = None,
             padding_mask   = None,
             look_ahead_mask    = None,
             enc_padding_mask   = None,
             
             training   = False,
             return_state       = False,
             return_attention   = True,
             ** kwargs
            ):
        """
            Arguments :
                - inputs    : the layers' input (the query) with shape [B, q_len, embedding_dim]
                - input_length  : the inputs' sequence lengths (to build the padding mask)
                - encoder_output    : encoder output with shape [B, in_seq_len, encoder_embedding_dim]
                
                - initial_state     : state to use (typically the previous iteration state)
                
                - mask  : the mask to use for the 1st MHA
                - padding_mask  : the padding mask for the 1st MHA          [B, 1, seq_len, seq_len]
                - look_ahead_mask   : the causal mask for the 1st MHA       [B, 1, 1, seq_len]
                - enc_padding_mask  : the padding mask used for the 2nd MHA [B, 1, 1, in_seq_len]
                
                - training  : whether it is training / inference phase
                - return_state      : whether to return the internal state or not
                - return_attention  : whether to return attention weights or not
            Return : output if not return_attention else [output, attention]
                - output    : the layer output with same shape as input
                - attention_weights : self-attention weights for each head of the MHA
        """
        if self.normalize == 'before':
            inputs = self.norm(inputs, training = training and self.norm_training)

        if mask is None:
            mask = build_mask(
                inputs, self.use_causal_attention, input_length = input_length,
                look_ahead_mask = look_ahead_mask, padding_mask = padding_mask,
                initial_state = initial_state
            )

        attn_outputs    = self.attention(
            inputs, inputs, inputs, mask = mask, training = training, initial_state = initial_state,
            return_attention = return_attention, return_state = return_state, normalize_kv = True
        )
        if not isinstance(attn_outputs, tuple): attn_outputs = (attn_outputs, )
        attn_out = attn_outputs[0]
        
        if self.enc_attention is not None:
            if encoder_output is None:
                raise ValueError("You must provide encoder output when using encoder attention !")
            
            enc_attn_outputs = self.enc_attention(
                attn_out, encoder_output, encoder_output, mask = enc_padding_mask,
                training = training, return_attention = return_attention,
                return_state = False, normalize_kv = False
            )
            attn_out = enc_attn_outputs
            if return_attention:
                attn_outputs    = attn_outputs[:-1] + ((attn_outputs[-1], enc_attn_outputs[-1]), )
                attn_out        = enc_attn_outputs[0]
        elif encoder_output is not None:
            raise ValueError(
                "You cannot pass `encoder_output` when `self.use_encoder_attention` is False !"
            )
        
        ffn_in = attn_out
        if self.normalize == 'middle':
            ffn_in = self.norm(ffn_in, training = training and self.norm_training)
        
        ffn_output  = self.ffn(ffn_in, training = training)
        ffn_output  = self.dropout(ffn_output, training = training)
        
        output  = ffn_output + attn_out
        
        if self.normalize == 'after':
            output = self.norm(output, training = training and self.norm_training)
        
        return output if len(attn_outputs) == 1 else ((output,) + attn_outputs[1:])
    
    def get_output_shape(self,
                         inputs,
                         encoder_output = None,
                         return_state   = False,
                         return_attention   = True,
                        ):
        attn_out_shape    = self.attention.get_output_shape(
            inputs, inputs, inputs, return_attention = return_attention, return_state = return_state
        )
        
        if self.enc_attention is not None:
            if encoder_output is None:
                raise ValueError("You must provide encoder output when using encoder attention !")
            
            enc_attn_out_shape = self.enc_attention.get_output_shape(
                inputs, encoder_output, encoder_output,
                return_attention = return_attention, return_state = False
            )
            if return_attention:
                attn_out_shape  = attn_out_shape[:-1] + (
                    (attn_out_shape[-1], enc_attn_out_shape[-1]), 
                )
        elif encoder_output is not None:
            raise ValueError(
                "You cannot pass `encoder_output` when `self.use_encoder_attention` is False !"
            )
        
        return attn_out_shape
    
    def get_config(self):
        config = super().get_config()
        return (self.hparams + config).get_config()

class TransformerBlock(tf.keras.Model):
    default_params  = HParamsTransformerBlock
    _attr_to_set    = [
        'embedding_dim', 'norm_training', 'use_causal_attention',
        'return_state', 'return_attention', 'return_hidden_states', 'return_mask'
    ]
    
    def __init__(self, embedding_dim, num_layers, name = None, ** kwargs):
        """ Simply a list of `num_layers` TransformerLayer applied sequentially """
        super().__init__(name = name)
        self.hparams    = self.default_params.extract(kwargs)
        self.hparams    = self.hparams(embedding_dim = embedding_dim, num_layers = num_layers)
        
        for config in self._attr_to_set:
            setattr(self, config, self.hparams[config])
        
        self._init_input_layers(** kwargs)
        
        self._layers = [
            TransformerLayer(name = 'layer_{}'.format(i), ** self.hparams)
            for i in range(self.hparams.num_layers)
        ]
    
    def _init_input_layers(self, ** kwargs):
        pass
    
    def _build(self):
        if hasattr(self, 'dummy_inputs'):
            self(self.dummy_inputs, training = False)
    
    def __len__(self):
        return len(self._layers)
    
    def __getitem__(self, idx):
        return self._layers[idx]
    
    def freeze(self, trainable = False):
        self.trainable = trainable

    @property
    def output_last_dim(self):
        return self.embedding_dim

    @timer(name = 'Transformer block call')
    def call(self,
             inputs,
             input_length   = None,
             encoder_output = None,
             initial_state  = None,
             
             mask       = None,
             padding_mask   = None,
             look_ahead_mask    = None,
             enc_padding_mask   = None,
             
             training   = False,
             
             first_layer_idx    = -1,
             last_layer_idx     = -1,
             
             return_state       = None,
             return_attention   = None,
             return_last_attention  = None,
             return_hidden_states   = None,
             return_mask        = None,
             as_dict    = False,
             ** kwargs
            ):
        """ See the TransformerLayer
            Arguments :
                - inputs    : block inputs with shape [batch_size, seq_len, embedding_dim], embedded inputs
                - mask      : attention mask (padding mask based in inputs)
                - training  : whether it is training / inference phase
                - return_attention  : whether to return attention weights or not
                - return_states     : whether to return intermediate representation or not
            Return : output if not return_attention else [output, attention]
                - output    : the layer output with same shape as input
                - attention_weights : dict self-attention weights for each head of the MHA of each layer
        """
        if first_layer_idx == -1:   first_layer_idx = 0
        if last_layer_idx == -1:    last_layer_idx = len(self._layers)
        
        if return_state is None:            return_state = self.return_state
        if return_attention is None:        return_attention = self.return_attention
        if return_hidden_states is None:    return_hidden_states = self.return_hidden_states
        if return_mask is None:             return_mask = self.return_mask

        states              = () if return_state else None
        attention_weights   = {} if return_attention or return_last_attention else None
        hidden_states       = {} if return_hidden_states else None

        if isinstance(inputs, (list, tuple)): inputs, input_length = inputs
        
        if mask is None:
            mask = build_mask(
                inputs, self.use_causal_attention, input_length = input_length,
                look_ahead_mask = look_ahead_mask, padding_mask = padding_mask,
                initial_state = initial_state[0] if initial_state is not None else None
            )
        
        output = inputs
        for i, layer in enumerate(self._layers[first_layer_idx : last_layer_idx], start = first_layer_idx):
            output, state, attn_weights = layer(
                output,
                input_length    = input_length,
                encoder_output  = encoder_output,
                initial_state   = initial_state[i] if initial_state is not None else None,
                training    = training,
                mask    = mask,
                padding_mask    = padding_mask,
                look_ahead_mask = look_ahead_mask,
                enc_padding_mask    = enc_padding_mask,
                return_attention    = True,
                return_state        = True,
                ** kwargs
            )
            if return_state:
                states  = states + (state, )
            
            if return_attention or (return_last_attention == True and i == len(self._layers) - 1):
                if not isinstance(attn_weights, tuple):
                    attention_weights['attn_{}'.format(layer.name)] = attn_weights
                else:
                    attention_weights['attn_{}'.format(layer.name)] = attn_weights[0]
                    attention_weights['enc_attn_{}'.format(layer.name)] = attn_weights[1]
            
            if return_hidden_states:
                hidden_states['state_{}'.format(layer.name)] = output
        
        return format_output(
            output,
            state   = states,
            attn_weights    = attention_weights,
            hidden_states   = hidden_states,
            mask    = mask,
            
            return_state        = return_state,
            return_attention    = return_attention or return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask         = return_mask,
            as_dict = as_dict
        )
    
    def get_output_shape(self,
                         inputs,
                         encoder_output = None,
                         return_state   = None,
                         return_attention   = None,
                         return_last_attention  = None,
                         return_hidden_states   = None,
                         return_mask        = None,
                         as_dict    = False
                        ):
        output_shape    = inputs[:-1] + (self.output_last_dim, )
        
        mask_shape  = None
        
        states_shape              = () if return_state else None
        attention_weights_shape   = {} if return_attention or return_last_attention else None
        hidden_states_shape       = {} if return_hidden_states else None
        
        output = inputs
        for i, layer in enumerate(self._layers):
            output, state, attn_weights = layer.get_output_shape(
                output,
                encoder_output  = encoder_output,
                return_attention    = True,
                return_state        = True
            )
            if return_state:
                states_shape  = states_shape + (state, )
            
            if return_attention or (return_last_attention == True and i == len(self._layers) - 1):
                if len(attn_weights) != 2:
                    attention_weights_shape['attn_{}'.format(layer.name)] = attn_weights
                else:
                    attention_weights_shape['attn_{}'.format(layer.name)] = attn_weights[0]
                    attention_weights_shape['enc_attn_{}'.format(layer.name)] = attn_weights[1]
            
            if return_hidden_states:
                hidden_states_shape['state_{}'.format(layer.name)] = output
        
        return format_output(
            output_shape,
            state   = states_shape,
            attn_weights    = attention_weights_shape,
            hidden_states   = hidden_states_shape,
            mask    = mask_shape,
            
            return_state        = return_state,
            return_attention    = return_attention or return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask         = return_mask,
            as_dict = as_dict
        )
    
    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

class TransformerEncoder(TransformerBlock):
    default_params = HParamsTransformerEncoder

class TransformerDecoder(TransformerBlock):
    default_params = HParamsTransformerDecoder

class Transformer(tf.keras.Model):
    encoder_class   = TransformerEncoder
    decoder_class   = TransformerDecoder
    default_params  = HParamsTransformer
    _shared_keys    = _shared_config
    _attr_to_set    = [
        'return_state', 'return_attention', 'return_hidden_states', 'return_mask'
    ]
    
    def __init__(self,
                 name = None,
                 shared_layers = {},
                 encoder_wrapper = None,
                 encoder_wrapper_params = None,
                 decoder_wrapper = None,
                 decoder_wrapper_params = None,
                 ** kwargs
                ):
        super().__init__(name = name)
        default_params  = self.default_params
        if encoder_wrapper is None: encoder_wrapper = lambda x, ** kwargs: x
        elif encoder_wrapper_params is not None:
            default_params = default_params(
                ** encoder_wrapper_params.get_config(add_prefix = 'encoder')
            )
        if decoder_wrapper is None: decoder_wrapper = lambda x, ** kwargs: x
        elif decoder_wrapper_params is not None:
            default_params = default_params(
                ** decoder_wrapper_params.get_config(add_prefix = 'decoder')
            )
        
        self.hparams = default_params.extract(kwargs)
        # Allow to have different embedding dim for encoder and decoder
        _shared = {}
        for k in self._shared_keys:
            if self.hparams[k] is not None:
                _shared.update({
                    'encoder_{}'.format(k) : self.hparams[k],
                    'decoder_{}'.format(k) : self.hparams[k]
                })
        self.hparams.update(_shared)
        
        for config in self._attr_to_set:
            setattr(self, config, self.hparams[config])
        
        self.encoder    = encoder_wrapper(self.encoder_class(
            ** self.hparams.get_config(prefix = 'encoder'), ** shared_layers, name = 'encoder'
        ), ** self.hparams.get_config(prefix = 'encoder'))
        
        self.decoder    = decoder_wrapper(self.decoder_class(
            ** self.hparams.get_config(prefix = 'decoder'), ** shared_layers, name = 'decoder'
        ), ** self.hparams.get_config(prefix = 'decoder'))
    
    def _build(self):
        if hasattr(self, 'dummy_inputs'):
            self(self.dummy_inputs, training = False)

    @timer(name = 'Transformer call')
    def call(self,
             inputs,
             input_length   = None,
             decoder_input  = None,
             decoder_input_length   = None,
             initial_state  = None,
             
             training   = False,
             mask       = None,
             padding_mask   = None,
             look_ahead_mask    = None,
             enc_padding_mask   = None,
             
             return_state       = None,
             return_attention   = None,
             return_last_attention  = None,
             return_hidden_states   = None,
             return_mask        = None,
             as_dict    = False,
             ** kwargs
            ):
        if return_state is None:            return_state = self.return_state
        if return_attention is None:        return_attention = self.return_attention
        if return_hidden_states is None:    return_hidden_states = self.return_hidden_states
        if return_mask is None:             return_mask = self.return_mask
        
        encoder_input = inputs
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                encoder_input, decoder_input = inputs
            else:
                encoder_input, decoder_input = inputs[:-2], inputs[-2:]
        
        time_logger.start_timer('Encoder')
        
        encoder_outputs = self.encoder(
            encoder_input,
            input_length    = input_length,
            mask    = enc_padding_mask,
            training    = training,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = True,
            as_dict = True,
            
            ** {k[8:] : v for k, v in kwargs.items() if k.startswith('encoder_')},
            ** kwargs
        )
        encoder_output      = encoder_outputs.output
        enc_padding_mask    = encoder_outputs.mask
        
        time_logger.stop_timer('Encoder')
        time_logger.start_timer('Decoder')
        
        decoder_outputs = self.decoder(
            decoder_input,
            input_length    = decoder_input_length,
            encoder_output  = encoder_output,
            mask    = mask,
            padding_mask    = padding_mask,
            look_ahead_mask = look_ahead_mask,
            enc_padding_mask    = enc_padding_mask,
            training    = training,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = return_mask,
            as_dict     = True,
            
            ** {k[8:] : v for k, v in kwargs.items() if k.startswith('decoder_')},
            ** kwargs
        )
        
        time_logger.stop_timer('Decoder')

        return format_output(
            decoder_outputs.output,
            state   = (encoder_outputs.state, decoder_outputs.state),
            attn_weights    = (encoder_outputs.attention_weights, decoder_outputs.attention_weights),
            hidden_states   = (encoder_outputs.hidden_states, decoder_outputs.hidden_states),
            mask    = (encoder_outputs.mask, decoder_outputs.mask),
            
            return_state            = return_state,
            return_attention        = return_attention or return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = return_mask,
            as_dict = as_dict
        )
    
    @timer
    def infer(self,
              inputs,
              input_length   = None,
              decoder_input  = None,
              decoder_input_length   = None,
              training  = False,
              enc_padding_mask   = None,
              
              use_cache = False,
              return_state       = None,
              return_attention   = None,
              return_last_attention = None,
              return_hidden_states   = None,
              return_mask        = None,
              as_dict   = False,
              
              ** kwargs
             ):
        if return_state is None:            return_state = self.return_state
        if return_attention is None:        return_attention = self.return_attention
        if return_hidden_states is None:    return_hidden_states = self.return_hidden_states
        if return_mask is None:             return_mask = self.return_mask
        
        encoder_outputs = self.encoder(
            inputs,
            input_length    = input_length,
            mask    = enc_padding_mask,
            training    = training,
            
            return_state    = False,
            return_attention    = False,
            return_hidden_states    = False,
            return_mask     = True,
            as_dict     = True,
            
            ** {k[8:] : v for k, v in kwargs.items() if k.startswith('encoder_')},
            ** kwargs
        )
        encoder_output      = encoder_outputs.output
        enc_padding_mask    = encoder_outputs.mask
        
        return self.decoder.infer(
            encoder_output  = encoder_output,
            enc_padding_mask    = enc_padding_mask,
            training    = training,
            use_cache   = use_cache,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask     = return_mask,
            as_dict = True,
            
            ** {k[8:] : v for k, v in kwargs.items() if k.startswith('decoder_')},
            ** kwargs
        )
    
    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

custom_functions    = {
    'FeedForwardNetwork'    : FeedForwardNetwork,
    'TransformerEncoder'    : TransformerEncoder,
    'TransformerDecoder'    : TransformerDecoder,
    'Transformer'       : Transformer
}

custom_objects  = {
    'MultiHeadAttention'        : MultiHeadAttention,
    
    'FeedForwardNetwork'    : FeedForwardNetwork,
    'TransformerLayer'      : TransformerLayer,
    'TransformerBlock'      : TransformerBlock,
    'TransformerEncoder'    : TransformerEncoder,
    'TransformerDecoder'    : TransformerDecoder,
    'Transformer'       : Transformer
}

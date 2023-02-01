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

""" TF 2.0 CLIP (Visual Transformer), compatible with the official CLIP implementation """

import tensorflow as tf

from custom_layers import get_activation, FasterEmbedding
from custom_architectures.current_blocks import _get_pooling_layer
from custom_architectures.transformers_arch.embedding_head import HParamsEmbeddingHead, EmbeddingHead
from custom_architectures.transformers_arch.transformer_arch import (
    HParamsTransformerEncoder, TransformerEncoder
)

HParamsVisualTransformer  = HParamsTransformerEncoder(
    ** HParamsEmbeddingHead(token_selector = 'first'),
    input_dim   = -1,
    
    filters = -1,
    kernel_size = 3,
    strides     = -1,
    conv_bias   = False,
    padding     = 'valid',
    
    conv_normalize  = True,
    conv_activation = None,
    
    pooling     = None,
    pool_size   = 2,
    pool_strides    = 2,
    
    conv_drop_rate  = 0.1,
    
    normalize   = 'middle',
    normalize_output    = True,
    mha_normalize   = False,
    mha_normalize_input = True,
    
    ffn_activation  = 'quick_gelu',
    
    mha_epsilon     = 1e-5,
    epsilon     = 1e-5
)

class VisualTransformer(TransformerEncoder):
    default_params  = HParamsVisualTransformer
    _attr_to_set    = TransformerEncoder._attr_to_set + ['input_dim']
    
    def __init__(self, input_dim, embedding_dim, ** kwargs):
        super().__init__(
            input_dim = input_dim, embedding_dim = embedding_dim, ** kwargs
        )
        
        self.embedding_head = EmbeddingHead(** self.hparams, name = 'embedding_layer')
    
    def _init_input_layers(self, ** kwargs):
        self.conv   = tf.keras.layers.Conv2D(
            filters     = self.hparams.filters,
            kernel_size = self.hparams.kernel_size,
            use_bias    = self.hparams.conv_bias,
            strides     = self.hparams.strides,
            padding     = self.hparams.padding,
            name        = 'conv1'
        )
        self.conv_norm  = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = 'norm_conv'
        ) if self.hparams.conv_normalize else None
        self.conv_act   = get_activation(self.hparams.conv_activation)
        self.pooling    = _get_pooling_layer(
            self.hparams.pooling, dim = '2d',
            pool_size   = self.hparams.pool_size,
            pool_strides    = self.hparams.pool_strides
        )
        self.conv_drop  = tf.keras.layers.Dropout(self.hparams.conv_drop_rate) if self.hparams.conv_drop_rate > 0 else None
        
        ctx_length  = (self.hparams.input_dim // self.hparams.strides) ** 2 + 1
        with tf.name_scope(self.name):
            self.class_embedding    = self.add_weight(
                shape = (self.embedding_dim, ), name = 'class_embedding'
            )
            self.positional_embedding   = self.add_weight(
                shape = (ctx_length, self.embedding_dim), name = 'positional_embedding'
            )
    
    @property
    def input_signature(self):
        return tf.TensorSpec(shape = (None, self.input_dim, self.input_dim, 3), dtype = tf.float32)
    
    def compute_output(self, output, training = False, mask = None, inputs = None, ** kwargs):
        output = super().compute_output(output, training = training, mask = mask)
        
        return self.embedding_head(output, training = training, mask = mask, ** kwargs)

    def prepare_input(self, inputs, training = False, mask = None, ** kwargs):
        embedded = self.conv(inputs)
        if self.conv_act is not None:   embedded = self.conv_act(embedded)
        if self.pooling is not None:    embedded = self.pooling(embedded)
        embedded = tf.reshape(embedded, [tf.shape(embedded)[0], -1, tf.shape(embedded)[-1]])
        
        embedded = tf.concat([
            tf.broadcast_to(
                self.class_embedding, [tf.shape(embedded)[0], 1, tf.shape(embedded)[-1]]
            ),
            embedded
        ], axis = 1)
        embedded = embedded + tf.expand_dims(self.positional_embedding, axis = 0)
        if self.conv_norm is not None:  embedded = self.conv_norm(embedded, training = training)
        if self.conv_drop is not None:  embedded = self.conv_drop(embedded, training = training)
        
        embedded._keras_mask = tf.ones((tf.shape(embedded)[0], tf.shape(embedded)[1]), dtype = tf.bool)
        return embedded
    
    def transfer_weights(self, pretrained, ** kwargs):
        from models.weights_converter import (
            _transformer_patterns, _attn_split, name_based_partial_transfer_learning
        )

        kwargs.setdefault('transforms', _attn_split)

        if isinstance(pretrained, dict):
            pretrained = {k : v for k, v in pretrained.items() if 'visual' in k}
        
        return name_based_partial_transfer_learning(
            self, pretrained, skip_root = False, patterns = {
                ** _transformer_patterns, 'norm_pre' : 'transformer/norm_conv'
            }, ** kwargs
        )

    @classmethod
    def from_pretrained(cls, pretrained_name = 'RN50', pretrained = None,** kwargs):
        from custom_architectures.clip_arch import load_clip
        from models.weights_converter import get_pt_variables, get_pt_layers, transpose_weights

        state_dict  = load_clip(pretrained_name, pretrained = pretrained)
        
        embedding_dim   = state_dict["visual.conv1.weight"].shape[0]
        kernel_size = state_dict["visual.conv1.weight"].shape[-1]

        num_layers  = len([
            k for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        input_dim = kernel_size * grid_size
        output_dim  = state_dict["visual.proj"].shape[1]
        
        config = HParamsVisualTransformer(
            input_dim   = input_dim,
            output_dim  = output_dim,
            output_bias = False,
            
            filters     = embedding_dim,
            kernel_size = kernel_size,
            strides     = kernel_size,
            padding     = 'valid',
            
            embedding_dim   = embedding_dim,
            num_layers      = num_layers,
            mha_num_heads   = embedding_dim // 64,
            ffn_dim         = embedding_dim * 4
        )
        with tf.device('cpu'):
            instance = cls(** config(** kwargs))
            instance._build()

        instance.transfer_weights(state_dict, ** kwargs)
        
        return instance

custom_functions    = {
    'VisualTransformer' : VisualTransformer
}

custom_objects  = {
    ** custom_functions,
    'EmbeddingHead' : EmbeddingHead
}

_encoders   = custom_functions
_transformers   = _encoders
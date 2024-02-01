# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
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
import tensorflow as tf

from custom_layers import get_activation, CustomActivation, MultiHeadAttention
from custom_architectures.current_blocks import _get_var, Conv2DBN, _get_pooling_layer

class Attention2D(tf.keras.layers.Layer):
    def __init__(self, spatial_dim, embedding_dim, num_heads, output_dim = None, ** kwargs):
        super().__init__(** kwargs)
        
        self.num_heads  = num_heads
        self.spatial_dim    = spatial_dim
        self.embedding_dim  = embedding_dim
        self.output_dim = output_dim
        
        with tf.name_scope(self.name):
            self.pos_encoding   = self.add_weight(
                shape = (self.spatial_dim ** 2 + 1, embedding_dim), name = 'pos_encoding'
            )

        self.mha    = MultiHeadAttention(
            num_heads   = num_heads,
            attention_dim   = embedding_dim,
            output_dim  = output_dim if output_dim else embedding_dim,
            residual    = False,
            normalize   = False,
            name    = 'mha'
        )
    
    def call(self, inputs, training = False):
        x = inputs
        x = tf.reshape(inputs, [tf.shape(x)[0], -1, tf.shape(x)[-1]])
        
        x = tf.concat([tf.reduce_mean(x, axis = 1, keepdims = True), x], axis = 1)
        x = x + tf.expand_dims(self.pos_encoding, axis = 0)
        
        return tf.squeeze(self.mha(x[:, :1], x, x, training = training, return_attention = False), 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads'  : self.num_heads,
            'spatial_dim'    : self.spatial_dim,
            'embedding_dim'  : self.embedding_dim,
            'output_dim' : self.output_dim
        })
        return config

def Bottleneck(x, filters, stride = 1, use_bias = False, activation = 'relu', expansion = 4,
               pool_type = 'avg', name = None, ** kwargs):
    residual = x
    
    x = Conv2DBN(
        x,
        filters     = filters,
        kernel_size = 1,
        use_bias    = use_bias,
        activation  = activation,
        bnorm       = 'after',
        bn_name     = name if not name else '{}bn1'.format(name),
        name        = name if not name else '{}conv1'.format(name),
        ** kwargs
    )
    
    x = tf.keras.layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(x)
    x = Conv2DBN(
        x,
        filters     = filters,
        kernel_size = 3,
        use_bias    = use_bias,
        activation  = activation,
        bnorm       = 'after',
        bn_name     = name if not name else '{}/bn2'.format(name),
        name        = name if not name else '{}/conv2'.format(name),
        ** kwargs
    )

    if stride > 1:
        x = _get_pooling_layer(pool_type, '2d', strides = stride)(x)
    
    x = Conv2DBN(
        x,
        filters     = filters * expansion,
        kernel_size = 1,
        use_bias    = use_bias,
        activation  = None,
        bnorm       = 'after',
        bn_name     = name if not name else '{}/bn3'.format(name),
        name        = name if not name else '{}/conv3'.format(name),
        ** kwargs
    )

    if stride > 1 or tuple(x.shape) != tuple(residual.shape):
        if stride > 1:
            residual = _get_pooling_layer(pool_type, '2d', strides = stride)(residual)
        residual = Conv2DBN(
            residual,
            filters     = filters * expansion,
            kernel_size = 1,
            use_bias    = use_bias,
            activation  = None,
            bnorm       = 'after',
            bn_name     = name if not name else '{}downsample/1'.format(name),
            name        = name if not name else '{}downsample/0'.format(name),
            ** kwargs
        )

    x = tf.keras.layers.Add(name = '{}/add'.format(name) if name else None)([x, residual])
    x = get_activation(activation)(x)
    
    return x

def ModifiedResnet(input_shape  = 224,
                   output_dim   = 1024,
                   blocks   = [3, 4, 6, 3],
                   width    = 64,
                   activation   = 'relu',
                   pool_type    = 'avg',
                   expansion    = 4,
                   num_heads    = None,
                   output_normalize = True,
                   name     = None,
                   ** kwargs
                  ):
    if not num_heads: num_heads = width * 32 // 64
    
    if not isinstance(input_shape, tuple): input_shape = (input_shape, input_shape, 3)
    inputs  = tf.keras.layers.Input(shape = input_shape, name = 'input_image')

    x = inputs
    for i, filters in enumerate([width // 2, width // 2, width]):
        x = tf.keras.layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(x)
        x = Conv2DBN(
            x,
            filters = filters,
            kernel_size = 3,
            strides = 2 if i == 0 else 1,
            use_bias    = False,
            bnorm   = 'after',
            epsilon = 1e-5,
            momentum    = 0.1,
            activation  = activation,
            pooling = pool_type if i == 2 else None,
            bn_name = 'bn{}'.format(i+1),
            name    = 'conv{}'.format(i+1)
        )

    for i, filters in enumerate([width, width * 2, width * 4, width * 8]):
        for b in range(_get_var(blocks, i)):
            x = Bottleneck(
                x,
                filters = filters,
                stride = 2 if i > 0 and b == 0 else 1,
                activation  = activation,
                expansion   = expansion,
                pool_type   = pool_type,
                epsilon     = 1e-5,
                momentum    = 0.1,
                name    = 'layer{}/{}/'.format(i+1, b)
            )
    
    embedding_dim = width * 32
    out = Attention2D(
        input_shape[0] // 32, embedding_dim, num_heads, output_dim, name = 'attnpool'
    )(x)

    if output_normalize: out = CustomActivation('l2_normalize')(out)
    
    return tf.keras.Model(inputs = inputs, outputs = out, name = name)

def transfer_weights(model, state_dict, ** kwargs):
    from models.weights_converter import _attn_patterns, name_based_partial_transfer_learning

    return name_based_partial_transfer_learning(
        model, state_dict, patterns = _attn_patterns
    )    

def from_clip_pretrained(pretrained_name = 'RN50', pretrained = None, ** kwargs):
    from custom_architectures.clip_arch import load_clip

    state_dict = load_clip(pretrained_name, pretrained = pretrained)
    
    blocks = [
        len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
        for b in [1, 2, 3, 4]
    ]
    output_grid_dim = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    assert output_grid_dim ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    input_dim = output_grid_dim * 32

    config  = {
        'input_shape' : input_dim,
        'output_dim'  : state_dict["visual.attnpool.c_proj.weight"].shape[0],
        'blocks'      : blocks,
        'width'       : state_dict["visual.layer1.0.conv1.weight"].shape[0],
    }

    model = ModifiedResnet(** {** config, ** kwargs})

    transfer_weights(model, state_dict)
    
    return model

custom_functions    = {
    'ModifiedResnet'    : ModifiedResnet
}

custom_objects      = {
    'CustomActivation'  : CustomActivation,
    'Attention2D'   : Attention2D
}
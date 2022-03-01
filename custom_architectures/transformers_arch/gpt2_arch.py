
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

""" TF 2.0 OpenAI GPT-2 model, compatible with the `transformers`' checkpoint."""

import tensorflow as tf

from custom_architectures.transformers_arch.text_transformer_arch import TextTransformerEncoder, HParamsTextTransformerEncoder

HParamsBaseGPT2  = HParamsTextTransformerEncoder

class BaseGPT2(TextTransformerEncoder):
    default_params  = HParamsBaseGPT2
    
    def __init__(self, vocab_size, embedding_dim, ** kwargs):
        super().__init__(vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs)
        
        self.norm       = tf.keras.layers.LayerNormalization(epsilon = self.hparams.epsilon)

    def compute_output(self, output, training = False, mask = None, ** kwargs):
        return self.norm(output, training = training and self.norm_training)

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'gpt2',
                        pretrained_task = 'generation',
                        pretrained      = None,
                        ** kwargs
                       ):
        from models.weights_converter import partial_transfer_learning

        if pretrained is None:
            from transformers import TFGPT2Model
            with tf.device('cpu') as d:
                pretrained = TFGPT2Model.from_pretrained(pretrained_name)

        config = HParamsBaseGPT2(
            vocab_size      = pretrained.config.vocab_size,
            embedding_dim   = pretrained.config.n_embd,
            max_input_length    = pretrained.config.n_positions,
            use_causal_attention    = True,
            normalize_embeddings    = False,
            sos_token   = 50256,
            eos_token   = 50256,
            
            num_layers  = pretrained.config.n_layer,
            normalize   = 'middle',
            ffn_dim     = 3072,
            ffn_activation  = 'gelu', #pretrained.config.activation_function,
            mha_normalize   = False,
            mha_normalize_input = True,
            mha_num_heads   = pretrained.config.n_head,
            mha_epsilon     = 1e-5,
            epsilon     = 1e-5,
        )

        instance = cls(** config(** kwargs))
        instance._build()

        offset, n_layer_weights = 2, 12
        
        p_weights = pretrained.get_weights()
        weights = [p_weights[1], p_weights[0]]
        # Invert `key` and `value` weights for each MHA layer
        for i in range(pretrained.config.n_layer):
            start_idx = i * n_layer_weights + offset
            # Attn Q / K / V weights
            attn_weights = [p_weights[start_idx + 2], p_weights[start_idx + 3]]
            for w1, w2 in zip(tf.split(attn_weights[0], 3, 1), tf.split(attn_weights[1][0], 3, 0)):
                weights.extend([w1, w2])
            # attention output dense
            weights.extend([tf.squeeze(w) for w in p_weights[start_idx + 4 : start_idx + 6]])
            # attention input normalization
            weights.extend(p_weights[start_idx : start_idx + 2])
            # FFN weights
            weights.extend([tf.squeeze(w) for w in p_weights[start_idx + 8 : start_idx + n_layer_weights]])
            # normalization
            weights.extend([tf.squeeze(w) for w in p_weights[start_idx + 6 : start_idx + 8]])

        weights.extend(p_weights[-2:])  # final normalization
        
        partial_transfer_learning(instance, weights)

        return instance

class GPT2(BaseGPT2):
    def compute_output(self, output, training = False, mask = None, ** kwargs):
        output = super().compute_output(output, training = training, mask = mask, ** kwargs)
        
        return self.embeddings.linear(output)
    
custom_functions    = {
    'BaseGPT2'      : BaseGPT2,
    'GPT2'          : GPT2
}

custom_objects  = custom_functions

_encoders   = {'GPT2' : GPT2}

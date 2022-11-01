
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

import enum
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import model_from_json

from loggers import timer
from hparams import HParams
from utils import get_enum_item
from utils.sequence_utils import pad_to_multiple
from custom_layers import FasterEmbedding
from custom_architectures.transformers_arch.transformer_arch import Transformer, TransformerBlock, build_mask, format_output
from custom_architectures.transformers_arch.bart_arch import Bart, BartEncoder, HParamsBart
from custom_architectures.transformers_arch.text_transformer_arch import *

class SubsamplingMode(enum.IntEnum):
    SELECT  = 0
    DENSE   = 1
    CONV    = 2
    SEPARABLE   = 3
    MIN     = 4
    MAX     = 5
    MEAN    = 6

HParamsMAGWrapper = HParams(
    subsample_at    = -1,
    subsample_after = True,
    subsampling_step    = -1,
    subsampling_offset  = 1,
    subsampling_mode    = 'select',
    subsampling_drop_rate   = 0.,

    repeat_pos_idx      = False,
    
    use_type_embedding      = False,
    random_training_type    = True,
    max_types   = 16
)

@timer
def concat_embeddings(embeddings,
                      mask      = None,
                      merge_embeddings  = False,
                      debug     = False,
                      ** kwargs
                     ):
    question, contexts = embeddings[0], embeddings[1:]
    q_mask, c_masks     = (mask[0], mask[1:]) if mask is not None else (None, None)
    
    c_lengths   = [tf.shape(c)[-2] for c in contexts]
    contexts    = tf.concat(contexts, axis = 1) if len(contexts) > 1 else contexts[0]
    if c_masks is not None:
        if tf.shape(c_masks[0])[-2] > 1:
            c_masks = tuple([tf.reduce_min(m, axis = -2, keepdims = True) for m in c_masks])
        c_masks = tf.concat(c_masks, axis = -1) if len(c_masks) > 1 else c_masks[0]
    if q_mask is not None and tf.shape(q_mask)[-2] > 1:
        q_mask = tf.reduce_min(q_mask, axis = -2, keepdims = True)
    
    lengths     = [tf.shape(question)[1]] + c_lengths
    
    if debug:
        tf.print("Sequence lengths :", lengths)
        tf.print("1st input shape :", tf.shape(question))
        tf.print("2nd input shape :", tf.shape(contexts))
        if c_masks is not None:
            tf.print("Masks shape :", tf.shape(c_masks))
    
    n_doc_per_batch = 1
    q_batch_size, c_batch_size = tf.shape(question)[0], tf.shape(contexts)[0]
    
    # flatten contexts from [B, n_doc, ctx_len, emb_dim] to [B, n_doc * ctx_len, emb_dim]
    if len(tf.shape(contexts)) == 4:
        if len(c_lengths) > 1:
            raise NotImplementedError("When passing multiple document / batch at once, you cannot pass multiple contexts, please flatten everything !")

        n_doc_per_batch = tf.shape(contexts)[1]
        
        ctx_types = tf.repeat(tf.range(1, n_doc_per_batch + 1), tf.shape(contexts)[2])
        
        contexts    = tf.reshape(contexts, [c_batch_size, -1, tf.shape(contexts)[-1]])
        if c_masks is not None:
            c_masks = tf.reshape(c_masks, [c_batch_size, 1, 1, -1])

        if debug:
            tf.print("Contexts (after flattening) shape :", tf.shape(contexts))
            if c_masks is not None:
                tf.print("Masks (after flattening) shape :", tf.shape(c_masks))
        
        if c_masks is not None:
            not_padding = tf.reduce_any(tf.reshape(c_masks, [c_batch_size, -1]) == 0, axis = 0)

            contexts    = tf.boolean_mask(contexts, not_padding, axis = 1)
            c_masks     = tf.boolean_mask(c_masks, not_padding, axis = 3)
            ctx_types   = tf.boolean_mask(ctx_types, not_padding, axis = 0)
            
            if debug:
                tf.print("# padding :", tf.reduce_sum(1 - tf.cast(not_padding, tf.int32)))
                tf.print("Contexts (after removing padding) shape :", tf.shape(contexts))
                tf.print("Masks (after removing padding) shape :", tf.shape(c_masks))
                
    elif len(c_lengths) > 1:
        ctx_types   = tf.concat([
            tf.fill([length], i + 1) for i, length in enumerate(c_lengths)
        ], axis = -1)
    else:
        ctx_types   = tf.fill((tf.shape(contexts)[1], ), 1)
    
    # Merge contexts (if required)
    if merge_embeddings and q_batch_size > 1 and q_batch_size == c_batch_size:
        if len(c_lengths) > 1:
            raise NotImplementedError("When merging contexts, you can only pass 1 context / batch !")
        
        ctx_add_type = tf.repeat(tf.range(q_batch_size), tf.shape(contexts)[1])

        contexts = tf.reshape(
            tf.tile(contexts, [c_batch_size, 1, 1]), 
            [c_batch_size, -1, tf.shape(contexts)[-1]]
        )
        if c_masks is not None:
            c_masks = tf.reshape(
                tf.tile(c_masks, [c_batch_size, 1, 1, 1]), 
                [c_batch_size, 1, 1, -1]
            )
        
        if debug:
            tf.print("Contexts (after merging) shape :", tf.shape(contexts))
            if c_masks is not None:
                tf.print("Masks (after merging) shape :", tf.shape(c_masks))
        
        ctx_types = tf.tile(ctx_types, [q_batch_size]) + n_doc_per_batch * ctx_add_type
        
        if c_masks is not None:
            not_padding = tf.reduce_any(tf.reshape(c_masks, [c_batch_size, -1]) == 0, axis = 0)

            contexts    = tf.boolean_mask(contexts, not_padding, axis = 1)
            c_masks     = tf.boolean_mask(c_masks, not_padding, axis = 3)
            ctx_types   = tf.boolean_mask(ctx_types, not_padding, axis = 0)
            
            if debug:
                tf.print("# padding :", tf.reduce_sum(1 - tf.cast(not_padding, tf.int32)))
                tf.print("Contexts (after removing padding) shape :", tf.shape(contexts))
                tf.print("Masks (after removing padding) shape :", tf.shape(c_masks))

    
    types   = tf.concat([tf.fill([tf.shape(question)[1]], 0), ctx_types], axis = -1)
    
    memory  = tf.concat([question, contexts], axis = 1)
    masks   = tf.concat([q_mask, c_masks], axis = -1) if q_mask is not None else None
    types   = tf.tile(tf.expand_dims(types, axis = 0), [q_batch_size, 1])

    return (memory, masks, types)

class MAGModelWrapper(tf.keras.Model):
    default_params  = HParamsMAGWrapper
    _attr_to_set    = [
        'subsample_at', 'subsample_after', 'subsampling_mode', 'subsampling_step',
        'subsampling_offset', 'max_types', 'random_training_type', 'repeat_pos_idx'
    ]

    def __init__(self, model, name = 'encoder', ** kwargs):
        super().__init__(name = name)
        self.hparams    = self.default_params.extract(kwargs)
        
        for config in self._attr_to_set:
            setattr(self, config, self.hparams[config])
        
        self.model  = model
        
        for config in self.model._attr_to_set:
            setattr(self, config, getattr(self.model, config))
        
        layer_idx = self.subsample_at
        if layer_idx < 0: layer_idx = len(self.model._layers) + layer_idx
        if self.subsample_after: layer_idx += 1
        self.M = max(0, min(len(self.model._layers), layer_idx))
        
        self.subsampling_layer  = None
        self.subsampling_drop_layer = tf.keras.layers.Dropout(
            self.hparams.subsampling_drop_rate
        ) if self.hparams.subsampling_drop_rate > 0 else None
        self.type_embedding_layer = None
        
        if self.subsampling_step > 1:
            self.subsampling_mode = get_enum_item(self.subsampling_mode, SubsamplingMode)
            
            if self.subsampling_mode == SubsamplingMode.CONV:
                self.subsampling_layer = tf.keras.layers.Conv1D(
                    filters = self.embedding_dim, kernel_size = self.subsampling_step,
                    strides = self.subsampling_step, padding = 'valid', name = 'subsampling_layer'
                )
            elif self.subsampling_mode == SubsamplingMode.SEPARABLE:
                self.subsampling_layer = tf.keras.layers.SeparableConv1D(
                    filters = self.embedding_dim, kernel_size = self.subsampling_step,
                    strides = self.subsampling_step, padding = 'valid', name = 'subsampling_layer'
                )
            elif self.subsampling_mode == SubsamplingMode.DENSE:
                self.subsampling_layer = tf.keras.layers.Dense(
                    units = self.embedding_dim, name = 'subsampling_layer',
                    kernel_initializer = self._mean_initializer
                )
        
        if self.hparams.use_type_embedding:
            self.type_embedding_layer = FasterEmbedding(
                self.max_types, self.embedding_dim, name = "type_embedding"
            )

    
    def _mean_initializer(self, shape, dtype = None):
        w = np.zeros(shape)
        for i in range(self.embedding_dim):
            w[i::self.embedding_dim, i] = 1
        w /= self.subsampling_step
        return tf.cast(w, dtype)

    def _build(self):
        self(self.dummy_inputs, training = False)

    @property
    def embedding_layers(self):
        return self.model._layers[: self.M]
    
    @property
    def memory_layers(self):
        return self.model._layers[self.M :]
    
    @property
    def dummy_inputs(self):
        batch_size, q_seq_len, c_seq_len = 2, 16, 32
        
        q_tokens    = tf.ones([batch_size, q_seq_len], dtype = tf.int32)
        q_length    = tf.fill([batch_size, 1], q_seq_len)
        
        c_tokens    = tf.ones([batch_size, c_seq_len], dtype = tf.int32)
        c_length    = tf.fill([batch_size, 1], c_seq_len)
        
        return [q_tokens, q_length, c_tokens, c_length]
    
    @timer
    def pad_to_multiple(self, output, mask = None):
        output = pad_to_multiple(output, self.subsampling_step, axis = 1)
        if mask is not None:
            mask = pad_to_multiple(mask, self.subsampling_step, axis = -1)
        
        return output, mask

    @timer
    def subsample(self, output, mask = None, training = False):
        if self.subsampling_step <= 1: return output, mask
        
        if self.subsampling_drop_layer is not None:
            output = self.subsampling_drop_layer(output, training = training)
        
        if self.subsampling_mode == SubsamplingMode.SELECT:
            indices = tf.range(self.subsampling_offset, tf.shape(output)[1], self.subsampling_step)
            indices = tf.tile(tf.expand_dims(indices, axis = 0), [tf.shape(output)[0], 1])

            output = tf.gather(output, indices, batch_dims = 1)

            if mask is not None:
                mask = tf.gather(tf.squeeze(mask, [1, 2]), indices, batch_dims = 1)
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1])
        elif self.subsampling_mode in (SubsamplingMode.CONV, SubsamplingMode.SEPARABLE):
            output = self.subsampling_layer(output, training = training)

            if mask is not None:
                indices = tf.range(0, tf.shape(output)[1]) * self.subsampling_step
                indices = tf.tile(tf.expand_dims(indices, axis = 0), [tf.shape(output)[0], 1])

                mask = tf.gather(tf.squeeze(mask, [1, 2]), indices, batch_dims = 1)
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1])
        elif self.subsampling_mode == SubsamplingMode.DENSE:
            output, mask = self.pad_to_multiple(output, mask)
            
            output = tf.reshape(
                output, [tf.shape(output)[0], -1, self.subsampling_step * tf.shape(output)[-1]]
            )
            output = self.subsampling_layer(output)
            
            if mask is not None:
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1, self.subsampling_step])
                mask = tf.reduce_min(mask, axis = -1)
        else:
            output, mask = self.pad_to_multiple(output, mask)
            
            output = tf.reshape(
                output, [tf.shape(output)[0], -1, self.subsampling_step, tf.shape(output)[-1]]
            )
            
            if mask is not None:
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1, self.subsampling_step])
                mask = tf.reduce_min(mask, axis = -1)
            
            if self.subsampling_mode == SubsamplingMode.MIN:
                output = tf.reduce_min(output, axis = 2)
            elif self.subsampling_mode == SubsamplingMode.MAX:
                output = tf.reduce_max(output, axis = 2)
            else:
                output = tf.reduce_mean(output, axis = 2)
        
        return output, mask

    @timer
    def embed_types(self, memory, types, training = False, debug = False, ** kwargs):
        if self.type_embedding_layer is None: return memory, types
        
        if self.max_types == 2:
            types = tf.cast(types > 0, tf.int32)
        elif self.random_training_type and training and tf.reduce_max(types) < self.max_types:
            random_offset = tf.random.uniform(
                (tf.shape(types)[0], 1),
                minval  = 0,
                maxval  = self.max_types - tf.reduce_max(types),
                dtype   = tf.int32
            )
            types = types + (random_offset * tf.cast(types > 0, tf.int32))
        
        if debug:
            tf.print("Types used :", types)
        
        memory = memory + self.type_embedding_layer(types)
        
        return memory, types
    
    @timer
    def embed(self,
              inputs,
              input_length  = None,
              token_types   = None,
              position_ids  = None,
              
              mask  = None,
              training  = False,
              padding_mask  = None,
              look_ahead_mask   = None,
              
              positional_offset = -1,
              force_not_subsampling = False,
              
              return_state       = False,
              return_attention   = False,
              return_hidden_states   = False,
              return_mask        = False,
              as_dict    = False,
              
              debug = False,
              ** kwargs
             ):
        if return_state is None:            return_state = self.model.return_state
        if return_attention is None:        return_attention = self.model.return_attention
        if return_hidden_states is None:    return_hidden_states = self.model.return_hidden_states
        if return_mask is None:             return_mask = self.model.return_mask
        
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) % 2 == 0
            
            if len(inputs) > 2:
                force_not_subsampling = tf.convert_to_tensor(force_not_subsampling)
                positional_offset     = tf.convert_to_tensor(positional_offset)
                
                if len(tf.shape(force_not_subsampling)) == 0:
                    force_not_subsampling = tf.repeat(
                        tf.expand_dims(force_not_subsampling, axis = 0), len(inputs) // 2
                    )
                if len(tf.shape(positional_offset)) == 0:
                    positional_offset = tf.repeat(
                        tf.expand_dims(positional_offset, axis = 0), len(inputs) // 2
                    )
                
                embeddings      = ()
                states          = () if return_state else None
                attn_weights    = () if return_attention else None
                hidden_states   = () if return_hidden_states else None
                masks           = () if return_mask else None
                
                for i in range(0, len(inputs), 2):
                    outputs_i = self.embed(
                        inputs[i],
                        input_length    = inputs[i+1],
                        token_types     = token_types[i // 2] if token_types is not None else None,
                        position_ids    = position_ids[i // 2] if position_ids is not None else None,
                        
                        training    = training,
                        
                        positional_offset   = positional_offset[i // 2],
                        force_not_subsampling   = force_not_subsampling[i // 2],
                        
                        return_state    = return_state,
                        return_attention    = return_attention,
                        return_hidden_states    = return_hidden_states,
                        return_mask = return_mask,
                        as_dict = True,
                        
                        debug   = debug,
                        ** kwargs
                    )
                    embeddings = embeddings + (outputs_i.output, )
                    if return_state:
                        states = states + (outputs_i.state, )
                    if return_attention:
                        attn_weights = attn_weights + (outputs_i.attention_weights, )
                    if return_hidden_states:
                        hidden_states = hidden_states + (outputs_i.hidden_states, )
                    if return_mask:
                        masks = masks + (outputs_i.mask, )
                
                return format_output(
                    output  = embeddings,
                    state   = states,
                    attn_weights    = attn_weights,
                    hidden_states   = hidden_states,
                    mask        = masks,
                    
                    return_state    = return_state,
                    return_attention    = return_attention,
                    return_hidden_states    = return_hidden_states,
                    return_mask = return_mask,
                    as_dict = as_dict
                )
            
            text, input_length = inputs
        else:
            text = inputs

        if debug:
            tf.print("Input tokens shape :", tf.shape(text), "-", input_length)
        
        batch_size = tf.shape(text)[0]
        n_doc_per_batch = -1
        if len(tf.shape(text)) == 3:
            n_doc_per_batch = tf.shape(text)[1]
            text            = tf.reshape(text, [-1, tf.shape(text)[-1]])
            input_length    = tf.reshape(input_length, [-1])
            if debug:
                tf.print("Input tokens reshaped shape :", tf.shape(text))
        
        if mask is None:
            mask = build_mask(
                text, self.use_causal_attention, input_length = input_length,
                look_ahead_mask = look_ahead_mask, padding_mask = padding_mask
            )

        outputs = self.model.call(
            text,
            input_length = input_length,
            token_types = token_types,
            position_ids    = position_ids,
            positional_offset   = positional_offset,
            
            mask    = mask,
            training    = training,
            
            last_layer_idx = self.M,
            
            return_state            = return_state,
            return_attention        = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = True,
            
            debug   = debug,
            ** kwargs
        )
        
        output, mask = outputs.output, outputs.mask

        if not force_not_subsampling:
            output, mask = self.subsample(output, mask = mask, training = training)

            if debug:
                tf.print("Output subsampled shape :", tf.shape(output))
        
        if n_doc_per_batch != -1:
            output  = tf.reshape(
                output, [batch_size, n_doc_per_batch, tf.shape(output)[1], tf.shape(output)[-1]]
            )
            mask    = tf.reshape(mask,   [batch_size, n_doc_per_batch, 1, 1, tf.shape(mask)[-1]])

            if debug:
                tf.print("Output reshaped shape :", tf.shape(output))
        
        return format_output(
            output,
            state   = outputs.state,
            attn_weights    = outputs.attention_weights,
            hidden_states   = outputs.hidden_states,
            mask    = mask,
            
            return_state        = return_state,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask         = return_mask,
            as_dict = as_dict
        )
    
    @timer
    def process_memory(self, embeddings, mask = None, training = False, ** kwargs):
        memory, mask, types = concat_embeddings(embeddings, mask = mask, training = training, ** kwargs)
        
        memory, types = self.embed_types(memory, types, training = training, ** kwargs)
        
        return self.model.call(
            memory, first_layer_idx = self.M, training = training, padding_mask = mask, ** kwargs
        )
    
    @timer
    def call(self,
             inputs,
             mask       = None,
             training   = False,
             
             merge_embeddings   = False,
             positional_offset  = -1, 
             
             return_state       = False,
             return_attention   = False,
             return_last_attention  = False,
             return_hidden_states   = False,
             return_mask        = False,
             as_dict    = False,
             
             ** kwargs
            ):
        if return_state is None:            return_state = self.model.return_state
        if return_attention is None:        return_attention = self.model.return_attention
        if return_hidden_states is None:    return_hidden_states = self.model.return_hidden_states
        if return_mask is None:             return_mask = self.model.return_mask

        memory_outputs = self.embed(
            inputs,
            mask    = mask,
            training    = training,
            positional_offset   = positional_offset,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = True,
            as_dict = True,
            ** kwargs
        )
        embeddings, masks = memory_outputs.output, memory_outputs.mask

        outputs = self.process_memory(
            embeddings,
            mask    = masks,
            training    = training,
            merge_embeddings    = merge_embeddings,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_last_attention   = return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = True,
            ** kwargs
        )
        
        return format_output(
            outputs.output,
            state   = (memory_outputs.state, outputs.state),
            attn_weights    = (memory_outputs.attention_weights, outputs.attention_weights),
            hidden_states   = (memory_outputs.hidden_states, outputs.hidden_states),
            mask    = outputs.mask,
            
            return_state    = return_state,
            return_attention    = return_attention or return_last_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = as_dict
        )
    
    def get_config(self):
        config = self.hparams.get_config()
        config['model'] = json.loads(self.model.to_json())
        return config

    def transfer_weights(self, * args, ** kwargs):
        self.model.transfer_weights(* args, ** kwargs)
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        config['model'] = model_from_json(
            json.dumps(config['model']), custom_objects = custom_objects
        )
        return cls(** config)

class MAGWrapper(tf.keras.Model):
    def __init__(self, model = None, ** kwargs):
        super().__init__(
            name = 'mag_{}'.format(model.name if model is not None else kwargs.get('name', 'wrapper'))
        )
        
        if model is None:
            from custom_architectures.transformers_arch.bart_arch import Bart
            kwargs.update(MAGWrapper.get_wrapper_kwargs())
            model = Bart(** kwargs)
        
        if not isinstance(model, Transformer):
            if not isinstance(model, MAGModelWrapper):
                model = MAGModelWrapper(model, ** kwargs)
        
        self.model = model
        
        for config in self.model._attr_to_set:
            setattr(self, config, getattr(self.model, config))
    
    @property
    def hparams(self):
        return self.model.hparams
    
    @property
    def dummy_inputs(self):
        return self.model.dummy_inputs
    
    @property
    def encoder(self):
        return self.model.encoder if isinstance(self.model, Transformer) else self.model
    
    @property
    def decoder(self):
        return self.model.decoder if isinstance(self.model, Transformer) else None
    
    def _build(self):
        self(self.dummy_inputs, training = False)

    def call(self, * args, ** kwargs):
        return self.model.call(* args, ** kwargs)
    
    def infer(self, * args, ** kwargs):
        return self.model.infer(* args, ** kwargs)
    
    def get_config(self):
        return {'model' : json.loads(self.model.to_json())}
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        if 'model' in config:
            from custom_architectures import get_architecture
            
            class_name  = config['model']['class_name']
            class_conf  = config['model']['config']
            class_conf.update(MAGWrapper.get_wrapper_kwargs())

            config['model'] = get_architecture(class_name, ** class_conf)
            
        return cls(** config)

    @classmethod
    def from_pretrained(cls, pretrained_name, * args, ** kwargs):
        from custom_architectures.transformers_arch import get_pretrained_transformer
        
        kwargs.update(MAGWrapper.get_wrapper_kwargs())
        return cls(get_pretrained_transformer(
            pretrained_name, * args, ** kwargs
        ))
    
    @staticmethod
    def get_wrapper_kwargs():
        return {
            'encoder_wrapper'   : lambda encoder, ** kwargs: MAGModelWrapper(model = encoder, ** kwargs),
            'encoder_wrapper_params'    : HParamsMAGWrapper
        }
        
custom_objects  = {
    'MAG'   : MAGWrapper,
    'MAGWrapper'    : MAGWrapper,
    'MAGModelWrapper'   : MAGModelWrapper
}

custom_functions    = custom_objects


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

import logging
import collections
import tensorflow as tf

from loggers import timer
from utils import get_object
from custom_layers import log_softmax
from utils.text import create_padding_mask
from custom_architectures.transformers_arch.transformer_arch import format_output

TransformerInferenceOutput = collections.namedtuple(
    "TransformerInferenceOutput", [
        "tokens",
        "lengths",
        "output",
        "score",
        "attention_weights"
    ]
)

TransformerInferenceState   = collections.namedtuple(
    "TransformerInferenceState", [
        "tokens",
        "input_length",
        "padding_mask",
        "finished",
        "logits",
        "scores",
        "attention_weights"
    ]
)

def get_shape_invariant(model, encoder_output = None, return_attention = None, use_cache = False,
                        ** kwargs):
    logits_shape, attn_shapes = model.get_output_shape(
        (None, None),
        encoder_output      = encoder_output.shape if encoder_output is not None else None,
        return_state        = use_cache,
        return_attention    = return_attention,
        return_last_attention   = True
    )
    attn_shapes = {
        k : tf.TensorSpec(shape = v, dtype = tf.float32) for k, v in attn_shapes.items()
    }
    
    return TransformerInferenceState(
        tokens          = tf.TensorSpec(shape = (None, None),       dtype = tf.int32),
        input_length    = tf.TensorSpec(shape = (None, 1),          dtype = tf.int32),
        padding_mask    = tf.TensorSpec(shape = (None, 1, 1, None), dtype = tf.float32),
        finished        = tf.TensorSpec(shape = (None, ),           dtype = tf.int32),
        logits          = tf.TensorSpec(shape = logits_shape,       dtype = tf.float32),
        scores          = tf.TensorSpec(shape = (None, ),           dtype = tf.float32),
        attention_weights   = attn_shapes
    )

@timer
def infer(model, * args, method = 'greedy', ** kwargs):
    return get_object(
        _inference_methods, method, model, * args, ** kwargs
    )

@tf.function(experimental_relax_shapes = True)
def _infer(self,
           tokens    = None,
           input_length  = None,
           encoder_output    = None,
           initial_state     = None,
           prefix       = None,
           
           enc_padding_mask  = None,
           padding_mask  = None,
           training  = False,
           use_cache = False,
           
           batch_size   = -1,
           max_length   = -1,
           use_sampling = False,
           early_stopping    = True,

           return_state       = False,
           return_attention   = False,
           return_last_attention    = False,
           return_hidden_states   = False,
           return_mask        = False,
           as_dict    = True,

           ** kwargs
          ):
    def cond(tokens, input_length, padding_mask, finished, logits, scores, attn_weights):
        return not (early_stopping and tf.reduce_sum(finished) == batch_size)
    
    def body(tokens, input_length, padding_mask, finished, logits, scores, attn_weights):
        outputs = self(
            tokens,
            input_length    = input_length,
            encoder_output  = encoder_output,
            prefix  = prefix if not use_cache or tf.shape(tokens)[1] == 1 else None,
            padding_mask    = padding_mask,
            enc_padding_mask    = enc_padding_mask,
            positional_offset   = -1 if not use_cache else input_length - 1 + model.positional_offset,
            training    = training,
            
            apply_softmax   = False,
            
            return_state    = use_cache,
            return_attention    = return_attention,
            return_last_attention   = True,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = True,
            
            ** kwargs
        )
        logits      = log_softmax(outputs.output[:, -1, :], axis = -1)
        
        next_token  = tf.cast(_select_next_token(
            logits, n = 1, previous = tokens, use_sampling = use_sampling
        ), tf.int32)

        scores = scores + tf.cast(1 - finished, logits.dtype) * tf.gather(logits, next_token, batch_dims = 1)

        tokens = tf.concat([
            tokens, tf.expand_dims(next_token, axis = 1)
        ], axis = -1)

        finished    = tf.maximum(
            finished, tf.cast(tf.math.equal(next_token, self.eos_token), finished.dtype)
        )

        input_length += tf.expand_dims(1 - finished, axis = 1)
        padding_mask = tf.concat([
            padding_mask,
            tf.reshape(tf.cast(finished, padding_mask.dtype), [-1, 1, 1, 1])
        ], axis = -1)
        
        return TransformerInferenceState(
            tokens  = tokens,
            input_length    = input_length,
            padding_mask    = padding_mask,
            finished    = finished,
            logits      = outputs.output,
            scores      = scores,
            attention_weights   = outputs.attention_weights if not skip_attention else attn_weights
        )
    
    if max_length < 0: max_length = self.max_input_length
    
    skip_attention = not (return_attention or return_last_attention)
    
    if batch_size == -1:
        assert encoder_output is not None or tokens is not None or prefix is not None
        if tokens is not None:
            batch_size = tf.shape(tokens)[0]
        elif encoder_output is not None:
            batch_size = tf.shape(encoder_output)[0]
        else:
            batch_size = tf.shape(prefix)[0]
    
    if tokens is None:
        tokens          = tf.fill((batch_size, 1), self.sos_token)
        input_length    = tf.fill((batch_size, 1), 1)
    elif isinstance(tokens, (list, tuple)):
        tokens, input_length    = tokens
    
    if input_length is None:
        input_length    = tf.fill((batch_size,), tf.shape(tokens)[1])
    
    if padding_mask is None:
        padding_mask    = create_padding_mask(tokens, seq_len = input_length, dtype = tf.float32)

    if prefix is not None and tf.shape(padding_mask)[-1] == tf.shape(tokens)[1]:
        padding_mask = tf.concat([
            tf.zeros((batch_size, 1, 1, tf.shape(prefix)[1]), dtype = tf.float32), padding_mask
        ], axis = -1)
        
    shapes_invariant    = get_shape_invariant(
        self, encoder_output = encoder_output, return_attention = return_attention
    )
    outputs = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = TransformerInferenceState(
            tokens  = tokens,
            input_length    = input_length,
            padding_mask    = padding_mask,
            finished    = tf.zeros((batch_size,), dtype = tf.int32),
            logits      = tf.zeros((batch_size, 1, shapes_invariant.logits.shape[-1])),
            scores      = tf.zeros((batch_size, )),
            attention_weights   = {
                k : tf.zeros([d if d is not None else 1 for d in shape.shape])
                for k, shape in shapes_invariant.attention_weights.items()
            }
        ),
        shape_invariants    = shapes_invariant,
        maximum_iterations  = max_length
    ))
    
    return TransformerInferenceOutput(
        tokens  = outputs.tokens[:, 1:],
        lengths = tf.squeeze(outputs.input_length, axis = 1) - 1,
        score   = outputs.scores,
        output  = outputs.logits,
        attention_weights   = outputs.attention_weights if not skip_attention else None
    )

@tf.function(experimental_relax_shapes = True)
def _infer_beam_search(self,
                       tokens    = None,
                       input_length  = None,
                       encoder_output    = None,
                       initial_state     = None,
                       prefix       = None,

                       num_beams    = 10,
                       num_sentences    = 5,

                       temperature  = 1.,
                       length_temperature   = 0.,
                       length_power = 0.,

                       enc_padding_mask  = None,
                       padding_mask  = None,
                       training  = False,
                       use_cache = False,

                       batch_size   = -1,
                       max_length   = -1,
                       use_sampling = False,
                       early_stopping    = True,

                       return_state       = False,
                       return_attention   = False,
                       return_last_attention    = False,
                       return_hidden_states   = False,
                       return_mask        = False,
                       as_dict    = True,

                       ** kwargs
                      ):
    def cond(tokens, input_length, padding_mask, finished, logits, scores, attn_weights):
        if not early_stopping: return True

        finished_per_batch = tf.reshape(finished, [batch_size, -1])
        return tf.reduce_sum(finished_per_batch[:,:num_sentences]) != batch_size * num_sentences
    
    def body(tokens, input_length, padding_mask, finished, logits, scores, attn_weights):
        outputs = self(
            tokens,
            input_length    = input_length,
            encoder_output  = encoder_output,
            prefix  = prefix if not use_cache or tf.shape(tokens)[1] == 1 else None,
            padding_mask    = padding_mask,
            enc_padding_mask    = enc_padding_mask,
            positional_offset   = -1 if not use_cache else input_length - 1 + model.positional_offset,
            training    = training,
            apply_softmax   = False,
            
            return_state    = use_cache,
            return_attention    = return_attention,
            return_last_attention   = True,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = True,
            
            ** kwargs
        )
        not_finish_mask = tf.expand_dims(tf.cast(1 - finished, tf.float32), axis = 1)

        logits  = outputs.output[:, -1, :]
        if temperature != 0.:
            logits = logits / temperature
        
        if length_temperature != 0.:
            _lengths = tf.cast(input_length + 1, tf.float32)
            if length_temperature == -1.:
                temp = tf.maximum(tf.math.log(_lengths), 1.)
            else:
                temp = _lengths ** length_temperature

            logits = logits * temp
        
        logits = logits * not_finish_mask + (1 - not_finish_mask) * eos_mask

        beam_scores = log_softmax(logits, axis = -1) + tf.expand_dims(scores, axis = 1)
        
        if tf.shape(tokens)[1] == 1:
            mask = tf.expand_dims(
                tf.cast(tf.tile(tf.range(num_beams), [batch_size]) == 0, tf.float32), axis = -1
            )
            beam_scores = beam_scores * mask + (1 - mask) * -1e4

        reshaped_scores = beam_scores / (tf.cast(input_length, tf.float32) ** length_power)
        reshaped_scores = tf.reshape(reshaped_scores, [batch_size, -1])

        next_token  = _select_next_token(
            reshaped_scores, n = num_beams, previous = None, use_sampling = use_sampling
        )
        next_token  = tf.reshape(
            tf.cast(next_token, tokens.dtype), [effective_batch_size]
        )
        
        token_batch_idx = next_token // self.vocab_size + batch_idx_add
        next_token      = next_token % self.vocab_size
        
        beam_scores     = tf.gather(beam_scores,    token_batch_idx)
        input_length    = tf.gather(input_length,   token_batch_idx)
        finished        = tf.gather(finished,       token_batch_idx)
        padding_mask    = tf.gather(padding_mask,   token_batch_idx)
        if return_attention:
            attention_weights   = {
                k : tf.gather(attn, token_batch_idx) for k, attn in outputs.attention_weights.items()
            }
        else:
            attention_weights   = outputs.attention_weights
        
        scores = tf.gather(beam_scores, next_token, batch_dims = 1)

        tokens  = tf.concat([
            tf.gather(tokens, token_batch_idx),
            tf.expand_dims(next_token, axis = 1)
        ], axis = -1)

        finished    = tf.maximum(
            finished, tf.cast(tf.math.equal(next_token, self.eos_token), tf.int32)
        )
        
        input_length += 1 - tf.expand_dims(finished, axis = 1)
        padding_mask = tf.concat([
            padding_mask,
            tf.reshape(tf.cast(finished, padding_mask.dtype), [-1, 1, 1, 1])
        ], axis = -1)
        
        return TransformerInferenceState(
            tokens  = tokens,
            input_length    = input_length,
            padding_mask    = padding_mask,
            finished    = finished,
            logits      = outputs.output,
            scores      = scores,
            attention_weights   = attention_weights if not skip_attention else attn_weights
        )

    if max_length == -1: max_length = self.max_input_length
    
    skip_attention  = not return_attention and not return_last_attention
    
    temperature     = tf.cast(temperature, tf.float32)
    length_power    = tf.cast(length_power, tf.float32)
    length_temperature  = tf.cast(length_temperature, tf.float32)
    
    if batch_size == -1:
        assert encoder_output is not None or tokens is not None or prefix is not None
        if tokens is not None:
            batch_size = tf.shape(tokens)[0]
        elif encoder_output is not None:
            batch_size = tf.shape(encoder_output)[0]
        else:
            batch_size = tf.shape(prefix)[0]
    
    if encoder_output is not None:
        encoder_output  = tf.repeat(encoder_output, num_beams, axis = 0)
        if enc_padding_mask is not None:
            enc_padding_mask  = tf.repeat(enc_padding_mask, num_beams, axis = 0)

    if prefix is not None:
        prefix = tf.repeat(prefix, num_beams, axis = 0)
    
    effective_batch_size    = batch_size * num_beams

    if tokens is None:
        tokens = tf.fill((effective_batch_size, 1), self.sos_token)
        input_length = tf.fill((effective_batch_size, 1), 1)
    else:
        tokens = tf.repeat(tokens, num_beams, axis = 0)
        if input_length is None:
            input_length = tf.fill((effective_batch_size,), tf.shape(tokens)[1])
        else:
            input_length = tf.repeat(input_length, num_beams, axis = 0)
    
    if padding_mask is None:
        padding_mask    = create_padding_mask(tokens, seq_len = input_length, dtype = tf.float32)
    
    if prefix is not None and tf.shape(padding_mask)[-1] == tf.shape(tokens)[1]:
        padding_mask = tf.concat([
            tf.zeros((effective_batch_size, 1, 1, tf.shape(prefix)[1]), dtype = tf.float32),
            padding_mask
        ], axis = -1)

    batch_idx_add   = tf.repeat(tf.range(batch_size), num_beams, axis = 0) * num_beams
    eos_mask        = tf.expand_dims(tf.cast(tf.range(self.vocab_size) != self.eos_token, tf.float32), axis = 0) * -1e4

    shapes_invariant    = get_shape_invariant(
        self, encoder_output = encoder_output, return_attention = return_attention
    )
    outputs = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = TransformerInferenceState(
            tokens  = tokens,
            input_length    = input_length,
            padding_mask    = padding_mask,
            finished    = tf.zeros((effective_batch_size,), dtype = tf.int32),
            logits      = tf.zeros((effective_batch_size, 1, shapes_invariant.logits.shape[-1])),
            scores      = tf.zeros((effective_batch_size, )),
            attention_weights   = {
                k : tf.zeros([d if d is not None else 1 for d in shape.shape])
                for k, shape in shapes_invariant.attention_weights.items()
            }
        ),
        shape_invariants    = shapes_invariant,
        maximum_iterations  = max_length
    ))
    
    scores  = outputs.scores
    if length_power != 0:
        lengths = tf.cast(tf.squeeze(outputs.input_length, axis = 1), tf.float32)
        scores  = outputs.scores / (lengths ** length_power)
    
    attn_weights = None
    if not skip_attention:
        attn_weights = {
            k : tf.reshape(attn, [
                batch_size, num_beams, tf.shape(attn)[1], tf.shape(attn)[2], tf.shape(attn)[3]
            ])[:, :num_sentences]
            for k, attn in outputs.attention_weights.items()
        }
    
    return TransformerInferenceOutput(
        tokens  = tf.reshape(outputs.tokens, [batch_size, num_beams, -1])[:, :num_sentences, 1:],
        lengths = tf.reshape(outputs.input_length, [batch_size, num_beams])[:, :num_sentences] - 1,
        score   = tf.reshape(scores,         [batch_size, num_beams])[:, :num_sentences],
        output  = tf.reshape(outputs.logits, [batch_size, num_beams, tf.shape(outputs.logits)[1], -1])[:, :num_sentences],
        attention_weights   = attn_weights
    )

def _select_next_token(scores, n = 1, previous = None, use_sampling = False):
    if not use_sampling:
        if n == 1: return tf.argmax(scores, axis = -1)
        return tf.nn.top_k(scores, k = n).indices
    
    raise NotImplementedError()
    
_inference_methods  = {
    'greedy'    : _infer,
    'sample'    : lambda * args, ** kwargs: _infer(* args, use_sampling = True, ** kwargs),
    'beam'      : _infer_beam_search
}
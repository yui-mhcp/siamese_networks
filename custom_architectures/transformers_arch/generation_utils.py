import logging
import collections
import tensorflow as tf

from loggers import timer
from utils import get_object
from utils.text import create_padding_mask
from custom_architectures.transformers_arch.transformer_arch import format_output

TransformerInferenceOutput = collections.namedtuple(
    "TransformerInferenceOutput", [
        "tokens",
        "output",
        "score"
    ]
)

@timer
def infer(model,
          * args,
          method = 'greedy',
          
          max_length    = None,

          return_state       = None,
          return_attention   = None,
          return_hidden_states   = None,
          return_mask        = None,
          as_dict    = False,

          ** kwargs
         ):
    if max_length is None:              max_length = model.max_input_length
    if return_state is None:            return_state = model.return_state
    if return_attention is None:        return_attention = model.return_attention
    if return_hidden_states is None:    return_hidden_states = model.return_hidden_states
    if return_mask is None:             return_mask = model.return_mask

    return get_object(
        _inference_methods,
        method,
        * args,
            
        model = model,
        max_length      = max_length,
        return_state    = return_state,
        return_attention    = return_attention,
        return_hidden_states    = return_hidden_states,
        return_mask         = return_mask,
        as_dict     = as_dict,
        
        ** kwargs
    )

def _infer(model,
           tokens    = None,
           input_length  = None,
           encoder_output    = None,
           initial_state     = None,
          
           enc_padding_mask  = None,
           padding_mask  = None,
           training  = False,
           use_cache = False,
           
           sos_token    = None,
           eos_token    = None,
           max_length   = None,
           use_sampling = False,
           early_stopping    = True,

           return_state       = None,
           return_attention   = None,
           return_hidden_states   = None,
           return_mask        = None,
           as_dict    = False,

           ** kwargs
          ):
    batch_size  = 1
    if encoder_output is not None:
        batch_size = tf.shape(encoder_output)[0]
    
    if tokens is None:
        tokens          = tf.fill((batch_size, 1), sos_token)
        input_length    = tf.fill((batch_size, 1), 1)
    elif isinstance(tokens, (list, tuple)):
        tokens, input_length    = tokens
    
    if batch_size == 1: batch_size = tf.shape(tokens)[0]
    
    if input_length is None:
        input_length    = tf.fill((batch_size,), tf.shape(tokens)[1])
    
    if padding_mask is None:
        padding_mask    = create_padding_mask(tokens, seq_len = input_length, dtype = tf.float32)

    finished    = tf.zeros((batch_size,), dtype = tf.int32)

    decoded_tokens  = None
    decoder_outputs = None
    
    while decoder_outputs is None or tf.shape(decoder_outputs)[1] < max_length:
        outputs = model(
            tokens,
            input_length    = input_length,
            encoder_output  = encoder_output,
            initial_state   = initial_state,
            padding_mask    = padding_mask,
            enc_padding_mask    = enc_padding_mask,
            positional_offset   = -1 if not use_cache else input_length - 1 + model.positional_offset,
            training    = training,
            
            return_state    = use_cache,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = True,
            
            ** kwargs
        )
        scores, initial_state  = outputs.output, outputs.state
        
        if use_cache and initial_state is None:
            logging.warning('use_cache is set to True but the model did not return `state`')
            use_cache = False
        
        next_token  = _select_next_token(
            scores[:,-1,:], n = 1, previous = decoded_tokens, use_sampling = use_sampling
        )
        next_token  = tf.reshape(
            tf.cast(next_token, tokens.dtype), [batch_size, 1]
        )

        if use_cache:
            tokens  = next_token
        else:
            tokens  = tf.concat([tokens, next_token], axis = -1)
        
        if decoded_tokens is None:
            decoded_tokens  = next_token
        else:
            decoded_tokens  = tf.concat([decoded_tokens, next_token], axis = -1)
        if use_cache and decoder_outputs is not None:
            decoder_outputs = tf.concat([decoder_outputs, scores], axis = 1)
        else:
            decoder_outputs = scores

        finished    = tf.maximum(
            finished, tf.cast(tf.math.equal(next_token[:,0], eos_token), tf.int32)
        )
        if early_stopping and tf.reduce_sum(finished) == batch_size:
            break
        
        input_length += 1 - finished
        padding_mask = tf.concat([
            padding_mask,
            tf.reshape(tf.cast(finished, padding_mask.dtype), [-1, 1, 1, 1])
        ], axis = -1)
    
    return TransformerInferenceOutput(
        tokens  = decoded_tokens,
        score   = _score_output(decoder_outputs, decoded_tokens),
        output  = format_output(
            decoder_outputs,
            state   = outputs.state,
            attn_weights    = outputs.attention_weights,
            hidden_states   = outputs.hidden_states,
            mask    = outputs.mask,

            return_state        = return_state,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask         = return_mask,
            as_dict = as_dict
        )
    )

def _score_output(probs, indices):
    return tf.reduce_mean(
        tf.math.log(tf.gather(probs, indices, batch_dims = 2, axis = -1)), axis = -1
    )

def _select_next_token(scores, n = 1, previous = None, use_sampling = False):
    if not use_sampling:
        if n == 1: return tf.argmax(scores, axis = -1)
        return tf.nn.top_k(scores, k = n)
    
    raise NotImplementedError()

_inference_methods  = {
    'greedy'    : _infer,
    'sample'    : lambda * args, ** kwargs: _infer(* args, use_sampling = True, ** kwargs)
}
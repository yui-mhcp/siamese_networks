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
""" TF 2.0 BERT model. """

import tensorflow as tf

from hparams.hparams import HParams
from custom_layers import get_activation
from utils.text import create_padding_mask
from custom_architectures.current_blocks import _get_layer, _get_pooling_layer
from custom_architectures.transformers_arch.embedding_head import EmbeddingHead, HParamsEmbeddingHead
from custom_architectures.transformers_arch.transformer_arch import TransformerEncoder, HParamsTransformerEncoder

HParamsTransformerTokenEmbedding = HParams(
    vocab_size      = -1,
    embedding_dim   = -1,
    type_vocab_size = 2,
    max_position_embeddings = 512,
    drop_rate = 0.1,
    epsilon   = 1e-6,
    norm_training   = True
)

HParamsBaseBERT = HParamsTransformerEncoder(
    ** HParamsTransformerTokenEmbedding,
    use_pooling = True
)

HParamsBertMLM  = HParamsBaseBERT(
    transform_activation    = None,
    epsilon   = 1e-6
)

HParamsBertClassifier   = HParamsBaseBERT(
    num_classes = -1,
    process_tokens  = False,
    process_first_token = False,
    final_drop_rate = 0.1,
    final_activation    = None,
    classifier_type     = 'dense',
    classifier_kwargs   = {}
)

HParamsBertEmbedding   = HParamsBaseBERT(
    ** HParamsEmbeddingHead, process_tokens = True
)

class BertTokenEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, vocab_size, embedding_dim, name = None, ** kwargs):
        super().__init__(name = name)
        self.hparams = HParamsTransformerTokenEmbedding.extract(kwargs)
        self.hparams = self.hparams(vocab_size = vocab_size, embedding_dim = embedding_dim)

        self.add_layer  = tf.keras.layers.Add()
        self.norm   = tf.keras.layers.LayerNormalization(
            epsilon = self.hparams.epsilon, name = "layer_norm"
        )
        self.dropout    = tf.keras.layers.Dropout(self.hparams.drop_rate)

    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name    = "weight",
                shape   = [self.hparams.vocab_size, self.hparams.embedding_dim]
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name    = "pos_embeddings",
                shape   = [self.hparams.max_position_embeddings, self.hparams.embedding_dim]
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name    = "token_type_embeddings",
                shape   = [self.hparams.type_vocab_size, self.hparams.embedding_dim]
            )

        super().build(input_shape)

    def call(self, token_ids, position_ids = None, token_type_ids = None, training = False):
        """
        Applies embedding based on inputs tensor.
        Returns:
            final_embeddings (:obj:`tf.Tensor`): output embedding tensor.
        """
        if token_type_ids is None:
            token_type_ids = tf.fill([tf.shape(token_ids)[0], tf.shape(token_ids)[1]], value = 0)

        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(0, tf.shape(token_ids)[1]), axis = 0)

        embedded_inputs     = tf.gather(self.weight, token_ids)
        embedded_pos        = tf.gather(self.position_embeddings, position_ids)
        embedded_pos        = tf.tile(embedded_pos, (tf.shape(embedded_inputs)[0], 1, 1))
        embedded_token_type = tf.gather(self.token_type_embeddings, token_type_ids)
        
        final_embeddings = self.add_layer([embedded_inputs, embedded_pos, embedded_token_type])
        final_embeddings = self.norm(final_embeddings, training = training and self.hparams.norm_training)
        final_embeddings = self.dropout(final_embeddings, training =training)

        return final_embeddings

    def get_config(self):
        config = super().get_config()
        return (self.hparams + config).get_config()
    
class BertPooler(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, name = None, ** kwargs):
        super().__init__(name = name)
        self.embedding_dim = embedding_dim
        
        self.dense = tf.keras.layers.Dense(embedding_dim, activation = "tanh", name = "dense")

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)

        return pooled_output

    def get_config(self):
        config = super().get_config()
        config['embedding_dim'] = self.embedding_dim
        return config


class BaseBERT(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, name = None, ** kwargs):
        super().__init__(name = name)

        self.hparams = HParamsBaseBERT.extract(kwargs)
        self.hparams = self.hparams(vocab_size = vocab_size, embedding_dim = embedding_dim)
        
        self.vocab_size     = vocab_size
        self.embedding_dim  = embedding_dim

        self.embeddings = BertTokenEmbeddings(** self.hparams, name = "embeddings")
        self.encoder    = TransformerEncoder(** self.hparams, name = "encoder")
        self.pooler     = BertPooler(** self.hparams, name = "pooler") if self.hparams.use_pooling else None
        
    def _build(self):
        batch_size, seq_len = 2, 32
        text = tf.ones([batch_size, seq_len], dtype = tf.int32)
        text_length = tf.fill([batch_size, 1], seq_len)
        
        self([text, text_length], training = False)
    
    def freeze(self, trainable = False):
        self.embeddings.trainable = trainable 
        self.encoder.trainable = trainable 
        self.pooler.trainable = trainable
        
    def process_inputs(self, inputs, mask = None, training = False):
        return inputs
    
    def process_outputs(self, encoder_outputs, pooled_outputs, mask = None, training = False):
        return (encoder_outputs, pooled_outputs)
        
    def call(self, inputs, mask = None, training = False, return_attention = None):
        """
            Perform BERT inference
            
            Arguments :
                - inputs    : list of length 2 or 3
                    text        : tokens with shape [batch_size, seq_len]
                    text_length : text length with shape [batch_size, 1]
                    token_type  : (Optional) type of token with shape [batch_size, seq_len]
                - mask      : padding mask (if not provided, built based on `text_length`)
                - training  : bool, whether it is training phase or not 
                - return_attention  : whether to return attention_scores for the TransformerEncoder
            Return : (output, pooled_output, attn_weights) if return_attention else output
                - output    : Encoder output with shape [batch_size, seq_len, embedding_dim]
                - attn_weights  : dict of Encoder attention weights
                    Each value corresponds to attention of a given layer with shape [batch_size, num_heads, seq_len, seq_len]
                
        """
        if return_attention is None: return_attention = self.hparams.return_attention
        
        inputs = self.process_inputs(inputs, mask = mask, training = training)
        
        text, text_lengths = inputs[:2]
        
        batch_size  = tf.shape(text)[0]
        seq_len     = tf.shape(text)[1]
        
        token_type = inputs[2] if len(inputs) == 3 else tf.fill([batch_size, seq_len], 0)
        
        if mask is None:
            mask = create_padding_mask(text, seq_len = text_lengths)
        
        mask = tf.reshape(tf.cast(mask, tf.float32), [batch_size, 1, 1, seq_len])

        embedding_output = self.embeddings(
            text, position_ids = None, token_type_ids = token_type, training = training
        )
        
        encoder_outputs, attn_weights = self.encoder(
            embedding_output, mask = mask, training = training, return_attention = True
        )

        pooled_output = self.pooler(encoder_outputs) if self.pooler is not None else None

        outputs = self.process_outputs(
            encoder_outputs, pooled_output, mask = mask, training = training
        )
        
        return (outputs, attn_weights) if return_attention else outputs

    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

    @classmethod
    def from_pretrained(cls, pretrained_name, pretrained_task = 'base', pretrained = None, ** kwargs):
        from models.weights_converter import partial_transfer_learning
        
        if pretrained is None:
            with tf.device('cpu') as d:
                pretrained = transformers_bert(pretrained_name, pretrained_task)

        config = HParamsBaseBERT(
            vocab_size      = pretrained.config.vocab_size,
            type_vocab_size = pretrained.config.type_vocab_size,
            max_position_embeddings = pretrained.config.max_position_embeddings,
            
            num_layers      = pretrained.config.num_hidden_layers,
            embedding_dim   = pretrained.config.hidden_size,
            drop_rate       = pretrained.config.hidden_dropout_prob,
            epsilon         = pretrained.config.layer_norm_eps,
            
            mha_num_heads   = pretrained.config.num_attention_heads,
            mha_mask_factor = -10000,
            mha_drop_rate   = pretrained.config.hidden_dropout_prob,
            mha_epsilon     = pretrained.config.layer_norm_eps,
            
            ffn_dim                 = pretrained.config.intermediate_size,
            ffn_activation          = pretrained.config.hidden_act,
            transform_activation    = pretrained.config.hidden_act
        )
        
        instance = cls(** config(** kwargs))
        instance._build()
        
        weights = pretrained.get_weights()
        weights[1], weights[2] = weights[2], weights[1]
        if pretrained_task in ('lm', 'mlm'): weights.append(weights.pop(-5))
        
        partial_transfer_learning(instance, weights)
        
        return instance

class BertMLM(BaseBERT):
    def __init__(self, vocab_size, embedding_dim, ** kwargs):
        kwargs['use_pooling'] = False
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
        
        self.hparams = HParamsBertMLM.extract(kwargs)
        self.hparams = self.hparams(vocab_size = vocab_size, embedding_dim = embedding_dim)
        
        self.dense  = tf.keras.layers.Dense(embedding_dim, name = "dense_transform")
        self.act    = get_activation(self.hparams.transform_activation)
        self.norm   = tf.keras.layers.LayerNormalization(epsilon = self.hparams.epsilon)
        
    def build(self, input_shape):
        self.bias = self.add_weight(
            shape = [self.hparams.vocab_size], initializer = "zeros", trainable = True, name = "bias"
        )
        
    def process_outputs(self, encoder_outputs, pooled_outputs, mask = None, training = False):
        outputs = self.dense(encoder_outputs, training = training)
        if self.act is not None: outputs = self.act(outputs)
        outputs = self.norm(outputs, training = training and self.hparams.norm_training)
        
        outputs = tf.reshape(outputs, [-1, self.embedding_dim])
        outputs = tf.matmul(outputs, self.embeddings.weight, transpose_b = True)
        outputs = tf.reshape(outputs, [tf.shape(encoder_outputs)[0], -1, self.vocab_size])
        outputs = outputs + self.bias
        
        return outputs

class BertClassifier(BaseBERT):
    def __init__(self, num_classes, vocab_size, embedding_dim, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
        
        self.hparams = HParamsBertClassifier.extract(kwargs)
        self.hparams = self.hparams(
            vocab_size = vocab_size, num_classes = num_classes, embedding_dim = embedding_dim
        )
        
        self.classifier = _get_layer(
            self.hparams.classifier_type, num_classes, name = 'classifier', ** self.hparams.classifier_kwargs
        )
            
        self.act        = get_activation(self.hparams.final_activation)
        self.dropout    = tf.keras.layers.Dropout(self.hparams.final_drop_rate)
        
    def process_outputs(self, encoder_outputs, pooled_outputs, mask = None, training = False):
        if self.hparams.process_tokens:
            output = encoder_outputs if not self.hparams.process_first_token else encoder_outputs[:, 0, :]
        else:
            output = pooled_outputs
        
        if self.dropout is not None: output = self.dropout(output, training = training)
        output = self.classifier(output, training = training)
        if self.act is not None: output = self.act(output)
            
        return output

class BertNSP(BertClassifier):
    def __init__(self, vocab_size, embedding_dim, num_classes = 2, process_tokens = False, ** kwargs):
        kwargs.update({'use_pooling' : True, 'process_tokens' : False})
        super().__init__(
            num_classes = num_classes, vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
    
class BertEmbedding(BaseBERT):
    def __init__(self, output_dim, vocab_size, embedding_dim, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
        
        self.hparams = HParamsBertEmbedding.extract(kwargs)
        self.hparams = self.hparams(
            vocab_size = vocab_size, output_dim = output_dim, embedding_dim = embedding_dim
        )
        
        self.embedding_head = EmbeddingHead(** self.hparams)

    def process_outputs(self, encoder_outputs, pooled_outputs, mask = None, training = False):
        output = encoder_outputs if self.hparams.process_tokens else pooled_outputs
        
        return self.embedding_head(output, mask = mask, training = training)

class BertQA(BertClassifier):
    def __init__(self, ** kwargs):
        kwargs.update({'num_classes' : 2, 'process_tokens' : True, 'process_first_token' : False})
        super().__init__(** kwargs)
    
    def process_outputs(self, encoder_outputs, pooled_outputs, mask = None, training = False):
        output = super().process_outputs(encoder_outputs, pooled_outputs, mask = None, training = False)

        start_prob  = tf.nn.softmax(output[:, :, 0], axis = -1)
        end_prob    = tf.nn.softmax(output[:, :, 1], axis = -1)
        
        return start_prob, end_prob

class DPR(BertEmbedding):
    @classmethod
    def from_pretrained(cls, pretrained_name, pretrained_task = 'question_encoder',
                        pretrained = None, ** kwargs):
        if pretrained is None:
            with tf.device('cpu') as d:
                pretrained = transformers_bert(pretrained_name, pretrained_task)
            
        kwargs.setdefault('output_dim', pretrained.config.projection_dim)
        kwargs.update({
            'process_tokens'        : True,
            'process_first_token'   : True,
            'hidden_layer_type' : 'dense',
            'final_pooling'     : None,
            'use_final_dense'   : False
        })
        
        return BertEmbedding.from_pretrained(
            pretrained_name = pretrained_name,
            pretrained_task = pretrained_task,
            pretrained  = pretrained,
            ** kwargs
        )
    
def transformers_bert(name, task = 'base'):
    #tf.config.set_visible_devices([], 'GPU')
    import transformers
    if task == 'base':
        return transformers.TFBertModel.from_pretrained(name)
    elif task == 'question_encoder':
        return transformers.TFDPRQuestionEncoder.from_pretrained(name)
    elif task in ('lm', 'mlm'):
        return transformers.TFBertForMaskedLM.from_pretrained(name)
    elif task == 'nsp':
        return transformers.TFBertForNextSentencePrediction.from_pretrained(name)
    else:
        raise ValueError("Unknown task !\n  Accepted : {}\n  Got : {}".format(
            tuple(_transformers_pretrained_task.keys()), task
        ))

custom_functions    = {
    'transformers_bert' : transformers_bert,
    
    'BaseBERT'  : BaseBERT,
    'base_bert' : BaseBERT,
    'BertMLM'   : BertMLM,
    'bert_mlm'  : BertMLM,
    'BertNSP'   : BertNSP,
    'bert_nsp'  : BertNSP,
    'BertEmbedding' : BertEmbedding,
    'BertQA'    : BertQA,
    'bert_qa'   : BertQA,
    'DPR'       : DPR
}

custom_objects  = {
    'TransformerEncoder'    : TransformerEncoder,
    
    'BertTokenEmbeddings'   : BertTokenEmbeddings,
    'BertPooler'        : BertPooler,
    'BaseBERT'          : BaseBERT,
    'BertMLM'           : BertMLM,
    'BertNSP'           : BertNSP,
    'BertEmbedding'     : BertEmbedding,
    'BertQA'            : BertQA,
    'DPR'               : DPR
}

_encoders   = {'Bert' : BertEmbedding, 'DPR' : DPR}
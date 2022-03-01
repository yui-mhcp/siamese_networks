
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

""" TF 2.0 BERT model, compatible with the `transformers`' implementation """

import tensorflow as tf

from custom_layers import get_activation
from custom_architectures.current_blocks import _get_layer, _get_pooling_layer
from custom_architectures.transformers_arch.embedding_head import EmbeddingHead, HParamsEmbeddingHead
from custom_architectures.transformers_arch.text_transformer_arch import TextTransformerEncoder, HParamsTextTransformerEncoder

HParamsBaseBERT = HParamsTextTransformerEncoder(
    use_pooling = True,
    epsilon     = 1e-6
)

HParamsBertMLM  = HParamsBaseBERT(
    transform_activation    = None
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
    
class BertPooler(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, name = None, ** kwargs):
        super().__init__(name = name)
        self.embedding_dim = embedding_dim
        
        self.dense = tf.keras.layers.Dense(embedding_dim, activation = "tanh", name = "dense")

    def call(self, output):
        return self.dense(output[:, 0])

    def get_config(self):
        config = super().get_config()
        config['embedding_dim'] = self.embedding_dim
        return config

class BaseBERT(TextTransformerEncoder):
    default_params = HParamsBaseBERT
    
    def __init__(self, vocab_size, embedding_dim, ** kwargs):
        super().__init__(vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs)

        self.pooler = BertPooler(** self.hparams, name = "pooler") if self.hparams.use_pooling else None
        
    def process_outputs(self, encoder_outputs, pooled_outputs, mask = None, training = False):
        return (encoder_outputs, pooled_outputs)
    
    def compute_output(self, output, training = False, mask = None, ** kwargs):
        pooled_output = self.pooler(output) if self.pooler is not None else None

        return self.process_outputs(
            output, pooled_output, mask = mask, training = training
        )

    @classmethod
    def from_pretrained(cls,
                        pretrained_name,
                        pretrained_task = 'base',
                        pretrained      = None,
                        ** kwargs
                       ):
        from models.weights_converter import partial_transfer_learning
        
        if pretrained is None:
            with tf.device('cpu') as d:
                pretrained = transformers_bert(pretrained_name, pretrained_task)

        config = HParamsBaseBERT(
            vocab_size      = pretrained.config.vocab_size,
            max_token_types = pretrained.config.type_vocab_size,
            max_input_length    = pretrained.config.max_position_embeddings,
            
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
        if pretrained_task in ('lm', 'mlm'): weights.append(weights.pop(-5))
        
        partial_transfer_learning(instance, weights)
        
        return instance

class BertMLM(BaseBERT):
    default_params = HParamsBertMLM

    def __init__(self, vocab_size, embedding_dim, ** kwargs):
        kwargs['use_pooling'] = False
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
        
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
        
        outputs = self.embeddings.linear(outputs) + self.bias
        
        return outputs

class BertClassifier(BaseBERT):
    default_params  = HParamsBertClassifier
    _attr_to_set    = BaseBERT._attr_to_set + ['process_tokens']
    
    def __init__(self, num_classes, vocab_size, embedding_dim, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
        
        self.classifier = _get_layer(
            self.hparams.classifier_type, num_classes, name = 'classifier', ** self.hparams.classifier_kwargs
        )
            
        self.act        = get_activation(self.hparams.final_activation)
        self.dropout    = tf.keras.layers.Dropout(self.hparams.final_drop_rate)
        
    def process_outputs(self, encoder_outputs, pooled_outputs, mask = None, training = False):
        if self.process_tokens:
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
    default_params  = HParamsBertEmbedding
    _attr_to_set    = BaseBERT._attr_to_set + ['process_tokens']

    def __init__(self, output_dim, vocab_size, embedding_dim, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, output_dim = output_dim, ** kwargs
        )

        self.embedding_head = EmbeddingHead(** self.hparams)

    def process_outputs(self, encoder_outputs, pooled_outputs, mask = None, training = False):
        output = encoder_outputs if self.process_tokens else pooled_outputs
        
        return self.embedding_head(output, mask = mask, training = training)

class BertQA(BertClassifier):
    def __init__(self, * args, ** kwargs):
        kwargs.update({'num_classes' : 2, 'process_tokens' : True, 'process_first_token' : False})
        super().__init__(* args, ** kwargs)
    
    def process_outputs(self, encoder_outputs, pooled_outputs, mask = None, training = False):
        output = super().process_outputs(encoder_outputs, pooled_outputs, mask = None, training = False)

        probs   = tf.nn.softmax(output, axis = -1)
        
        return probs[:, :, 0], probs[:, :, 1]

class DPR(BertEmbedding):
    @classmethod
    def from_pretrained(cls,
                        pretrained_name,
                        pretrained_task = 'question_encoder',
                        pretrained      = None,
                        ** kwargs
                       ):
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
    'TextTransformerEncoder'  : TextTransformerEncoder,
    
    'BertPooler'        : BertPooler,
    'BaseBERT'          : BaseBERT,
    'BertMLM'           : BertMLM,
    'BertNSP'           : BertNSP,
    'BertEmbedding'     : BertEmbedding,
    'BertQA'            : BertQA,
    'DPR'               : DPR
}

_encoders   = {'Bert' : BertEmbedding, 'DPR' : DPR}

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

import os
import re
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from custom_layers import SimilarityLayer
from models.interfaces import BaseModel
from utils.distance import distance, KNN
from utils.thread_utils import Pipeline
from utils.embeddings import _embedding_filename, _default_embedding_ext, load_embedding
from utils import plot_embedding, pad_batch

logger  = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

_ab_pattern = re.compile('(^(a|b)_|_(a|b)_|_(a|b)$)')
_default_attr_cache = {attr : attr for attr in ['a_renaming', 'b_renaming']}

def _rename_attribute(text, a_renaming, b_renaming):
    return text.replace('a', a_renaming).replace('b', b_renaming)

class BaseComparator(BaseModel):
    """
        Base class for Comparators architecture 
        
        The concept of `Comparator` is similar to `Siamese Networks` except that there are 2 encoders that encode 2 different elements and try to encode 2-related elements the same way. 
        
        For instance, it can be used for : 
        - `context retrieval`   : the encoder A encodes a `question` while the encoder B encodes a `context` and the target is to encode the context containing the answer the closest to the question. It is the base concept of the `Dense Passage Retrieval (DPR)` architecture
        - `text 2 image` (or `image 2 text`)    : encode the textual representation (caption) the closest to the corresponding image. It is the concept of `CLIP`
        
        /!\ WARNING /!\ The main difference compared to `Siamese` is that the inputs are **not** interchangeable. 
        - In `Siamese`, the encoder is the same for input A and B so giving input A (left) and B (right) is the same as giving B (left) and A (right) inputs. 
        - In `Comparator`, encoders A and B are different so you **must** pass inputs A and B in the right order. 
        
        The output of a `Comparator` is the output of a Dense with 1 neuron and a sigmoid function. 
        - It means that if `embed_distance == True`, an output of 0.75 can be interpreted as "75% that the 2 inputs are different (unrelated)"
        - Otherwise if `embed_distance == False`, an output of 0.75 can beinterpreted as "75% that the 2 inputs are similar (have a close relation)"
        
        You must define functions : 
            - build_encoder_{a / b}(** kwargs)  : builds the encoder model A / B
            - get_input_{a / b}(data)       : load a single data input for encoder A / B (for dataset pipeline) \*
            - augment_input_{a / b}(data)   : augment a single input data (A or B)
            - preprocess_input_{a / b}(inputs)  : apply a preprocessing on a batch of inputs
        
        To simplify the usage of `Comparator`, you can define `a_renaming` and `b_renaming` to set an alias for functions' renaming.
        For instance in `CLIP`, `a_renaming = 'image'` and `b_renaming = 'text'`. Therefore you can define `get_input_image` which will be interpreted as `get_input_a` by the processing pipeline : it is a syntactic sugar to have more meaningful function's names.
        
        \* `data` will be the output of the original dataset without processing. In `Siamese`, suffixes _x and _y were removed to determine input A and B but in `Comparator` you have to determine which one will be for A or B.
        
        The dataset pipeline is as follow :
            1) encode_data receives a 2-tuple (same, not_same)
                Where both are dict (output of the original dataset) and will be pass (separately) to `get_input_a` and `get_input_b`
            2) augment_data(same, not_same) : receives input A / B (non-batched) once at a time so that you can augment them in independant way
                They will be passed respectively to `augment_input_a` and `autment_input_b`
            
            3) preprocess_data(same, not_same)  : receives batched datas for same and not same. They are then concatenated to form a single batch of same and not_same
                Note that if the 1st dimension mismatch (variable length data), they are padded to match the longest one
    """
    
    a_renaming  = 'a'
    b_renaming  = 'b'
    
    def __init__(self,
                 distance_metric    = 'euclidian',
                 embed_distance     = True,
                 threshold          = 0.5,
                 ** kwargs
                ):
        """
            Constructor for a base Comparator Network

            Arguments :
                - distance_metric   : the distance function to compute between the 2 embeddings
                - embed_distance    : whether to embed distance or similarity
                    If True     : the higher the output, the higher the distance
                    If False    : the higher the output, the higher the similarity
                - threshold     : thereshold for the decision boundary (currently only 0.5 is correctly handled in loss / metrics)
        """
        self.threshold      = threshold
        self.embed_distance     = embed_distance
        self.distance_metric    = distance_metric
        
        super(BaseComparator, self).__init__(**kwargs)

    def build_encoder_a(self, ** kwargs):
        """ Return a `tf.keras.Model` model which is the encoder A of the Comparator network  """
        raise NotImplementedError("You must define the `build_encoder_a` method !")
    
    def build_encoder_b(self, ** kwargs):
        """ Return a `tf.keras.Model` model which is the encoder B of the Comparator network  """
        raise NotImplementedError("You must define the `build_encoder_b` method !")

    def get_input_a(self, data):
        """
            Process `data` to return a single output for the encoder A
            
            `data` is basically a dict or a pd.Series but can take every type of value you want
        """
        raise NotImplementedError("You must define the `get_input_a(data)` method !")

    def get_input_b(self, data):
        """
            Process `data` to return a single output for the encoder B
            
            `data` is basically a dict or a pd.Series but can take every type of value you want
        """
        raise NotImplementedError("You must define the `get_input_b(data)` method !")
        
    def augment_input_a(self, inp):
        """ Augment a single processed input (from get_input_a()) """
        return inp
    
    def augment_input_b(self, inp):
        """ Augment a single processed input (from get_input_b()) """
        return inp
    
    def preprocess_input_a(self, inputs):
        """
        Preprocess a batch of inputs A (if you need to do processing on the whole batch)
        """
        return inputs
    
    def preprocess_input_b(self, inputs):
        """
        Preprocess a batch of inputs B (if you need to do processing on the whole batch)
        """
        return inputs
    
    def _build_model(self, normalize = True, ** kwargs):
        """ Build the `comparator` architecture with self.build_encoder_{a/b}() as encoders """
        def get_signature(encoder, is_encoder_a):
            if isinstance(encoder, tf.keras.Sequential):
                signature = encoder.input_shape[1:]
            else:
                signature = self.encoder_a_input_signature if is_encoder_a else self.encoder_b_input_signature
                
            return signature
        
        encoder_a = self.build_encoder_a(normalize = normalize, ** kwargs)
        encoder_b = self.build_encoder_b(normalize = normalize, ** kwargs)
        
        input_kwargs    = {
            'input_signature_a' : get_signature(encoder_a, True),
            'input_signature_b' : get_signature(encoder_b, False)
        }
        
        comparator_config = {
            'architecture_name' : 'comparator',
            'encoder_a'     : encoder_a,
            'encoder_b'     : encoder_b,
            'distance_metric'   : self.distance_metric,
            'normalize'     : normalize,
            ** input_kwargs
        }
                
        super()._build_model(comparator = comparator_config)
    
    @property
    def encoder_a_input_shape(self):
        raise NotImplementedError()
    
    @property
    def encoder_b_input_shape(self):
        raise NotImplementedError()

    @property
    def encoder_a_input_signature(self):
        return tf.TensorSpec(shape = self.encoder_a_input_shape, dtype = tf.float32)
    
    @property
    def encoder_b_input_signature(self):
        return tf.TensorSpec(shape = self.encoder_b_input_shape, dtype = tf.float32)

    @property
    def embedding_dim(self):
        for model in [self.encoder_a, self.encoder_b]:
            if isinstance(model.output_shape, tuple) and isinstance(model.output_shape[1], int):
                return model.output_shape[1]
        raise ValueError('Unable to determine embedding_dim')
    
    @property
    def input_signature(self):
        return (self.encoder_a_input_signature, self.encoder_b_input_signature)
    
    @property
    def output_signature(self):
        return tf.TensorSpec(shape = (None,), dtype = tf.int32)
    
    @property
    def nb_inputs(self):
        signature_a = self.encoder_a_input_signature
        signature_b = self.encoder_b_input_signature

        n1 = 1 if not isinstance(signature_a, (list, tuple)) else len(signature_a)
        n2 = 1 if not isinstance(signature_b, (list, tuple)) else len(signature_b)
        return n1 + n2

    @property
    def encoder_a(self):
        return self.comparator.layers[-3]
    
    @property
    def encoder_b(self):
        return self.comparator.layers[-2]

    @property
    def similarity_layer(self):
        return self.comparator.layers[-1]
    
    @property
    def decoder(self):
        if isinstance(self.comparator.layers[-1], SimilarityLayer):
            return self.comparator.layers[-1]
        
        n = self.nb_inputs
        
        inputs = [
            tf.keras.layers.Input(shape = (None, self.embedding_dim)),
            tf.keras.layers.Input(shape = (None, self.embedding_dim))
        ]
        out = inputs
        for l in self.comparator.layers[n + 2:]:
            out = l(out)
        return tf.keras.Model(inputs, out, name = 'comparator_decoder')
    
    def __getattribute__(self, name):
        if name == '_attr_cache': return object.__getattribute__(self, name)
        if not hasattr(self, '_attr_cache'): self._attr_cache = _default_attr_cache.copy()
        
        if name not in self._attr_cache:
            renamed = re.sub(
                _ab_pattern,
                lambda m: _rename_attribute(m.group(0), self.a_renaming, self.b_renaming),
                name
            )
            if renamed != name and not hasattr(self, renamed): renamed = name
            self._attr_cache[name] = renamed
        return object.__getattribute__(self, self._attr_cache[name])
    
    def __str__(self):
        des = super().__str__()
        des += "- Embedding dim   : {}\n".format(self.embedding_dim)
        des += "- Distance metric : {}\n".format(self.distance_metric)
        return des
                
    def compile(self, loss = 'binary_crossentropy', metrics = ['binary_accuracy', 'eer'], ** kwargs):
        super().compile(loss = loss, metrics = metrics, ** kwargs)
        
    def decode_output(self, output):
        """
            Return whether the 2 inputs can be considered as the same class based on the output score 
            
            This function returns whether the inputs can be considered as same independently of `self.embed_distance`
        """
        return output < self.threshold if self.embed_distance else output > self.threshold
    
    def distance(self, embedded_1, embedded_2, pred_probability = False, ** kwargs):
        """ Return distance between embeddings (based on self.distance_metric) """
        return self.similarity_layer(
            [embedded_1, embedded_2], pred_probability = pred_probability, ** kwargs
        )
    
    def get_output(self, data):
        if 'same' in data: same = data['same']
        elif 'id' in data: same = 1
        else: same = data['id_x'] == data['id_y']
        
        return tf.cast(same, tf.int32)
        
    def encode(self, data):
        """
            Call self.get_input() on normalized version of data (by removing the _x / _y suffix)
            This function process separately `same` and `not_same`
        """
        inp_a   = self.get_input_a(data)
        inp_b   = self.get_input_b(data)
        
        same    = self.get_output(data)
        
        return (inp_a, inp_b), same
    
    def augment(self, data):
        """ Augment `same` or `not_same` separately """
        (inp_a, inp_b), target = data
        
        inp_a = self.augment_input_a(inp_a)
        inp_b = self.augment_input_b(inp_b)
        
        return (inp_a, inp_b), target
    
    def concat(self, x_same, x_not_same):
        """
            Concat both batched `x_same` and `x_not_same` together (same function called for the y) 
        """
        return tf.concat([x_same, x_not_same], axis = 0)

    def encode_data(self, same, not_same):
        """ Apply `self.encode()` on same and not_same separately """
        return self.encode(same), self.encode(not_same)
    
    def augment_data(self, same, not_same):
        """ Apply `self.augment()` on same and not_same separately """
        return self.augment(same), self.augment(not_same)
    
    def preprocess_data(self, same, not_same):
        """ 
            oncat `x` and `y` from same and not_same and call `self.preprocess_input` on the batched result
            
            In theory it should also pad inputs but it is not working (quite strange...)
            As a solution, I created the `self.concat(x, y)` method so that you can pad as you want (can be useful if you have special padding_values)
        """
        (same_a, same_b), target_same = same
        (not_same_a, not_same_b), target_not_same = not_same
        
        inp_a = self.preprocess_input_a(self.concat(same_a, not_same_a))
        inp_b = self.preprocess_input_b(self.concat(same_b, not_same_b))
        
        target = tf.concat([target_same, target_not_same], axis = 0)
        if self.embed_distance: target = 1 - target
        
        return (inp_a, inp_b), target
        
    def get_dataset_config(self, ** kwargs):
        """ Add default configuration for siamese dataset """
        kwargs['siamese'] = True
        kwargs['batch_before_map']  = True
        
        return super().get_dataset_config(** kwargs)
        
    def _get_train_config(self, * args, test_size = 1, test_batch_size = 32, ** kwargs):
        """
        Set new default test_batch_size to embed 128 data (32 same + 32 not-same pairs) 
        """
        return super()._get_train_config(
            * args, test_size = test_size, test_batch_size = test_batch_size, ** kwargs
        )
        
    def predict_with_target(self, batch, step, prefix, directory = None, **kwargs):
        """
            Embed the x / y in batch and plot their embeddings 
            This function should be improved to add labels information but as ids are not in batch, I do not know how to add information of similarity in the plots...
        """
        if directory is None: directory = self.train_test_dir
        else: os.makedirs(directory, exist_ok = True)
        kwargs.setdefault('show', False)
        
        (inp_a, inp_b), _ = batch
        
        embedded_a = self.encoder_a(inp_a, training = False)
        embedded_b = self.encoder_b(inp_b, training = False)
        embedded = tf.concat([embedded_a, embedded_b], axis = 0)
        
        ids = list(range(len(embedded_a))) * 2
        
        title       = 'embedding space (step {})'.format(step)
        filename    = os.path.join(directory, prefix + '.png')
        plot_embedding(
            embedded, filename = filename, title = title, ids = ids, ** kwargs
        )
    
    @timer
    def embed(self, data, use_encoder_a, batch_size = 128, tqdm = lambda x: x, ** kwargs):
        """
            Embed a list of data
            
            Pipeline : 
                1) Call self.get_input(data) to have encoded data
                2) Take a batch of `batch_size` inputs
                3) Call pad_batch(batch) to have pad it (if necessary)
                4) Call self.preprocess_input(batch) to apply a preprocessing (if needed)
                5) Pass the processed batch to self.encoder
                6) Concat all produced embeddings to return [len(data), self.embedding_dim] matrix
            
            This function is the core of the `siamese networks` as embeddings are used for everything (predict similarity / distance), label predictions, clustering, make funny colored plots, ...
        """
        suffix = 'a' if use_encoder_a else 'b'
        
        time_logger.start_timer('processing')
        if not isinstance(data, (list, tuple, pd.DataFrame)): data = [data]
        
        inputs = getattr(self, 'get_input_{}'.format(suffix))(data, ** kwargs)

        time_logger.stop_timer('processing')
        
        encoder = getattr(self, 'encoder_{}'.format(suffix))
        
        embedded = []
        for idx in tqdm(range(0, len(inputs), batch_size)):
            time_logger.start_timer('processing')

            batch = inputs[idx : idx + batch_size]
            batch = pad_batch(batch) if not isinstance(batch[0], (list, tuple)) else [pad_batch(b) for b in zip(* batch)]
            batch = getattr(self, 'preprocess_input_{}'.format(suffix))(batch)
            
            time_logger.stop_timer('processing')
            time_logger.start_timer('encoding')

            embedded_batch = encoder(batch, training = False)
            if not isinstance(embedded_batch, tf.Tensor): embedded_batch = embedded_batch[0]
            
            time_logger.stop_timer('encoding')

            embedded.append(embedded_batch)

        return tf.concat(embedded, axis = 0)
    
    def embed_a(self, data, ** kwargs):
        return self.embed(data, use_encoder_a = True, ** kwargs)
    
    def embed_b(self, data, ** kwargs):
        return self.embed(data, use_encoder_a = False, ** kwargs)
    
    def pred_similarity(self, x, y, decoder = None, ** kwargs):
        """
            Return a similarity score between x and y
            Arguments : 
                - x : a single embedding vector
                - y : a single / matrix of embedding vector(s)
                - decoder   : decoder to use (basically `self.decoder` but used to not build it for every `x`)
        """
        if decoder is None: decoder = self.decoder
        
        scores = decoder([x, y], ** kwargs)
        if self.embed_distance: scores = 1. - scores
        
        return scores
    
    def pred_distance(self, x, y, decoder = None, ** kwargs):
        """
            Return a score of distance for all pairs
            The result is symetric matrix [n,n] where the element [i,j] is the disimilarity probability between i-th and j-th embeddings
        """
        return 1. - self.pred_similarity(x, y, decoder, ** kwargs)
    
    def pred_similarity_matrix(self, x, y, decoder = None, ** kwargs):
        """
            Return a score of distance for all pairs
            The result is symetric matrix [n,n] where the element [i,j] is the similarity probability between i-th and j-th embeddings
        """
        return self.pred_similarity(x, y, decoder, pred_matrix = True, ** kwargs)
    
    def pred_distance_matrix(self, x, y, decoder = None, ** kwargs):
        """
            Return a score of distance for all pairs
            The result is symetric matrix [n,n] where the element [i,j] is the disimilarity probability between i-th and j-th embeddings
        """
        return 1. - self.pred_similarity_matrix(x, y, decoder, ** kwargs)
    
    def embed_dataset(self, directory, dataset, use_encoder_a, embedding_name = None, ** kwargs):
        """
            Calls `self.predict` and save the result to `{directory}/embeddings/{embedding_name}` (`embedding_name = self.nom` by default)
        """
        if not directory.endswith('embeddings'): directory = os.path.join(directory, 'embeddings')
        
        return self.predict(
            dataset,
            use_encoder_a   = use_encoder_a,
            save    = True,
            directory   = directory,
            embedding_name  = embedding_name if embedding_name else self.nom,
            ** kwargs
        )
    
    def embed_dataset_a(self, * args, ** kwargs):
        return self.embed_dataset(* args, use_encoder_a = True, ** kwargs)
    
    def embed_dataset_b(self, * args, ** kwargs):
        return self.embed_dataset(* args, use_encoder_a = False, ** kwargs)

    def get_pipeline(self,
                     use_encoder_a,
                     
                     id_key = 'filename',
                     batch_size = 1,
                     
                     add_ab_prefix  = True,
                     
                     save   = True,
                     directory  = None,
                     embedding_name = _embedding_filename,
                     ** kwargs
                    ):
        @timer
        def preprocess(row, ** kw):
            inputs = getattr(self, 'get_input_{}'.format(suffix))(row)
            if not isinstance(row, (dict, pd.Series)): row = {}
            row['processed'] = inputs
            return row
        
        @timer
        def inference(inputs, ** kw):
            batch_inputs = inputs if isinstance(inputs, list) else [inputs]
            
            batch = [inp.pop('processed') for inp in batch_inputs]
            batch = pad_batch(batch) if not isinstance(batch[0], (list, tuple)) else [
                pad_batch(b) for b in zip(* batch)
            ]
            batch = getattr(self, 'preprocess_input_{}'.format(suffix))(batch)
            
            embeddings = encoder(batch, training = False)
            
            for row, embedding in zip(batch_inputs, embeddings): row[embedding_key] = embedding
            
            return inputs
        
        if save:
            embedding_file = embedding_name
            if '{}' in embedding_file: embedding_file = embedding_file.format(self.embedding_dim)
            if not os.path.splitext(embedding_file)[1]: embedding_file += _default_embedding_ext
            if directory is None: directory = self.pred_dir
            filename = os.path.join(directory, embedding_file)
        
        suffix = 'a' if use_encoder_a else 'b'

        encoder = self.encoder_a if use_encoder_a else use_encoder_b

        embedding_key = 'embedding'
        if add_ab_prefix:
            embedding_key = '{}_{}'.format(
                self.a_renaming if use_encoder_a else self.b_renaming, embedding_key
            )

        pipeline = Pipeline(** {
            ** kwargs,
            'filename'  : None if not save else filename,
            'id_key'    : id_key,
            'save_keys' : ['id', embedding_key],
            'as_list'   : True,
            
            'tasks'     : [
                preprocess,
                {'consumer' : inference, 'batch_size' : batch_size, 'allow_multithread' : False}
            ]
        })
        pipeline.start()
        return pipeline
    
    @timer
    def predict(self, data, use_encoder_a, ** kwargs):
        pipeline = self.get_pipeline(use_encoder_a = use_encoder_a, ** kwargs)
        
        return pipeline.extend_and_wait(data, ** kwargs)

    def predict_a(self, data, ** kwargs):
        return self.predict(data, use_encoder_a = True, ** kwargs)

    def predict_b(self, data, ** kwargs):
        return self.predict(data, use_encoder_a = False, ** kwargs)

    def get_config(self, *args, ** kwargs):
        """ Return base configuration for a `siamese network` """
        config = super().get_config(* args, ** kwargs)
        config.update({
            'threshold' : self.threshold,
            'embed_distance'    : self.embed_distance,
            'distance_metric'   : self.distance_metric
        })
        
        return config
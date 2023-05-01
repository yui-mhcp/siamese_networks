
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

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils import shuffle as sklearn_shuffle

from custom_train_objects.generators.file_cache_generator import FileCacheGenerator

class SiameseGenerator(FileCacheGenerator):
    def __init__(self,
                 # General informations
                 dataset,
                 # Column for merging / loading
                 id_column  = 'id',
                 file_column    = 'filename',
                 processed_column   = '',

                 suffixes       = ('_x', '_y'), 
                 
                 ** kwargs
                ):
        self.suffixes   = suffixes
        
        super().__init__(
            dataset,
            id_column   = id_column,
            file_column = file_column,
            processed_column    = processed_column,
            ** kwargs
        )
        
        self.id_col_x   = self.id_column + self.suffixes[0]
        self.id_col_y   = self.id_column + self.suffixes[1]
        
        self.file_col_x = self.file_column + self.suffixes[0]
        self.file_col_y = self.file_column + self.suffixes[1]
        
        self.processed_col_x = self.processed_column + self.suffixes[0]
        self.processed_col_y = self.processed_column + self.suffixes[1]
        
    @property
    def processed_output_shape(self):
        raise NotImplementedError
    
    def load_file(self, filename):
        raise NotImplementedError

    def build(self, ** kwargs):
        from datasets.dataset_utils import build_siamese_dataset
        
        kwargs.setdefault('random_state', self.random_state)
        self.same, self.not_same = build_siamese_dataset(
            self.dataset,
            column      = self.id_column,
            suffixes    = self.suffixes,
            as_tf_dataset   = False,
            shuffle     = True,
            ** kwargs
        )
        
        self.ids = self.same[self.id_column].unique()
    
    @property
    def all_ids(self):
        return np.unique(np.concatenate([
            self.same[self.id_column].values, 
            self.not_same[self.id_col_x].values, 
            self.not_same[self.id_col_y].values
        ]))
        
    @property
    def all_files(self):
        return np.concatenate([
            self.same[self.file_col_x].values, 
            self.same[self.file_col_y].values,
            self.not_same[self.file_col_x].values, 
            self.not_same[self.file_col_y].values
        ])
            
    @property
    def output_signature(self):
        signatures  = {
            self.processed_column   : tf.TensorSpec(
                shape = self.processed_output_shape, dtype = tf.float32
            ),
            self.file_column    : tf.TensorSpec(shape = (), dtype = tf.string),
        }
        same_sign   = {self.id_column : tf.TensorSpec(shape = (), dtype = tf.string)}
        not_same_sign = {
            self.id_column + suffix : tf.TensorSpec(shape = (), dtype = tf.string)
            for suffix in self.suffixes
        }
        for k, sign in signatures.items():
            for suffix in self.suffixes:
                same_sign[k + suffix] = sign
                not_same_sign[k + suffix] = sign
        
        return (same_sign, not_same_sign)
    
    def __str__(self):
        des = super().__str__()
        des += "- Same dataset length     : {}\n".format(len(self.same))
        des += "- Not same dataset length : {}\n".format(len(self.not_same))
        return des
    
    def __len__(self):
        return max(len(self.same), len(self.not_same))
    
    def __getitem__(self, idx):
        if idx == 0 and self.shuffle:
            self.same       = sklearn_shuffle(self.same)
            self.not_same   = sklearn_shuffle(self.not_same)
        return self.get_same(idx), self.get_not_same(idx)
    
    def get_same(self, idx):
        if idx > len(self.same): idx = idx % len(self.same)
        row = self.same.loc[idx]
        
        return {
            self.id_column      : row[self.id_column],
            self.file_col_x     : row[self.file_col_x],
            self.file_col_y     : row[self.file_col_y],
            self.processed_col_x    : self.load(row[self.file_col_x]),
            self.processed_col_y    : self.load(row[self.file_col_y])
        }
    
    def get_not_same(self, idx):
        if idx > len(self.not_same): idx = idx % len(self.not_same)
        row = self.not_same.loc[idx]
        
        return {
            self.id_col_x       : row[self.id_col_x],
            self.id_col_y       : row[self.id_col_y],
            self.file_col_x     : row[self.file_col_x],
            self.file_col_y     : row[self.file_col_y],
            self.processed_col_x    : self.load(row[self.file_col_x]),
            self.processed_col_y    : self.load(row[self.file_col_y])
        }
    
    def sample(self, n = 1, ids = None, random_state = None, ** kwargs):
        if not random_state: random_state = self.random_state
        if ids is None: 
            return self.dataset.sample(n, random_state = random_state, ** kwargs)
        
        if not isinstance(ids, (tuple, list, np.ndarray)): ids = [ids]
            
        samples = []
        for speaker_id in ids:
            subset = self.dataset[self.dataset[self.id_column] == speaker_id]
            
            sample_id = subset.sample(
                min(len(subset), n), random_state = random_state, ** kwargs
            )
            samples.append(sample_id)
        
        return pd.concat(samples)


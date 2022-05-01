
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

from tqdm import tqdm
from sklearn.utils import shuffle as sklearn_shuffle

from datasets.dataset_utils import prepare_dataset, build_siamese_dataset

class SiameseGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 # General informations
                 dataset,
                 # Cache parameters
                 min_apparition = 3,
                 cache_size     = 30000,
                 preload        = False,
                 # Column for merging / loading
                 id_column  = 'id',
                 file_column    = 'filename',
                 processed_column   = '',
                 # additional informations
                 shuffle        = False,
                 suffixes       = ('_x', '_y'), 
                 random_state   = 10,
                 
                 ** kwargs
                ):
        assert isinstance(dataset, pd.DataFrame)
        self.dataset    = dataset
        
        self.shuffle        = shuffle
        self.suffixes       = suffixes
        self.random_state   = random_state
        
        self.cache      = {}
        self.cache_size = cache_size
        self.min_apparition = min_apparition
        
        self.id_column  = id_column
        self.id_col_x   = self.id_column + self.suffixes[0]
        self.id_col_y   = self.id_column + self.suffixes[1]
        
        self.file_column   = file_column
        self.file_col_x = self.file_column + self.suffixes[0]
        self.file_col_y = self.file_column + self.suffixes[1]
        
        self.processed_column   = processed_column
        self.processed_col_x = self.processed_column + self.suffixes[0]
        self.processed_col_y = self.processed_column + self.suffixes[1]
        
        self.save   = None
        self.not_same   = None
        self.unique_files   = None
        self.frequencies    = None
        self.files_to_cache = None
        
        self.build_datasets(** kwargs)
        self.build_cache(cache_size, min_apparition, preload)
    
    @property
    def processed_output_shape(self):
        raise NotImplementedError
    
    def load_file(self, filename):
        raise NotImplementedError

    def build_datasets(self, ** kwargs):
        kwargs.setdefault('random_state', self.random_state)
        self.same, self.not_same = build_siamese_dataset(
            self.dataset,
            column      = self.id_column,
            suffixes    = self.suffixes,
            as_tf_dataset   = False,
            shuffle     = True,
            ** kwargs
        )
        
        self.unique_files, self.frequencies = self.get_uniques()
    
    def get_uniques(self):
        uniques = {}
        for file in self.all_files:
            uniques.setdefault(file, 0)
            uniques[file] += 1
            
        return np.array(list(uniques.keys())), np.array(list(uniques.values()))
        
    def build_cache(self, size, min_apparition = 2, preload = False):
        # compute 'size' most present files (in 2 ds)
        cache_idx   = np.flip(np.argsort(self.frequencies))[:size]
        to_cache    = self.unique_files[cache_idx]
        # get files with at least 'min_apparition' apparition
        freq        = self.frequencies[cache_idx]
        to_cache    = to_cache[np.where(freq >= min_apparition)]
        
        self.files_to_cache = to_cache
        
        if preload:
            self.load_cache()
        
        return to_cache
    
    def load_cache(self, tqdm = tqdm, ** kwargs):
        self.cache = {f : self.cache[f] for f in self.unique_files if f in self.cache}
        
        not_cached = np.array([f for f in self.files_to_cache if f not in self.cache])
        ds = prepare_dataset(
            pd.DataFrame([{'filename' : f} for f in not_cached]),
            batch_size  = 0,                     
            map_fn      = self.load_file,
            prefetch    = True,
            cache       = False,
            ** kwargs
        )
        for filename, processed in tqdm(zip(not_cached, ds), total = len(not_cached)):
            self.cache[filename] = processed
    
    @property
    def ids(self):
        return self.same[self.id_column].unique()
    
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
    def output_types(self):
        types = {
            self.processed_column   : tf.float32,
            self.file_column    : tf.string,
        }
        same_types = {self.id_column : tf.string}
        not_same_types = {
            self.id_column + suffix : tf.string for suffix in self.suffixes
        }
        for k, t in types.items():
            for suffix in self.suffixes:
                same_types[k + suffix] = t
                not_same_types[k + suffix] = t
        
        return (same_types, not_same_types)
    
    @property
    def output_shapes(self):
        shapes = {
            self.processed_column   : self.processed_output_shape,
            self.file_column    : [],
        }
        same_shapes = {self.id_column : []}
        not_same_shapes = {
            self.id_column + suffix : [] for suffix in self.suffixes
        }
        for k, t in shapes.items():
            for suffix in self.suffixes:
                same_shapes[k + suffix] = t
                not_same_shapes[k + suffix] = t
        
        return (same_shapes, not_same_shapes)
    
    def __str__(self):
        des = "Siamese Generator :\n"
        des += "- Unique ids : {}\n".format(len(self.ids))
        des += "- Same dataset length     : {}\n".format(len(self.same))
        des += "- Not same dataset length : {}\n".format(len(self.not_same))
        des += "- Total files  : {}\n".format(len(self.all_files))
        des += "- Unique files : {} ({:.2f} %)\n".format(
            len(self.unique_files), 100 * len(self.unique_files) / len(self.all_files)
        )
        des += "- Cache size   : {} (loaded : {:.2f} %)".format(
            len(self.files_to_cache), 100 * len(self.cache) / len(self.files_to_cache)
        )
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
    
    def load(self, filename):
        if filename not in self.cache:
            data = self.load_file(filename)
            if filename not in self.files_to_cache: return data
            
            self.cache[filename] = data
        
        return self.cache[filename]
    
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


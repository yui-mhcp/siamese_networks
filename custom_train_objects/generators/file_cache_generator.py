
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

class FileCacheGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 dataset,
                 
                 id_column = 'id',
                 file_column    = 'filename',
                 processed_column   = '',
                 
                 preload        = False,
                 cache_size     = 30000,
                 min_occurence  = 3,
                 
                 shuffle        = False,
                 random_state   = None,
                 
                 process_kwargs = {},
                 ** kwargs
                ):
        assert isinstance(dataset, pd.DataFrame)
        self.dataset    = dataset
        self.process_kwargs = process_kwargs
        
        self.id_column  = id_column
        self.file_column    = file_column
        self.processed_column   = processed_column
        
        self.preload    = preload
        self.cache_size = cache_size
        self.min_occurence  = min_occurence
        
        self.shuffle        = shuffle
        self.random_state   = random_state
        
        self.build(** kwargs)
        
        self.unique_files, self.frequencies = self.get_uniques()

        self.cache  = {}
        self.files_to_cache = []
        self.build_cache(cache_size = cache_size, min_occurence = min_occurence, preload = preload)
    
    def build(self, ** kwargs):
        raise NotImplementedError()
        
    def load_file(self, filename):
        raise NotImplementedError()

    @property
    def all_files(self):
        raise NotImplementedError()

    @property
    def processed_output_shape(self):
        raise NotImplementedError()
    
    @property
    def output_signature(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, idx):
        raise NotImplementedError()

    def __str__(self):
        des = "{} Generator :\n".format(self.__class__.__name__.replace('Generator', ''))
        des += "- Length : {}\n".format(len(self))
        des += "- Unique ids   : {}\n".format(len(self.ids))
        des += "- Total files  : {}\n".format(len(self.all_files))
        des += "- Unique files : {} ({:.2f} %)\n".format(
            len(self.unique_files), 100 * len(self.unique_files) / len(self.all_files)
        )
        des += "- Cache size   : {} (loaded : {:.2f} %)".format(
            len(self.files_to_cache), 100 * len(self.cache) / max(len(self.files_to_cache), 1)
        )
        return des

    def get_uniques(self):
        uniques = {}
        for file in self.all_files:
            uniques.setdefault(file, 0)
            uniques[file] += 1
            
        return np.array(list(uniques.keys())), np.array(list(uniques.values()))
        
    def build_cache(self, cache_size = 30000, min_occurence = 2, preload = False, ** kwargs):
        # compute 'cache_size' most present files (in 2 ds)
        cache_idx   = np.flip(np.argsort(self.frequencies))[:cache_size]
        to_cache    = self.unique_files[cache_idx]
        # get files with at least 'min_occurence' occurences
        freq        = self.frequencies[cache_idx]
        to_cache    = to_cache[np.where(freq >= min_occurence)]
        
        self.files_to_cache = set(to_cache)
        
        if preload: self.load_cache(** kwargs)
        
        return to_cache
    
    def load_cache(self, tqdm = tqdm, filename = None, ** kwargs):
        from datasets import prepare_dataset
        
        if filename:
            from utils.file_utils import load_data
            self.cache.update(load_data(filename))
        
        self.cache = {f : self.cache[f] for f in self.unique_files if f in self.cache}
        
        not_cached = np.array([f for f in self.files_to_cache if f not in self.cache])
        if len(not_cached) > 0:
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

    def load(self, filename):
        if filename not in self.cache:
            data = self.load_file(filename)
            if filename not in self.files_to_cache: return data
            
            self.cache[filename] = data
        
        return self.cache[filename]

    def save_cache(self, filename):
        from utils.file_utils import dump_data
        dump_data(filename = filename, data = self.cache)
    
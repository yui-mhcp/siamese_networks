
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

from custom_train_objects.generators.file_cache_generator import FileCacheGenerator

class GrouperGenerator(FileCacheGenerator):
    def __init__(self, dataset, n_utterance, ** kwargs):
        super().__init__(dataset, n_utterance = n_utterance, ** kwargs)
    
    def load_file(self, filename):
        raise NotImplementedError()

    @property
    def processed_output_shape(self):
        raise NotImplementedError()

    def build(self,
              n_utterance   = None,
              max_ids   = None,
              n_round   = 100,
              max_length    = None,
              max_repeat    = 5,
              tqdm  = lambda x: x,
              random_state  = None,
              ** kwargs
             ):
        if n_utterance is None: n_utterance = self.n_utterance
        if random_state is None: random_state = self.random_state
        self.n_utterance = n_utterance

        self.ids    = []
        self.groups = []
        self.group_ids  = []
        
        ids_occurences = self.dataset[self.id_column].value_counts()
        ids_occurences = ids_occurences[ids_occurences >= n_utterance]
        if max_ids: ids_occurences = ids_occurences[:max_ids]
        
        if max_length: n_round = min(n_round, max_length // (len(ids_occurences) * n_utterance) + 1)
        
        self.ids    = {id_name : i for i, id_name in enumerate(ids_occurences.index)}
        
        groups  = [
            (data_id, datas[self.file_column].values, np.zeros((len(datas),)))
            for data_id, datas in self.dataset.groupby(self.id_column)
            if data_id in self.ids
        ]
        for i in tqdm(range(n_round)):
            rnd = None if random_state is None else random_state + i
            for data_id, files, n_repeat in groups:
                indexes = np.arange(len(files))[n_repeat < max_repeat]
                if len(indexes) < n_utterance: continue
                
                indexes = np.random.choice(indexes, size = n_utterance, replace = False)
                
                n_repeat[indexes] += 1
                
                self.groups.append(files[indexes])
                self.group_ids.append(self.ids[data_id])
        
        return self.groups, self.group_ids
    
    @property
    def all_files(self):
        flattened = []
        for group in self.groups: flattened.extend(group)
        return flattened
    
    @property
    def output_signature(self):
        label_dtype = tf.string if isinstance(self.group_ids[0], str) else tf.int32
        return {
            self.processed_column   : tf.TensorSpec(
                shape = self.processed_output_shape, dtype = tf.float32
            ),
            'id'    : tf.TensorSpec(shape = (), dtype = label_dtype)
        }

    def __len__(self):
        return len(self.groups) * self.n_utterance
    
    def __getitem__(self, idx):
        if idx == 0 and self.shuffle: self.shuffle_groups()
        return {
            self.processed_column   : self.load(
                self.groups[idx // self.n_utterance][idx % self.n_utterance]
            ),
            'id'    : self.group_ids[idx // self.n_utterance]
        }

    def shuffle_groups(self):
        indexes = sklearn_shuffle(np.arange(len(self.groups)))
        
        self.groups = [self.groups[i] for i in indexes]
        self.group_ids  = [self.group_ids[i] for i in indexes]

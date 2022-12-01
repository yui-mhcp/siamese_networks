
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

class GrouperGenerator(FileCacheGenerator):
    def __init__(self, dataset, n_utterance, batch_size = None, ** kwargs):
        super().__init__(dataset, n_utterance = n_utterance, ** kwargs)
        self.shuffle_groups = None
        if batch_size: self.set_batch_size(batch_size)
    
    def load_file(self, filename):
        raise NotImplementedError()

    @property
    def processed_output_shape(self):
        raise NotImplementedError()

    def build(self,
              n_utterance   = None,
              max_ids   = None,
              n_round   = 100,
              min_round_size    = 1,
              max_length    = None,
              max_repeat    = 5,
              tqdm  = lambda x: x,
              random_state  = None,
              ** kwargs
             ):
        if n_utterance is None: n_utterance = self.n_utterance
        if random_state is None: random_state = self.random_state
        self.n_utterance = n_utterance

        rnd = np.random.RandomState(random_state)
        
        self.ids    = []
        self.rounds = [0]
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
            for data_id, files, n_repeat in groups:
                indexes = np.arange(len(files))[n_repeat < max_repeat]
                if len(indexes) < n_utterance: continue
                
                indexes = rnd.choice(indexes, size = n_utterance, replace = False)
                
                n_repeat[indexes] += 1
                
                self.groups.append(files[indexes])
                self.group_ids.append(self.ids[data_id])
            
            if len(self.groups) - self.rounds[-1] < min_round_size:
                self.groups     = self.groups[:self.rounds[-1]]
                self.group_ids  = self.group_ids[:self.rounds[-1]]
                break
        
            self.rounds.append(len(self.groups))
        
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
        if idx == 0 and self.shuffle: self.shuffle_rounds()
        return {
            self.processed_column   : self.load(
                self.groups[idx // self.n_utterance][idx % self.n_utterance]
            ),
            'id'    : self.group_ids[idx // self.n_utterance]
        }

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
        group_size = batch_size // self.n_utterance
        
        self.shuffle_groups = [0]
        
        current_idx, group_ids = 0, set()
        while current_idx < len(self):
            overlap = len(group_ids) > 0 and any(
                id_i in group_ids for id_i in self.group_ids[current_idx : current_idx + group_size]
            )
            if current_idx > len(self) or overlap:
                group_ids = set()
                self.shuffle_groups.append(current_idx)
                continue
            
            group_ids.update(self.group_ids[current_idx : current_idx + group_size])
            current_idx += group_size
        
    def shuffle_rounds(self):
        assert self.shuffle_groups is not None, 'You must set `batch_size` with `set_batch_size()`'
        
        indexes = []
        for i, start in enumerate(self.shuffle_groups[:-1]):
            indexes.extend(sklearn_shuffle(
                list(range(start, self.shuffle_groups[i + 1])), random_state = self.random_state
            ))
        
        self.groups = [self.groups[i] for i in indexes]
        self.group_ids  = [self.group_ids[i] for i in indexes]

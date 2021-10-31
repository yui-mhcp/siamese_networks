import os
import numpy as np
import pandas as pd

from utils import load_json
from utils.text.cleaners import remove_control
from datasets.sqlite_dataset import preprocess_sqlite_database
from datasets.custom_datasets.preprocessing import parse_nq_annots

_siamese_renaming = {'sentence1' : 'text_x', 'sentence2' : 'text_y'}
_spaces = (' ', '\n', '\t')

def _clean_paragraph(para):
    para['context'] = remove_control(para['context'])
    for qa in para['qas']:
        qa['question']  = remove_control(qa['question'])
        qa['answers']   = [
            _clean_answer(para['context'], a['text'], a['answer_start']) for a in qa['answers']
        ]
    return para
        
def _clean_answer(context, answer, answer_start):
    answer = remove_control(answer)
    start, end = answer_start, answer_start + len(answer)
    
    while start > 0 and context[start - 1] not in _spaces: start -= 1
    while end < len(context) - 1 and context[end + 1] not in _spaces: end += 1
    
    return {'text' : context[start : end + 1], 'answer_start' : start, 'answer_end' : end}

def preprocess_europarl_annots(directory, base_name, input_lang, output_lang):
    input_filename = os.path.join(directory, '{}.{}'.format(base_name, input_lang))
    output_filename = os.path.join(directory, '{}.{}'.format(base_name, output_lang))
    
    with open(input_filename, 'r', encoding = 'utf-8') as input_file:
        inputs = input_file.read().split('\n')
        
    with open(output_filename, 'r', encoding = 'utf-8') as output_file:
        outputs = output_file.read().split('\n')

    datas = [[inp, out] for inp, out in zip(inputs, outputs)]
    return pd.DataFrame(data = datas, columns = [input_lang, output_lang])

def preprocess_nq_annots(directory, subset = 'train', file_no = -1, use_long_answer = False,
                         include_document = False, tqdm = lambda x: x, ** kwargs):
    if file_no == -1: file_no = list(range(50 if subset == 'train' else 5))
    if isinstance(file_no, (list, tuple)):
        return pd.concat([preprocess_nq_annots(
            directory, subset = subset, file_no = no, use_long_answer = use_long_answer,
            include_document = include_document, tqdm = tqdm, ** kwargs
        ) for no in file_no], ignore_index = True)
    
    dataset = parse_nq_annots(directory, subset = subset, file_no = file_no, tqdm = tqdm, ** kwargs)
    dataset = pd.DataFrame(dataset)
    if len(dataset) == 0: return dataset

    if not include_document: dataset.pop('paragraphs')
    if use_long_answer:
        dataset['answers'] = dataset['long_answer']
    else:
        dataset['answers'] = dataset.apply(
            lambda row: row['short_answers'][0] if isinstance(row['short_answers'], list) and len(row['short_answers']) > 0 else row['long_answer'],
            axis = 'columns'
        )
    
    dataset.pop('long_answer')
    dataset.pop('short_answers')

    return dataset

def preprocess_parade_annots(directory, subset = 'train', rename_siamese = True):
    filename = os.path.join(directory, 'PARADE_{}.txt'.format(subset))
    
    dataset = pd.read_csv(filename, sep = '\t')
    dataset['Binary labels'] = dataset['Binary labels'].astype(np.bool)
    
    if rename_siamese:
        dataset = dataset.rename(
            columns = {'Binary labels' : 'same', 'Definition1' : 'text_x', 'Definition2' : 'text_y', 'Entity' : 'title'}
        )
    
    return dataset

def preprocess_paws_annots(directory, subset = 'train', rename_siamese = True):
    filename = os.path.join(directory, '{}.tsv'.format(subset))
    
    dataset = pd.read_csv(filename, sep = '\t')
    dataset['label'] = dataset['label'].astype(np.bool)
    
    if rename_siamese:
        dataset = dataset.rename(
            columns = {'label' : 'same', 'sentence1' : 'text_x', 'sentence2' : 'text_y'}
        )
    
    return dataset

def preprocess_qqp_annots(directory, subset = 'train', rename_siamese = True):
    filename = os.path.join(directory, '{}.csv'.format(subset))
    
    dataset = pd.read_csv(filename, index_col = 0)
    dataset = dataset.dropna('index')
    dataset['is_duplicate'] = dataset['is_duplicate'].astype(np.bool)
    
    if rename_siamese:
        dataset = dataset.rename(
            columns = {'is_duplicate' : 'same', 'question1' : 'text_x', 'question2' : 'text_y'}
        )
    
    return dataset

def preprocess_snli_annots(directory, subset, version = '1.0', skip_parsed = True,
                           skip_sub_labels = True, skip_ukn_label = True, skip_id = True,
                           rename = _siamese_renaming, ** kwargs):
    if isinstance(subset, (list, tuple)):
        return pd.concat([preprocess_snli_annots(
            directory, sub, version = version, skip_parsed = skip_parsed,
            skip_sub_labels = skip_sub_labels, ** kwargs
        )] for sub in subset)
    
    filename = os.path.join(directory, 'snli_{}_{}.txt'.format(version, subset))
    dataset = pd.read_csv(filename, sep  = '\t')
    
    if skip_parsed or skip_id or skip_sub_label:
        for col in dataset.columns:
            if 'parse' in col and skip_parsed: dataset.pop(col)
            elif 'ID' in col and skip_id: dataset.pop(col)
            elif col != 'gold_label' and 'label' in col and skip_sub_labels: dataset.pop(col)
    
    if skip_ukn_label:
        dataset = dataset[dataset['gold_label'] != '-']
    
    dataset['same'] = dataset['gold_label'] == 'entailment'
    rename.setdefault('gold_label', 'label')
    dataset = dataset.rename(columns = rename)
    
    return dataset.dropna()

def process_sts_annots(directory, subset, rename = _siamese_renaming, ** kwargs):
    if isinstance(subset, (list, tuple)):
        return pd.concat([preprocess_snli_annots(
            directory, sub, ** kwargs
        )] for sub in subset)
    
    filename = os.path.join(directory, 'sts-{}.csv'.format(subset))

    with open(filename, 'r', encoding = 'utf-8') as file:
        lines = file.read().split('\n')

    dataset = pd.DataFrame(
        [l.split('\t')[:7] for l in lines if len(l) > 0],
        columns = ['category', 'type', 'date', 'id', 'score', 'sentence1', 'sentence2']
    )
    
    dataset = dataset.rename(columns = rename)
    
    return dataset

def preprocess_SQUAD_annots(directory, subset, version = '2.0', skip_impossible = False,
                            clean_text = True, keep_mode = 'longest', ** kwargs):
    assert keep_mode in ('all', 'longest', 'shortest', 'one_per_line')
    
    filename = os.path.join(directory, '{}-v{}.json'.format(subset, version))
    metadata = load_json(filename)['data']
    
    dataset = []
    contexts = {}
    for data in metadata:
        for para in data['paragraphs']:
            if clean_text: para = _clean_paragraph(para)
            contexts.setdefault(para['context'], len(contexts))
            for qa in para['qas']:
                _base_infos = {
                    'title' : data['title'], 'context_id' : contexts[para['context']],
                    'context' : para['context'], 'question' : qa['question']
                }
                
                if  qa['is_impossible']:
                    if not skip_impossible:
                        dataset.append({
                            ** _base_infos, 'answers' : '', 'answer_start' : -1
                        })
                    continue
                
                if keep_mode == 'one_per_line':
                    for a in qa['answers']:
                        dataset.append({
                            ** _base_infos, 'answers' : a['text'], 'answer_start' : a['answer_start']
                        })
                    continue

                if keep_mode == 'all':
                    answer  = [a['text'] for a in qa['answers']]
                    answer_start    = [a['answer_start'] for a in qa['answers']]
                elif keep_mode in ('longest', 'shortest'):
                    sort = sorted(
                        qa['answers'], key = lambda a: len(a['text']), reverse = keep_mode == 'longest'
                    )
                    answer, answer_start = sort[0]['text'], sort[0]['answer_start']
                
                dataset.append({
                    ** _base_infos, 'answers' : answer, 'answer_start' : answer_start
                })
    
    dataset = pd.DataFrame(dataset)
    
    return dataset

def preprocess_triviaqa_annots(directory, unfiltered = False, wikipedia = True, load_context = False,
                               keep_doc_mode = 'one_per_line', subset = 'train', tqdm = lambda x: x, ** kwargs):
    def get_contexts(contexts):
        result = []
        for c in contexts:
            if 'Filename' not in c: continue
            f = c['Filename']
            for char in ('?', ':', '*', '"'): f = f.replace(char, '_')
            f = os.path.join(directory, 'evidence', prefix, f)
            if not os.path.exists(f): print("File for context {} does not exist !".format(c))
            text = None
            if load_context:
                with open(f, 'r', encoding = 'utf-8') as file:
                    text = file.read()
            
            result.append({
                'context_id'    : c['Title'],
                'filename' : f,
                'title'    : c['Title'],
                'context'  : text
            })
        return result

    if unfiltered: wikipedia = False
    prefix = 'wikipedia' if wikipedia else 'web'
    
    filename = '{}-{}.json'.format(prefix, subset)
    if unfiltered:
        filename = os.path.join('triviaqa-unfiltered', 'unfiltered-' + filename)
    else:
        filename = os.path.join('qa', filename)
    
    filename = os.path.join(directory, filename)

    data = load_json(filename)['Data']
    
    metadata = []
    for qa in tqdm(data):
        contexts = get_contexts(qa.get('EntityPages', []))
        if len(contexts) == 0: continue
        
        if keep_doc_mode == 'one_per_line':
            for i, c in enumerate(contexts):
                metadata.append({
                    'id'       : qa['QuestionId'] + '_doc_{}'.format(i),
                    'question' : qa['Question'],
                    'answers'  : qa['Answer']['Value'],
                    ** c
                })
        else:
            c = contexts[0] if keep_doc_mode == 'first' else contexts[-1]
            metadata.append({
                'id'       : qa['QuestionId'],
                'question' : qa['Question'],
                'answers'  : qa['Answer']['Value'],
                ** c
            })
    
    return pd.DataFrame(metadata)

def preprocess_parsed_wiki_annots(directory):
    filename = os.path.join(directory, 'psgs_w100.tsv')
    return pd.read_csv(filename, sep = '\t')


_custom_text_datasets = {
    'europarl'  : {
        'directory' : '{}/Europarl',
        'base_name' : 'europarl-v7.fr-en',
        'input_lang'    : 'en',
        'output_lang'   : 'fr'
    },
    'nq'        : {
        'train' : {'directory' : '{}/NaturalQuestions', 'subset' : 'train'},
        'valid' : {'directory' : '{}/NaturalQuestions', 'subset' : 'dev'}
    },
    'parade'    : {
        'train' : {'directory' : '{}/PARADE', 'subset' : 'train'},
        'valid' : {'directory' : '{}/PARADE', 'subset' : 'validation'}
    },
    'paws'  : {
        'train' : {'directory' : '{}/PAWS', 'subset' : 'train'},
        'valid' : {'directory' : '{}/PAWS', 'subset' : 'dev'}
    },
    'qqp'   : {
        'directory' : '{}/QQP',
        'subset'    : 'train'
    },
    'snli'  : {
        'train' : {'directory' : '{}/snli_1.0', 'subset' : 'train'},
        'valid' : {'directory' : '{}/snli_1.0', 'subset' : 'dev'}
    },
    'sts'   : {
        'train' : {'directory' : '{}/sts_benchmark', 'subset' : 'train'},
        'valid' : {'directory' : '{}/sts_benchmark', 'subset' : 'dev'}
    },
    'squad' : {
        'train' : {'directory' : '{}/SQUAD2.0', 'subset' : 'train'},
        'valid' : {'directory' : '{}/SQUAD2.0', 'subset' : 'dev'}
    },
    'triviaqa'  : {
        'train' : {'directory' : '{}/TriviaQA', 'subset' : 'train'},
        'valid' : {'directory' : '{}/TriviaQA', 'subset' : 'dev'}
    },
    'wikipedia' : {'directory' : '{}/wikipedia', 'filename' : 'docs.db'},
    'wikipedia_parsed'  : {'directory' : '{}/wikipedia'}
}

_text_dataset_processing  = {
    'europarl'      : preprocess_europarl_annots,
    'nq'            : preprocess_nq_annots,
    'parade'        : preprocess_parade_annots,
    'paws'          : preprocess_paws_annots,
    'qqp'           : preprocess_qqp_annots,
    'snli'          : preprocess_snli_annots,
    'sts'           : process_sts_annots,
    'squad'         : preprocess_SQUAD_annots,
    'triviaqa'      : preprocess_triviaqa_annots,
    'wikipedia'     : preprocess_sqlite_database,
    'wikipedia_parsed'  : preprocess_parsed_wiki_annots
}
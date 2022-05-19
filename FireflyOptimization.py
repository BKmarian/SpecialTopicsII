import os
import pdb
import xml.etree.ElementTree as etree
import numpy as np
import json
import nltk
import re

from nltk import word_tokenize
from nltk.corpus import wordnet, wordnet_ic
from tqdm import trange
from improved_lesk import lesk_distance, pos_map, STOPWORDS, lemmatizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from functools import lru_cache
from glob import glob
synset_types = {1:'n', 2:'v', 3:'a', 4:'r', 5:'s'}

firefly_results_path = os.path.join("logs", "results_firefly.json")
def save_results(results):
    with open(firefly_results_path, "w") as fout:
        json.dump(results, fout, indent=4)

swarm_size = 40
alpha = 0.2
gamma = 1
window_size = 5
window_stride = 1
local_rate = 0.15
lfa = 1700

max_iterations = 30000
eps_convergence = 1e-3
patience = 10

ic_brown = wordnet_ic.ic('ic-brown.dat')

intensity_db = {}
lesk_db = {}
resink_db = {}
firefly_hashing = lambda syn_vec: " ".join(map(str, map(int, syn_vec)))
concepts_hasing = lambda *args: " ".join(sorted(map(str, args)))
# firefly_best.syn_set[0].lemmas()[0].key() - if needed

def extract_sentences_from_xml(xml_path):
    with open(xml_path, 'r') as fin:
        content = fin.read()
        tree = etree.fromstring(content)
    
    dataset = []
    # contextfile > context > p > s
    context = tree.find('context')
    for i, elem in enumerate(context.findall('p')):
        sentence_elem = elem.find('s')

        sentence = []
        for wf_elem in sentence_elem.findall('wf'):
            wf_atributes = wf_elem.attrib
            wf_lemma = wf_atributes.get("lemma", "")
            wf_wasn = wf_atributes.get("wnsn", "")
            if wf_atributes['cmd'] != 'ignore' and wf_atributes["pos"] != "NNP" and wf_lemma != "" and wf_wasn != "":
                pos_ = wf_atributes["pos"]
                pos_nltk = pos_map(pos_)
                
                sentence.append({
                    "pos": pos_,
                    "pos_nltk": pos_nltk,
                    "lemma": wf_lemma,
                    "wnsn": wf_wasn
                })
        dataset += sentence
    return dataset

def xml_path_to_id(xml_path):
    basename = os.path.basename(xml_path)
    return basename

def extract_sentences_from_senseval(xml_path, keys_dict, version):
    with open(xml_path, 'r') as fin:
        content = fin.read()
        tree = etree.fromstring(content)

    dataset = {}
    for text_id, elem in enumerate(tree.findall('text')):
        text_dicts = []

        for sent in elem.findall('sentence'):
            for wf_elem in sent.findall('instance'):
                wf_atributes = wf_elem.attrib
                instance_id = wf_atributes.get("id")
                
                syn_name = keys_dict.get(instance_id).name()
                lemma, pos_nltk, wasn = syn_name.split('.')

                text_dicts.append({
                    "pos": wf_atributes.get("pos", ""),
                    "lemma": lemma,
                    "pos_nltk": pos_nltk,
                    "wnsn": wasn
                })
        text_key = f'senseval{version}_{text_id}'
        dataset[text_key] = text_dicts
    return dataset

def synset_from_key(sense_key):
    lemma, ss_type, _, lex_id, _, _ = re.match(r"(.*)\%(.*):(.*):(.*):(.*):(.*)", sense_key).groups()
    ss_idx = '.'.join([lemma, synset_types[int(ss_type)], lex_id])
    try:
        syn = wordnet.synset(ss_idx)
    except:
        syns = wordnet.synsets(lemma, pos=synset_types[int(ss_type)])
        lex_id_int = int(lex_id[1:] if lex_id[0] == '0' else lex_id)

        lex_id_int = min(lex_id_int, len(syns) - 1)
        syn = syns[lex_id_int]
    return syn

def get_senseval_dataset():
    dataset = {}
    for version in [2, 3]:
        xml_path = os.path.join(f'senseval{version}', f'senseval{version}.data.xml')
        keys_path =  os.path.join(f'senseval{version}', f'senseval{version}.gold.key.txt')

        with open(keys_path, 'r') as fin:
            keys_list = fin.read().split('\n')
        
        keys_dict = {}
        for elem in keys_list[:-1]:
            instance_id, sense_key = elem.split(' ')[:2]
            keys_dict[instance_id] = synset_from_key(sense_key)
        
        senseval_set = extract_sentences_from_senseval(xml_path, keys_dict, version)
        dataset.update(senseval_set)
    return dataset


class Firefly:
    def __init__(self, syn_set=[], syn_vec=np.array([])):
        self.syn_set = syn_set
        self.syn_vec = syn_vec
        self.beta = 0
    
    def to_json(self):
        return {
            "position": self.syn_vec.tolist(),
            "synsets": [syn.name() for syn in self.syn_set],
            "intensity": float(self.beta)
        }
        
    def __repr__(self):
        vec_str = ' '.join(map(str, self.syn_vec))
        syn_str = ' '.join([syn.name() for syn in self.syn_set])
        return f'{vec_str}\n{syn_str}\nIntensity: {self.beta}'


    @classmethod
    def from_indices(self, firefly_indices, synsets_list):
        firefly_synsets = []
        for idx, synsets in zip(firefly_indices, synsets_list):
            idx = int(max(0, min(idx, len(synsets) - 1)))
            firefly_synsets.append(synsets[idx])
        
        firefly = Firefly(firefly_synsets, np.array(firefly_indices, dtype=np.float32))
        return firefly

    @classmethod
    def from_sample(self, synsets_list):
        """ builder that genrates the cartesian product extracted randomly uniform
        """
        firefly_indices = []
        for synsets in synsets_list:
            syn_range = len(synsets)
            firefly_idx = np.random.randint(low=0, high=syn_range)
            firefly_indices.append(firefly_idx)
        
        firefly = Firefly.from_indices(firefly_indices, synsets_list)
        return firefly
    
    @classmethod
    def from_semcor(self, sent_dict, synsets_list):
        firefly_indices = []
        firefly_synsets = []
        for word_dict, syn_list in zip(sent_dict, synsets_list):
            lemma = lemmatizer.lemmatize(word_dict['lemma'], pos=pos_map(word_dict['pos_nltk']))
            synset_str = '.'.join([lemma, word_dict['pos_nltk'], word_dict['wnsn']])
            try:
                synset = wordnet.synset(synset_str)
            except: 
                rand_id = np.random.randint(0, len(syn_list), size=1)[0]
                synset = syn_list[rand_id]
            
            syn_id = syn_list.index(synset) if synset in syn_list else -1
            firefly_indices.append(syn_id)
            firefly_synsets.append(synset)
        
        firefly = Firefly(firefly_synsets, np.array(firefly_indices, dtype=np.float32))
        return firefly


def extract_fireflies(sent_dict):
    senses_list = []
    for word_dict in sent_dict:
        word_senses = wordnet.synsets(word_dict["lemma"], pos=word_dict['pos_nltk'])
        if len(word_senses) > 0:
            senses_list.append(word_senses)

    fireflies = []
    for _ in range(swarm_size):
        firefly = Firefly.from_sample(senses_list)
        fireflies.append(firefly)
    return fireflies, senses_list

@lru_cache(maxsize=None)  
def get_information_content(concept_i, concept_j):
    pos_i = concept_i.name().split('.')[1]
    pos_j = concept_j.name().split('.')[1]
    if pos_i != pos_j:
        return 0
    
    try:
        resink_similarity = concept_i.res_similarity(concept_j, ic=ic_brown)
    except:
        resink_similarity = 0
    return resink_similarity

def compute_individual_light_intensity(firefly):
    light_intensity = 0
    synsets = firefly.syn_set
    firefly_len = len(synsets)

    last_start = max(1, firefly_len - window_size)
    for start in range(0, last_start, window_stride):
        end = start + window_size
        firefly_window = synsets[start: end]
        for i in range(len(firefly_window)):
            concept_i = firefly_window[i]
            for j in range(i):
                concept_j = firefly_window[j]

                light_intensity += lesk_distance(concept_i, concept_j, max_ngrams=5) + \
                                   get_information_content(concept_i, concept_j)

    firefly.beta = light_intensity
    return light_intensity

def compute_light_intensity(fireflies, show_progress=False):
    # we also memorize the already computed light intensities for reducing the computational effort
    light_intesities = []
    for idx, firefly in enumerate(fireflies):

        firefly_hash = firefly_hashing(firefly.syn_vec)
        light_intensity = intensity_db.get(firefly_hash, -1)
        
        if light_intensity < 0:
            light_intensity = compute_individual_light_intensity(firefly)
            
        light_intesities.append(light_intensity)
        intensity_db[firefly_hash] = light_intensity

        if show_progress:
            print(f"Computing light intensity progress {idx}/{len(fireflies)}", end='\r')
    
    return light_intesities

def get_euclidean_distance(x_i, x_j):
    return np.sqrt(np.sum(np.square(x_i - x_j)))

def update_firefly_if_needed(fireflies, senses_list):
    for firefly in fireflies:
        syn_vec = np.round(firefly.syn_vec)
        firefly.syn_set = [synsets[max(0, min(int(index), len(synsets) - 1))] \
                           for index, synsets in zip(syn_vec, senses_list)]

def apply_fa(entry):
    fireflies, senses_list = extract_fireflies(entry)                       
    global intensity_db, lesk_db
    intensity_db = {}
    lesk_db = {}
    prev_intensity = -1
    patience_idx = 0

    light_intensities = compute_light_intensity(fireflies)
    for _ in trange(max_iterations):
        for i in range(len(fireflies)):
            firefly_i = fireflies[i]
            x_i = firefly_i.syn_vec
            
            for j in range(i):
                firefly_j = fireflies[j]
                x_j = firefly_j.syn_vec

                if firefly_j.beta > firefly_i.beta:
                    r_i_j = get_euclidean_distance(x_i, x_j)
                    beta = firefly_i.beta * np.exp(- gamma * r_i_j ** 2)

                    rand_ = np.random.uniform(0, 1)
                    x_i += beta * r_i_j * (x_i - x_j) * alpha * (rand_ - 0.5)
                    firefly_i.syn_vec = x_i

        update_firefly_if_needed(fireflies, senses_list)
        light_intensities = compute_light_intensity(fireflies)
        i_best = np.argmax(light_intensities)
        firefly_best = fireflies[i_best]
        
        firefly_best = apply_local_search(firefly_best, senses_list)
        if firefly_best not in fireflies:
            fireflies.append(firefly_best)
        
        best_intensity = firefly_best.beta
        
        if best_intensity - prev_intensity < eps_convergence:
            if patience_idx < patience:
                patience_idx += 1
            else:
                break
        else:
            patience_idx = 0

        prev_intensity = best_intensity
        

    return firefly_best, senses_list

def variate_vec(neighbour_vec, radius=1):
    for i in range(len(neighbour_vec)):
        neighbour_vec[i] = np.round(neighbour_vec[i] + np.random.uniform(-radius, radius, 1))
    return neighbour_vec

def apply_local_search(firefly_best, senses_list):
    rand_ = np.random.binomial(1, local_rate, 1)[0]
    if rand_ == 0:
        return firefly_best
    
    best_vec = firefly_best.syn_vec
    firefly_neighbours = {}
    for i in range(lfa):
        neighbour_vec = np.copy(best_vec)
        neighbour_vec = variate_vec(neighbour_vec)
        
        neighbour_hash = firefly_hashing(neighbour_vec)
        existing_neighbour = firefly_neighbours.get(neighbour_hash)
        if existing_neighbour is not None:
            continue
        
        neighbour = Firefly.from_indices(neighbour_vec, senses_list)
        firefly_neighbours[neighbour_hash] = neighbour
        print(f"Generating neighbours LHAC {i}/{lfa}", end='\r')
    print()
    
    neighbour_list = list(firefly_neighbours.values())
    light_intensities = compute_light_intensity(neighbour_list, show_progress=True)
    
    neighbour_best_id = np.argmax(light_intensities)
    neighbour_best = neighbour_list[neighbour_best_id]
    if firefly_best.beta > light_intensities[neighbour_best_id]:
        return neighbour_best
    return firefly_best

def get_sent_dict(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("\s+", " ", text)
    sent_pair = nltk.pos_tag(word_tokenize(text))
    sent_dict = [
        {
            "lemma": lemmatizer.lemmatize(token, pos=pos_map(pos)),
            "pos_nltk": pos_map(pos)
        } 
        for token, pos in sent_pair if token not in STOPWORDS
    ]
    return sent_dict


def inference(text):
    sent_dict = get_sent_dict(text)
    firefly_best, _ = apply_fa(sent_dict)
    print(firefly_best.to_json())


def main():
    xml_re = os.path.join('semcor', 'semcor', 'brown1', 'tagfiles', '*.xml')
    # xml_re = os.path.join('semcor', 'semcor', 'brown2', 'tagfiles', '*.xml')
    # xml_re = os.path.join('semcor', 'semcor', 'brownv', 'tagfiles', '*.xml')
    dataset = {xml_path_to_id(xml_path): extract_sentences_from_xml(xml_path) for xml_path in glob(xml_re)}
    
    # dataset = get_senseval_dataset()

    results = []
    if os.path.join(firefly_results_path):
        with open(firefly_results_path, 'r') as fin:
            results = json.load(fin)

    accuracy_s = []
    f1_macro_s = []
    precisions = []
    recalls = []
    for key, entry in dataset.items():
        if key not in results.keys():
            firefly_best, senses_list = apply_fa(entry)
            firefly_gt = Firefly.from_semcor(entry, senses_list)
            compute_individual_light_intensity(firefly_gt)

            gt_list = [syn.name() for syn in firefly_gt.syn_set]
            best_list = [syn.name() for syn in firefly_best.syn_set]
            accuracy = accuracy_score(gt_list, best_list)
            f1_macro = f1_score(gt_list, best_list, average="macro")
            precision = precision_score(gt_list, best_list, average="macro", zero_division=True)
            recall = recall_score(gt_list, best_list, average="macro", zero_division=True)
            
            results[key] = {
                "firefly_gt": firefly_gt.to_json(),
                "firefly_found": firefly_best.to_json(),
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "precision": precision,
                "recall": recall
            }
            accuracy_s.append(accuracy)
            f1_macro_s.append(f1_macro)
            precisions.append(precision)
            recalls.append(recall)

            save_results(results)

    mean_accuracy = np.mean(accuracy_s)
    mean_f1 = np.mean(f1_macro_s)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    print(f"Finished with mean accuracy {mean_accuracy}, mean F1 {mean_f1}, mean precision {mean_precision}, mean recall {mean_recall}")
    save_results(results)

if __name__ == "__main__":
    main()
    # inference("Screw each stringer to the top of the largest deck frame with a drill.")

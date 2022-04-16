import os
import pdb
import xml.etree.ElementTree as etree
import numpy as np

from nltk.corpus import wordnet, wordnet_ic
from tqdm import trange
from improved_lesk import lesk_distance
from firefly_utils import swarm_size, window_size, window_stride, max_iterations, alpha, gamma
ic_brown = wordnet_ic.ic('ic-brown.dat')

def pos_map(pos_):
    first_letter = pos_[0].lower()
    if first_letter == 'j':
        return 'a'
    return first_letter

def extract_sentences_from_xml(xml_path):
    with open(xml_path, 'r') as fin:
        content = fin.read()
        tree = etree.fromstring(content)
    
    dataset = []
    # contextfile > context > p > s
    context = tree.find('context')
    for i, elem in enumerate(context.findall('p')):
        sentence_elem = elem.find('s')
        
        # for testing purposes
        if i > 10: 
            break

        sentence = []
        for wf_elem in sentence_elem.findall('wf'):
            wf_atributes = wf_elem.attrib
            if wf_atributes['cmd'] != 'ignore' and wf_atributes["pos"] != "NNP":
                pos_ = wf_atributes["pos"]
                pos_nltk = pos_map(pos_)
                
                sentence.append({
                    "pos": pos_,
                    "pos_nltk": pos_nltk,
                    "lemma": wf_atributes.get("lemma", ""),
                    "wnsn": wf_atributes.get("wnsn", "")
                })
        dataset.append(sentence)
    return dataset


class Firefly:
    def __init__(self, syn_set=[], syn_vec=np.array([])):
        self.syn_set = syn_set
        self.syn_vec = syn_vec
        self.beta = 0

    @classmethod
    def from_sample(self, synsets_list):
        """ builder that genrates the cartesian product extracted randomly uniform
        """
        firefly_indices = []
        firefly_synsets = []
        for synsets in synsets_list:
            syn_range = len(synsets)
            firefly_idx = np.random.randint(low=0, high=syn_range)
            firefly_indices.append(firefly_idx)
            firefly_synsets.append(synsets[firefly_idx])
        
        firefly = Firefly(firefly_synsets, np.array(firefly_indices, dtype=np.float32))
        return firefly


def extract_fireflies(sent_dict):
    senses_list = []
    for word_dict in sent_dict:
        word_senses = wordnet.synsets(word_dict["lemma"], pos=word_dict['pos_nltk'])
        senses_list.append(word_senses)

    fireflies = []
    for _ in range(swarm_size):
        firefly = Firefly.from_sample(senses_list)
        fireflies.append(firefly)
    return fireflies, senses_list


def get_lcs(concept_i, concept_j):
    path_i = concept_i.hypernym_paths()[0]
    path_j = concept_j.hypernym_paths()[0]

    for hypernym in path_j[::-1]:
        if hypernym in path_i:
            return hypernym
    return None

def get_information_content(concept_i, concept_j):
    pos_i = concept_i.name().split('.')[1]
    pos_j = concept_j.name().split('.')[1]
    if pos_i != pos_j:
        return 0
    
    lcs_concept = get_lcs(concept_i, concept_j)
    
    # ISSUE: don't know what is the second argument
    resink_similarity = 0
    resink_similarity += concept_i.res_similarity(lcs_concept, ic=ic_brown)
    resink_similarity += concept_j.res_similarity(lcs_concept, ic=ic_brown)
    return resink_similarity

def compute_light_intensity(fireflies):
    light_intesities = []
    for firefly in fireflies:
        light_intensity = 0

        synsets = firefly.syn_set
        firefly_len = len(synsets)

        last_start = max(1, firefly_len - window_size)
        for start in range(0, last_start, window_stride):
            end = start + window_size
            firefly_window = synsets[start: end]
            for i in range(window_size):
                concept_i = firefly_window[i]
                for j in range(i):
                    concept_j = firefly_window[j]
                    light_intensity += lesk_distance(concept_i, concept_j) + get_information_content(concept_i, concept_j)
        # TODO: double check this
        firefly.beta = light_intensity
        light_intesities.append(light_intensity)
    return light_intesities

def get_euclidean_distance(x_i, x_j):
    return np.sqrt(np.sum(np.square(x_i - x_j)))


def update_firefly_if_needed(fireflies, senses_list):
    for firefly in fireflies:
        firefly.syn_vec = np.round(firefly.syn_vec)
        # TODO: double check if it's ok to cycle by %
        firefly.syn_set = [synsets[int(index) % len(synsets)] for index, synsets in zip(firefly.syn_vec, senses_list)]

def apply_fa(entry):
    fireflies, senses_list = extract_fireflies(entry)
    firefly_len = len(fireflies)

    light_intensities = compute_light_intensity(fireflies)
    for _ in trange(max_iterations):
        for i in range(firefly_len):
            firefly_i = fireflies[i]
            x_i = firefly_i.syn_vec
            
            for j in range(i):
                firefly_j = fireflies[j]
                x_j = firefly_j.syn_vec

                r_i_j = get_euclidean_distance(x_i, x_j)
                beta = firefly_i.beta * np.exp(- gamma * r_i_j ** 2)

                rand_ = np.random.uniform(0, 1)
                x_i += beta * r_i_j * (x_i - x_j) * alpha * (rand_ - 0.5)
                firefly_i.syn_vec = x_i

        update_firefly_if_needed(fireflies, senses_list)
        light_intensities = compute_light_intensity(fireflies)

    i_best = np.argmin(light_intensities)
    firefly_best = fireflies[i_best]
    return firefly_best

xml_path = os.path.join('semcor', 'semcor', 'brown1', 'tagfiles', 'br-a01.xml')
dataset = extract_sentences_from_xml(xml_path)

for entry in dataset:
    firefly_best = apply_fa(entry)
    pdb.set_trace()

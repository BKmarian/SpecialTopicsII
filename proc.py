import os
import pdb
import xml.etree.ElementTree as etree
import numpy as np

from nltk.corpus import wordnet, wordnet_ic
from tqdm import trange
from improved_lesk import lesk_distance
from firefly_utils import swarm_size, window_size, window_stride, max_iterations, alpha, gamma, local_rate, lfa
ic_brown = wordnet_ic.ic('ic-brown.dat')

intensity_db = {}
lesk_db = {}
resink_db = {}
firefly_hashing = lambda syn_vec: " ".join(map(str, map(int, syn_vec)))
concepts_hasing = lambda *args: " ".join(sorted(map(str, args)))
# firefly_best.syn_set[0].lemmas()[0].key() - if needed

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

def lesk_distance_wrapper(concept_i, concept_j):
    concepts_hash = concepts_hasing(concept_i, concept_j)
    distance = lesk_db.get(concepts_hash)
    if distance is not None:
        return distance
    
    distance = lesk_distance(concept_i, concept_j)
    lesk_db[concepts_hash] = distance
    return distance

def get_information_content(concept_i, concept_j):
    pos_i = concept_i.name().split('.')[1]
    pos_j = concept_j.name().split('.')[1]
    if pos_i != pos_j:
        return 0

    concepts_hash = concepts_hasing(concept_i, concept_j)
    resink_similarity = resink_db.get(concepts_hash)
    
    if resink_similarity is not None:
        return resink_similarity
    
    resink_similarity = concept_i.res_similarity(concept_j, ic=ic_brown)
    resink_db[concepts_hash] = resink_similarity
    return resink_similarity

def compute_individual_light_intensity(firefly):
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
                light_intensity += lesk_distance_wrapper(concept_i, concept_j) + \
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

    light_intensities = compute_light_intensity(fireflies)
    for _ in trange(max_iterations):
        for i in range(len(fireflies)):
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
        i_best = np.argmax(light_intensities)
        firefly_best = fireflies[i_best]
        
        firefly_best = apply_local_search(firefly_best, senses_list)
        if firefly_best not in fireflies:
            fireflies.append(firefly_best)

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

xml_path = os.path.join('semcor', 'semcor', 'brown1', 'tagfiles', 'br-a01.xml')
dataset = extract_sentences_from_xml(xml_path)

for entry in dataset:
    firefly_best, senses_list = apply_fa(entry)
    pdb.set_trace()
    # TODO: implement a saving method called after some epochs/iterations

#   72/30000 [1:42:52]
#  609/30000 [1:33:52]

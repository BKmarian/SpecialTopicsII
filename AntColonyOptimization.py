import random
from nltk.corpus import stopwords
from functools import reduce
import spacy
#from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
import enum
import os
import pdb
import xml.etree.ElementTree as etree
import numpy
from nltk.corpus import wordnet, wordnet_ic
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
import datetime
import os
import neattext as nt
import nltk
import math
import re
import json

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('wordnet_ic')
ic_brown = wordnet_ic.ic('ic-brown.dat')

EVAPORATE_RATE = 0.3
E_max = 40
E_0 = 5
omega = 25  # ant life duration
Ea = 16  # energy_taken_by_ant_when arriving on node
deltav = 0.6
max_iterations = 200  # CYCLES
pheromone_deposit = 10
odour_length = 100
nodes_list = list()
edges_list = list()
STOPWORDS = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_md")
f = open("results.txt", "w")

synset_types = {1:'n', 2:'v', 3:'a', 4:'r', 5:'s'}

class NodeType(enum.Enum):
    sense = 1
    word = 2
    sentence = 3
    text = 4


class EdgeType(enum.Enum):
    edge = 1
    bridge = 2


def clean(mytext):
    docx = nt.TextFrame(text=mytext)
    docx.remove_puncts()
    docx.remove_stopwords()
    docx.remove_html_tags()
    docx.remove_special_characters()
    docx.remove_emojis()
    docx.fix_contractions()
    docx.remove_numbers()
    return docx.text


class Node:
    def __init__(self, sense, parent, typeN):
        self.type = typeN
        self.energy = E_0
        self.sense = sense
        if (self.type == NodeType.sense):
            words = clean(sense.definition()).split(' ')
            self.odour = set([word.lower().strip() for word in words if word not in STOPWORDS and word != ''])
        else:
            self.odour = list()  # [None] * 100
        self.parent = parent
        self.children = []
        self.neighbours = list()
        if self.parent != None:
            parent.children.append(self)

    def add_neighbour(self, edge, node):
        self.neighbours.append((edge, node))

    def printT(self, depth):
        for i in range(0, depth):
            print("    ", end=" ")
        print(self.type)
        for child in self.children:
            child.printT(depth + 1)

    def reduce_energy(self):
        self.energy = self.energy - 1

    def setParent(self, node):
        self.parent = node

    def produce_ant(self):
        return (numpy.arctan(self.energy)) / numpy.pi + 0.5


class Ant:
    def __init__(self, node: Node):
        self.direction = 1
        self.lifespan = omega
        self.energy = 1
        self.odour = node.odour  # senses words
        self.currentNode = node
        self.nest = node
        self.nodeChosen = None
        self.edgeChosen = None

    def change_direction(self):
        self.direction = self.direction * (-1)

    def should_return(self):
        r = random.randint(0, 100)
        return r < (float(self.energy / E_max) * 100)

    def is_dead(self):
        return self.lifespan == 0


class Edge:
    def __init__(self, source: Node, dest: Node, edgeType: EdgeType):
        self.source = source
        self.dest = dest
        self.pheromone = 0.0
        self.edgeType = edgeType
        source.add_neighbour(self, dest)
        dest.add_neighbour(self, source)

    def eval_f(self):
        return 1 - self.pheromone  # create bridge when 0

    def change_pheromone(self):
        self.pheromone = (float(1 - EVAPORATE_RATE)) * self.pheromone


@lru_cache(maxsize=None)
def setence_sim_spacy(sent1, sent2):
    doc1 = nlp(sent1)
    doc2 = nlp(sent2)
    return doc1.similarity(doc2)


def full_probability(probabilities):
    return numpy.random.choice(len(probabilities), 1, p=probabilities)[0]


def get_neighbours(node: Node):
    return node.neighbours


def apply_similarity_metric(description1, description2):
    description1 = list(filter(lambda x : nlp.vocab[x].has_vector ,description1))
    description2 = list(filter(lambda x : nlp.vocab[x].has_vector ,description2))

    if len(description1) == 0 or len(description2) == 0:
        return 0

    return setence_sim_spacy(" ".join(description1), " ".join(description2))
    # return lesk_distance_full(description1,description2)


def iterate():
    ants_list = list()
    global edges_list
    # bridges_list = list()
    for i in range(0, max_iterations):
        # print("Iteration ",i)
        for ant in [ant for ant in ants_list if ant.is_dead == True]:
            ant.currentNode.energy = ant.currentNode.energy + ant.energy
        ants_list = [ant for ant in ants_list if ant.is_dead == False]
        edges_list = [edge for edge in edges_list if (edge.pheromone != 0 or edge.edgeType != EdgeType.bridge)]

        for node in nodes_list:
            if node.type == NodeType.sense and node.produce_ant() > random.random():
                ant = Ant(node)
                node.reduce_energy()
                ants_list.append(ant)

        # probabilities = list()
        for ant in ants_list:
            probabilities = list()
            if ant.direction == 1 and ant.should_return():
                ant.change_direction()
            eval_sum = 0
            neighboursRoutes = get_neighbours(ant.currentNode)
            if ant.direction == 1:
                energy_sum = sum([node.energy for _, node in neighboursRoutes])
                for (edge, node) in neighboursRoutes:
                    nodeEval = float(node.energy / energy_sum)
                    edgeEval = float(1 - edge.pheromone)
                    probabilities.append(nodeEval + edgeEval)
                    eval_sum += nodeEval + edgeEval

                if eval_sum == 0:
                    (ant.edgeChosen, ant.nodeChosen) = random.choice(neighboursRoutes)
                else:
                    probabilities_new = [float(prob / eval_sum) for prob in probabilities]
                    index = full_probability(probabilities_new)
                    (ant.edgeChosen, ant.nodeChosen) = neighboursRoutes[index]
            else:
                suma = sum([apply_similarity_metric(node.odour, ant.odour) for _, node in neighboursRoutes])
                for (edge, node) in neighboursRoutes:
                    edgeEval = edge.pheromone
                    if suma == 0:
                        nodeEval = 0
                    else:
                        nodeEval = float(apply_similarity_metric(node.odour, ant.odour) / suma)
                    probabilities.append(nodeEval + edgeEval)
                    eval_sum += nodeEval + edgeEval

                if eval_sum == 0:
                    (ant.edgeChosen, ant.nodeChosen) = random.choice(neighboursRoutes)
                else:
                    probabilities_new = [0 if eval_sum == 0 else float(prob / eval_sum) for prob in probabilities]
                    index = full_probability(probabilities_new)
                    (ant.edgeChosen, ant.nodeChosen) = neighboursRoutes[index]

            ant.currentNode = ant.nodeChosen
            ant.lifespan = ant.lifespan - 1

            if ant.currentNode.type != NodeType.sense:
                depositedOdour = random.sample(ant.odour, len(ant.odour))
                pos = int(len(depositedOdour) * deltav)
                depositedOdour = depositedOdour[:pos]
                for elem in depositedOdour:
                    if len(ant.currentNode.odour) < odour_length:
                        if random.random() < 0.5 or len(ant.currentNode.odour) == 0:  # TODO
                            ant.currentNode.odour.append(elem)
                        else:
                            ant.currentNode.odour[random.randrange(0, len(ant.currentNode.odour))] = elem
                    else:
                        ant.currentNode.odour[random.randrange(0, 100)] = elem

        for ant in ants_list:
            ant.edgeChosen = ant.edgeChosen.pheromone + pheromone_deposit  # update pheromone
            ant.energy = ant.energy + min(ant.nodeChosen.energy, Ea)  # update energy
            if ant.currentNode.parent != ant.nest.parent and ant.currentNode.type == NodeType.sense:
                new_bridge = Edge(ant.currentNode, ant.nest, EdgeType.bridge)
                edges_list.append(new_bridge)

        for edge in edges_list:
            edge.change_pheromone()


def pos_map(pos_):
    first_letter = pos_[0].lower()
    if first_letter == 'j':
        return 'a'
    return first_letter

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

def extract_sentences_from_xml(xml_path, keys_dict):
    with open(xml_path, 'r') as fin:
        content = fin.read()
        tree = etree.fromstring(content)

    dataset = {}
    for text_id, elem in enumerate(tree.findall('text')):
        text_dicts = []
        sentences = []
        for index,sent in enumerate(elem.findall('sentence')):
            sent_dicts = []
            for wf_elem in sent.findall('instance'):
                wf_atributes = wf_elem.attrib
                instance_id = wf_atributes.get("id")
                
                syn_name = keys_dict.get(instance_id).name()
                lemma, pos_nltk, wasn = syn_name.split('.')

                sent_dicts.append({
                    "pos": wf_atributes.get("pos", ""),
                    "lemma": lemma,
                    "pos_nltk": pos_nltk,
                    "wnsn": wasn
                })
            dataset["sentence"+str(index)] = sent_dicts

        # text_key = f'senseval2_{text_id}'
        # dataset[text_key] = text_dicts
    return dataset

def get_senseval_dataset():
    dataset = {}
    version = 2
    xml_path = os.path.join('dataset',f'senseval{version}', f'senseval{version}.data.xml')
    keys_path = os.path.join('dataset',f'senseval{version}', f'senseval{version}.gold.key.txt')

    with open(keys_path, 'r') as fin:
        keys_list = fin.read().split('\n')
    
    keys_dict = {}
    for elem in keys_list[:-1]:
        instance_id, sense_key = elem.split(' ')[:2]
        keys_dict[instance_id] = synset_from_key(sense_key)
    
    senseval_set = extract_sentences_from_xml(xml_path, keys_dict)
    dataset.update(senseval_set)
    return dataset

def print_output(final_senses, test_results):
    print("Final senses:")
    with open('final_senses.txt', 'w') as f:
        for item in final_senses:
            f.write("%s\n" % item + '\n')

    print("Test results:")
    with open('test_results.txt', 'w') as f:
        for item in test_results:
            f.write("%s\n" % item + '\n')


def run(dataset):
    start_time = datetime.datetime.now()
    test_results = list()

    global nodes_list
    global edges_list

    # Create graph
    root = Node(None, None, NodeType.text)
    nodes_list.append(root)
    for key in dataset.keys():
        entry = dataset[key]
        sentence = Node(None, root, NodeType.sentence)
        nodes_list.append(sentence)
        edges_list.append(Edge(root, sentence, EdgeType.edge))
        for word in entry:
            # if str(word["wnsn"]) == "0" or str(word["lemma"]) == "":
            #     test_results.append("0")
            # else:
            test_results.append(str(word["lemma"]) + "." + str(word["pos_nltk"]) + "." + str(word["wnsn"]))
            # if(len(wordnet.synsets(word)) != 0): #TODO
            word_node = Node(None, sentence, NodeType.word)
            edges_list.append(Edge(sentence, word_node, EdgeType.edge))
            nodes_list.append(word_node)
            for sense in wordnet.synsets(word["lemma"], pos=word['pos_nltk']):
                sense_node = Node(sense, word_node, NodeType.sense)
                edges_list.append(Edge(word_node, sense_node, EdgeType.edge))
                nodes_list.append(sense_node)
    # root.printT(0)

    # Scenario
    iterate()

    # Print Path
    final_senses = list()
    words = [node for node in nodes_list if node.type == NodeType.word]
    for word in words:
        if word in STOPWORDS:
            final_senses.append("0")
        else:
            if len(word.children) == 0:
                final_senses.append("0")
            else:
                nest = max([nest for nest in word.children], key=lambda nest: nest.energy)
                sense = nest.sense.name()  # .split(".")[0] #nest.sense.name()
                final_senses.append(sense)

    # print_output(final_senses,test_results)

    acc = accuracy_score(final_senses, test_results)
    f1 = 0
    #f1 = f1_score(final_senses, test_results, average='micro')
    run_time = datetime.datetime.now() - start_time

    print("F1_Score: ")
    print(acc)

    # out_file = open("test_results.json", "w")
    # json.dump(test_results, out_file, indent = 4)
    # out_file.close()

    # out_file = open("final_senses.json", "w")
    # json.dump(final_senses, out_file, indent = 4)
    # out_file.close()

    print("--- %s Time ---" % (run_time))
    return {"Acc": acc, "F1": f1, "run_time": run_time}


# def main():
#     global nodes_list
#     global edges_list

#     xml_path = os.path.join('dataset','semeval2007')
#     total_acc = 0
#     total_f1 = 0
#     total_runtime = 0
#     files_number = 0
#     for index, filename in enumerate(os.listdir(xml_path)):
#         if (index == 200):
#             break  # TODO for test purpose only
#         file_path = os.path.join(xml_path, filename)
#         nodes_list = []
#         edges_list = []
#         if os.path.isfile(file_path):
#             dataset = extract_sentences_from_xml(file_path)
#             results = run(dataset)
#             files_number = files_number + 1
#             total_acc += results["Acc"]
#             total_f1 += results["F1"]
#             print("--- Run Time ---")
#             print(results["run_time"])

#     print("F1_Score: ")
#     print(float(total_acc / files_number))
    # print("F1 Score: ")


def main():
    global nodes_list
    global edges_list

    dataset = get_senseval_dataset()

    # out_file = open("dataset.json", "w")
    # json.dump(dataset, out_file, indent = 4)
    # out_file.close()

    results = run(dataset)
    print("--- Run Time ---")
    print(results["run_time"])
    print("F1_Score: ", results["Acc"])


if __name__ == "__main__":
    main()

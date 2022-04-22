import random
import numpy
from functools import reduce
from improved_lesk import lesk
import enum

EVAPORATE_RATE = 0.9 #TODO
DEPOSIT_RATE = 0.2 #TODO
deltav = 0.9
E_max = 60
E_0 = 30
omega = 25 # ant life duration
Ea = 16
max_iterations = 500 #CYCLES
nodes_list = list()
ants_list = list()
edges_list = list()
nests_list = list()
bridges_list = list()
odour_length = 100
#nodes_neighbours = dict() #Node,set([Node])

class NodeType(enum.Enum):
    sense = 1
    word = 2
    sentence = 3
    text = 4

class EdgeType(enum.Enum):
    edge = 1
    bridge = 2

class Node:
    def __init__(self, data, parent, typeN):
        self.type = typeN
        self.energy = 30
        if(self.type == NodeType.sense):
            self.odout = #TODO
        else:
           # self.energy = random.randint(5,60)
            self.odour = []
        self.parent = parent 
        self.children = []
        self.data = data

    def reduce_energy(self):
        self.energy = self.energy - 1

    def insert(self, node):
        self.children.append(node)
        node.setParent(self)

    def setParent(self,node):
       self.parent = node
    
    def produce_ant(self):
        return (numpy.arctan(self.energy))/numpy.pi + 0.5

    def eval_f(self, sense):
        sense_sum = reduce(lambda x : lesk(x,SENTENCE)[0] + lesk(y,SENTENCE)[0] , self.senses)       
        return lesk(sense,SENTENCE)/sense_sum  

# class Nest:
#     def __init__(self, senses, parent):
#         self.senses = senses
#         self.parent = parent
#         self.energy = 0
    
#     def eval_f(self, sense):
#         sense_sum = reduce(lambda x : lesk(x,SENTENCE)[0] + lesk(y,SENTENCE)[0] , self.senses)       
#         return lesk(sense,SENTENCE)/sense_sum    

class Ant:
    def __init__(self, energy, odour, node: Node):
        self.direction = 1
        self.lifespan = omega
        self.energy = energy
        self.odour = odour  #senses words
        self.currentNode = node
        self.nest = node
        self.nodeChosen = None
        self.edgeChosen = None

    def change_direction(self):
        self.direction = self.direction * (-1)

    def should_return(self):
        r = random.randint(0,100)
        return r < (self.energy/E_max * 100)

    def is_dead(self):
        self.lifespan == 0

    def move_ant(self): 
        if self.currentNode.type != NodeType.sense:
            #set odour on node
            odour_deposited = random.shuffle(list(self.odour))
            odour_deposited = odour_deposited[:len(odour_deposited) * DEPOSIT_RATE]
            for od in odour_deposited:
                if(len(self.currentNode.odour) < odour_length):
                    if random.random() < 0.5:
                        self.currentNode.odour.append(od)
                    else:
                        self.currentNode.odour[random.randint(0,len(self.currentNode.odour))] = od
                else:
                    self.currentNode.odour[random.randint(0,len(self.currentNode.odour))] = od

class Edge:    
    def __init__(self, source: Node, dest: Node, edgeType: EdgeType):
        self.source = source
        self.dest = dest
        self.pheromone = 0.0
        self.edgeType = edgeType

    def eval_f(self):
        return 1 - self.pheromone #create bridge when 0

    def change_pheromone(self):
        self.pheromone = (1 - EVAPORATE_RATE) * self.pheromone

    
def fitness(sentence):
    words = list(sentence.split(" "))
    configuration_sum = 0
    for word in words:
        configuration_sum += lesk(word,sentence)
    return configuration_sum

def fitness_at_pos(pos,sentence):
    configuration_sum = 0
    words = list(sentence.split(" "))
    return lesk(words[pos],sentence)

def full_probability(probabilities):
    r = random.random()
    sum = 0 
    for ip,prob in enumerate(probabilities):
        sum += prob
        if(r < prob):
            return ip

def get_neighbours(node: Node):
    #return nodes_neighbours.get(node)
    #newNeighbours = set()
    # if(node.type == NodeType.sense):
    #     for nest in nests_list:
    #         if nest.parent != node.parent:
    #             newEdge = Edge(node,nest,EdgeType.bridge)
    #             newNeighbours.append(newEdge,nest)

    return set([(edge,edge.dest) for edge in edges_list if edge.source == node]).union([(edge,edge.source) for edge in edges_list if edge.dest == node])

def itereaza():
    for i in range(0,max_iterations):
        for ant in [ant for ant in ants_list if ant.is_dead == True]:
            ant.currentNode.energy = ant.currentNode.energy + ant.energy
        ants_list = [ant for ant in ants_list if ant.is_dead == False]
        edges_list = [edge for edge in edges_list if (edge.pheromone != 0 or edge.edgeType != EdgeType.bridge]

        for node in nodes_list:
            if node.produce_ant() > random.random():
                ant = Ant(node)
                node.reduce_energy()
                ants_list.append(ant)
        
        probabilities = list()
        for ant in ants_list:
            if ant.direction == 1:
                if(ant.should_return()):
                    ant.change_direction()
            eval_sum = 0
            neighboursRoutes = get_neighbours(ant.currentNode) # TODO
            if ant.direction == 1:
                energy_sum = sum([node.energy for _ , node in neighboursRoutes])
                for (edge,node) in neighboursRoutes:
                    nodeEval = node.energy / energy_sum
                    edgeEval = 1 - edge.pheromone
                    probabilities.append(nodeEval + edgeEval)
                    eval_sum += nodeEval + edgeEval
                probabilities_new = [prob/eval_sum for prob in probabilities]
                index = full_probability(probabilities_new)
                (ant.edgeChosen, ant.nodeChosen) = neighboursRoutes[index]
            else:
                suma = sum([lesk_distance(wordnet.synsets(node.odour),wordnet.synsets(ant.odour)) for _ , node in neighboursRoutes])
                for (edge,node) in neighboursRoutes:
                    #synsets_edge = wordnet.synsets(edge.odour)
                    #synsets_node = wordnet.synsets(node.odour)
                    edgeEval = edge.pheromone
                    nodeEval = lesk_distance(wordnet.synsets(node.odour) , wordnet.synsets(ant.odour)) / suma
                    probabilities.append(nodeEval + edgeEval)
                    eval_sum += nodeEval + edgeEval

                probabilities_new = [prob/eval_sum for prob in probabilities]
                index = full_probability(probabilities_new)
                (ant.edgeChosen, ant.nodeChosen) = neighboursRoutes[index]

            ant.currentNode = ant.nodeChosen
            ant.lifespan = ant.lifespan - 1
        
        for ant in ants_list:
            ant.edgeChosen = ant.edgeChosen.pheromone * (1 + DEPOSIT_RATE)
            if ant.currentNode.parent != ant.nest.parent and ant.currentNode.type == NodeType.sense:
                new_bridge = new Edge(ant.currentNode,ant.nest,EdgeType.bridge)
                # nodes_neighbours.get(ant.currentNode).append(ant.nest)
                # nodes_neighbours.get(ant.nest).append(ant.currentNode)
                edges_list.append(new_bridge)

        for edge in edges_list:
            edge.change_pheromone()

        #TODO Update odour

import os
import pdb
import xml.etree.ElementTree as etree
import numpy as np

from nltk.corpus import wordnet, wordnet_ic
from tqdm import trange
from improved_lesk import lesk_distance
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

def main():
    xml_path = os.path.join('semcor', 'semcor', 'brown1', 'tagfiles', 'br-a01.xml')
    dataset = extract_sentences_from_xml(xml_path)

    for entry in dataset:
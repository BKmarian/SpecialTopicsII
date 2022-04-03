import random
import numpy
from functools import reduce
from improved_lesk import lesk

CYCLES = 500
EVAPORATE_RATE = 0.1 #TODO
DEPOSIT_RATE = 0.1 #TODO
SENTENCE = ""
E_max = 60
#odour_vector_length = 100 not needed i suppose

class Node:
   def __init__(self, data, parent):
      self.energy = random.randint(5,60)
      self.parent = parent 
      self.nodes = []
      self.data = data
      self.odour = []

   def insert(self, node):
      self.nodes.append(node)
      node.setParent(self)

   def setParent(self,node):
       self.parent = node

   def printTree(self):
      print(self.data)
      for node in self.nodes:
          if isinstance(node,Nest) == False:
            self.node.printTree()
    
    def produce_ant(self):
        return (numpy.arctan(self.energy))/numpy.pi + 0.5

class Nest:
    def __init__(self, senses, parent):
        self.senses = senses
        self.parent = parent
        self.energy = 0
    
    def eval_f(self, sense):
        sense_sum = reduce(lambda x : lesk(x,SENTENCE)[0] + lesk(y,SENTENCE)[0] , self.senses)       
        return lesk(sense,SENTENCE)/sense_sum    

class Ant:
    def __init__(self, energy, odour):
        self.lifespan = random.randint(1,30)
        self.energy = energy
        self.odour = odour  #senses words

    def eval_f(self):
        sense_sum = reduce(lambda x : lesk(x,SENTENCE)[0] + lesk(y,SENTENCE)[0] , self.odour)       
        return lesk(sense,SENTENCE)/sense_sum    

    def should_return(self):
        r = random.randint(0,100)
        return r < (self.energy/E_max * 100)

class Edge    
    def __init__(self, node: Node,nest: Nest):
        self.node = node
        self.nest = nest
        self.pheromone = 0

    def eval_f(self):
        return 1 - self.pheromone #create bridge when 0

    def change_pheromone(self):
        self.pheromone = (1 - EVAPORATE_RATE) * self.pheromone

def move_ant(a:Ant , n:Node): 

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



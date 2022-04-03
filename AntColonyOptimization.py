class Node:
   def __init__(self, data, parent):
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

class Nest:
    def __init__(self, senses, parent):
        self.senses = senses
        self.parent = parent
        self.energy = 0

    
class Ant:
    def __init__(self, energy, senses , odour):
        self.energy = energy
        self.odour = odour
        self.senses = senses #sysnset
    
class Bridge    
    def __init__(self, node: Node,nest: Nest):
        self.node = node
        self.nest = nest
        self.pheromone = 0
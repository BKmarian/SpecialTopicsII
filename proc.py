import os
import pdb
import xml.etree.ElementTree as etree

# 'a', 'r', 's', 'n', 'v'
def pos_map():
    # TODO: map to POS wordnet
    pass

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
            if wf_atributes['cmd'] != 'ignore':
                sentence.append({
                    "pos": wf_atributes["pos"],
                    "lemma": wf_atributes.get("lemma", ""),
                    "wnsn": wf_atributes.get("wnsn", "")
                })
        dataset.append(sentence)
    return dataset


xml_path = os.path.join('semcor', 'semcor', 'brown1', 'tagfiles', 'br-a01.xml')
dataset = extract_sentences_from_xml(xml_path)
pdb.set_trace()


# TODO: use this as demo
# tree=wordnet.synset('tree.n.01')
# tree.path_similarity(wordnet.synset('plant.n.01'))

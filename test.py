import xml.etree.ElementTree as etree
import os

def pos_map(pos_):
    first_letter = pos_[0].lower()
    if first_letter == 'j':
        return 'a'
    return first_letter
    
def extract_sentences_from_xml():
    xml_path = os.path.join('dataset','senseval2','senseval2.data.xml')
    with open(xml_path, 'r') as fin:
        content = fin.read()
        tree = etree.fromstring(content)

    dataset = []
    # contextfile > context > p > s
    context = tree#.find('corpus')
    for i, elem in enumerate(context.findall('text')):
        #sentence_elem = elem.find('s')
        for i, sent in enumerate(elem.findall('sentence')):
            sentence = []
            for wf_elem in sent.findall('wf'):
                wf_atributes = wf_elem.attrib
                wf_lemma = wf_atributes.get("lemma", "")
                if wf_atributes["pos"] != "NNP" and wf_lemma != "":
                    pos_ = wf_atributes["pos"]
                    pos_nltk = pos_map(pos_)

                    sentence.append({
                        "pos": pos_,
                        "pos_nltk": pos_nltk,
                        "lemma": wf_lemma,
                        "wnsn": wf_atributes.get("wnsn", "")
                    })
            dataset.append(sentence)
    return dataset

print(extract_sentences_from_xml())
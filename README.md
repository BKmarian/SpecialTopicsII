# SpecialTopicsII

## Ant Conlony vs Firefly (Accuracy)

| Algorithm | Semcor | Senseval2 | Senseval3 | 
| :---: | :---: | :---: | :---: |
| Firefly | 17.30% | 44.71% | 40.99% |
| Ant Colony | 40,46% | 43,86% | 41,63% | (with spacy embeddings)

Before running them, please download the datasets in the folders: semcor, senseval2 and senseval3. Afterwards, you can run in a terminal:
```
python AntColonyOptimization.py
python FireflyOptimization.py
```
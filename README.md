# Detecting Causal Language Use in Science Findings

This is a collaborative project with the School of Information at Syracuse University. The goal of the project is to develop an automated NLP model that can identify causal language use in science findings, and to further study whether causal language use differs by countries and languages. We have developed an annotated corpus and trained a BioBert model with 0.88 macro-F1 score to categorize conclusion sentences into direct causal, conditional causal, correlational, and no relationships. We then applied this model to the observational studies in PubMed, and observed different levels of causal language use by authors from different countries and language backgrounds. This result challenges the notion of a shared consensus on causal language use in global science community.

### How to cite ###
Yu, B., Li, Y. and Wang, J. (2019). Detecting Causal Language Use in Science Findings. EMNLP 2019, pages 4656–4666, Hong Kong, China, November 3–7, 2019. [PDF](https://www.aclweb.org/anthology/D19-1473.pdf)

```
@inproceedings{yu2019EMNLPCausalLanguage,
  title={Detecting Causal Language Use in Science Findings},
  author={Yu, Bei and Li, Yingya and Wang, Jun},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={4656-4666},
  year={2019},
  url={https://www.aclweb.org/anthology/D19-1473.pdf}
}
```

## Usage

STEP 1: Install bert-sklearn from [https://github.com/junwang4/bert-sklearn-with-class-weight](https://github.com/junwang4/bert-sklearn-with-class-weight) (for handling imbalanced classes)


STEP 2: Clone this repo and run

    git clone https://github.com/junwang4/causal-language-use-in-science
    cd causal-language-use-in-science
    python3 main.py 


## Performance
On Ubuntu 16.04 with a GPU of 1080TI (your performance numbers may be different but should be similar)

5-fold; 5 epochs; BioBERT

```
       Acc     F1   F1_0   F1_1   F1_2   F1_3      P    P_0  ...    P_3      R    R_0    R_1    R_2    R_3  size  weight
0    0.876  0.859  0.885  0.853  0.804  0.893  0.848  0.911  ...  0.868  0.876  0.860  0.818  0.907  0.920   614   0.201
1    0.905  0.896  0.905  0.892  0.864  0.921  0.887  0.930  ...  0.908  0.905  0.882  0.919  0.884  0.935   613   0.200
2    0.902  0.889  0.902  0.892  0.843  0.919  0.897  0.901  ...  0.907  0.882  0.904  0.879  0.814  0.930   613   0.200
3    0.935  0.914  0.945  0.907  0.854  0.952  0.915  0.945  ...  0.964  0.914  0.945  0.939  0.833  0.940   611   0.200
4    0.880  0.855  0.907  0.821  0.804  0.889  0.849  0.893  ...  0.915  0.866  0.923  0.796  0.881  0.864   610   0.199
avg  0.900  0.883  0.909  0.873  0.834  0.915  0.879  0.916  ...  0.912  0.889  0.903  0.870  0.864  0.918   612   0.200
time used: 916s
```

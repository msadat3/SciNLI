# SciNLI: A Corpus for Natural Language Inference on Scientific Text
This repository contains the dataset released in the ACL 2022 paper "SciNLI: A Corpus for Natural Language Inference on Scientific Text". **The dataset can be downloaded from [here](https://drive.google.com/drive/folders/1kjBTVBV1HlMWW5xK8V096LahsU3pULHU?usp=sharing).** The code for loading the dataset and running experiments for baseline models will be made available soon.

**If you face any issues while downloading the dataset, raise an issue in this repository or contact us at msadat3@uic.edu.**

## Abstract
Existing Natural Language Inference (NLI) datasets, while being instrumental in the advancement of Natural Language Understanding (NLU) research, are not related to scientific text. In this paper, we introduce SciNLI, a large dataset for NLI that captures the formality in scientific text and contains 107,412 sentence pairs extracted from scholarly papers on NLP and computational linguistics. Given that the text used in scientific literature differs vastly from the text used in everyday language both in terms of vocabulary and sentence structure, our dataset is well suited to serve as a benchmark for the evaluation of scientific NLU models. Our experiments show that SciNLI is harder to classify than the existing NLI datasets. Our best performing model with XLNet achieves a Macro F1 score of only 78.18% and an accuracy of 78.23% showing that there is substantial room for improvement.

## Dataset Description
We derive [SciNLI](https://drive.google.com/drive/folders/1kjBTVBV1HlMWW5xK8V096LahsU3pULHU?usp=sharing) from the papers published in the ACL anthology on NLP and computational linguistics. Specifically, we extract sentence pairs from papers published between 2000 and 2019 to create our training set and papers published in 2020 for our test and develpment sets.

For annotating the sentence pairs for our training set, we employ our distant supervision method which makes use of linking phrases indicative of the semantic relation between the sentences they occur in and their respective previous sentences. We train our models on these potentially noisy sentence pairs. However, for a realistic evaluation benchmark, we manually annotate the sentence pairs for the test and development sets. We refer the reader to our [paper](https://arxiv.org/pdf/2203.06728.pdf) for an in-depth description of our dataset construction process. 

### Examples
![Alt text](Images/Examples.png?raw=False "Title")

### Files

  => train.csv, test.csv and dev.csv contain the training, testing and development data, respectively. Each file has three columns: 
  
    * 'sentence1': the premise of each sample.
    
    * 'sentence2': the hypothesis of each sample.
    
    * 'label': corresponding label representing the semantic relation between the premise and hypothesis. 


  => train.jsonl, test.jsonl and dev.jsonl contain the same data as the CSV files but they are formatted in a json formal similar to SNLI and MNLI. Precisely, each line in these files is a json dictionary where the keys are 'sentence1', 'sentence2' and 'label' with the premise, hypothesis and the label as the values.
  
  
  
### Dataset Size

  * Train: 101,412 - automatically annotated.

  * Test: 4,000 - human annotated.

  * Dev: 2,000 - human annotated.

  * Total: 107,412.

## Citation
If you use this dataset, please cite our paper:

```
@inproceedings{sadat-caragea-2022-scinli,
    title = "{S}ci{NLI}: A Corpus for Natural Language Inference on Scientific Text",
    author = "Sadat, Mobashir  and
      Caragea, Cornelia",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.511",
    pages = "7399--7409",
}
```
## License
SciNLI is licensed with Attribution-ShareAlike 4.0 International [(CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

## Contact
Please contact us at msadat3@uic.edu with any questions.

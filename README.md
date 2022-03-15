# SciNLI
This repository contains the dataset released in the ACL 2022 paper "SciNLI: A Corpus for Natural Language Inference on Scientific Text". The code for loading the dataset and running experiments for baseline models will be made available soon.

If you use this dataset, please cite our paper:

```
@inproceedings{sadat-caragea-2022-SciNLI,
        title = "SciNLI: A Corpus for Natural Language Inference on Scientific Text",
        author = "Sadat, Mobashir  and
          Caragea, Cornelia",
        booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        year = "2022",
        address = "Dublin, Ireland",
        publisher = "Association for Computational Linguistics",
    }
```

## Dataset
The directory containing the benchmark train, test and development splits of the SciNLI dataset is available [here](https://drive.google.com/drive/folders/1kjBTVBV1HlMWW5xK8V096LahsU3pULHU?usp=sharing).

###Files###

  => train.csv, test.csv and dev.csv contain the training, testing and development data, respectively. Each file has three columns: 
  
    * 'sentence1': the premise of each sample.
    
    * 'sentence2': the hypothesis of each sample.
    
    * 'label': corresponding label representing the semantic relation between the premise and hypothesis. 


  => train.jsonl, test.jsonl and dev.jsonl contain the same data as the CSV files but they are formatted in a json formal similar to SNLI and MNLI. Precisely, each line in these files is a json dictionary where the keys are 'sentence1', 'sentence2' and 'label' with the premise, hypothesis and the label as the values.
  
  
  
###Dataset Size###

  * Train: 101,412 - automatically annotated.

  * Test: 4,000 - human annotated.

  * Dev: 2,000 - human annotated.

  * Total: 107,412.

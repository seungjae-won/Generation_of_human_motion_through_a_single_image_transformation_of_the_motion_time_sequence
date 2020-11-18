 -- This is an ongoing project and not the final result. --

## Improvement of data imbalance problem in human motion recognition
<br>
Using the ACGAN idear feature generator
<br><br>
<img src="https://github.com/seungjae-won/feature_generator__human_motion/blob/master/figure/model_figure.PNG" align="left" height="300" width="800" >

<br><br><br><br><br><br><br><br><br><br><br><br>


<h3>Abstract</h3>
Class imbalance problem of data degrades classification performance. The same goes for the field of human motion recognition. Through a feature generator using ACGAN idea, I'm trying to improve the data imbalance problem in the field of human motion recognition. To compare performance through over-sampling and weight balancing, which are used to solve traditional data imbalance problems. "MSRC-12" provided by Microsoft is used as the dataset (Other datasets will be used in the future)


### Dataset
Download-dataset : [MSRC-12 - download](https://www.microsoft.com/en-us/download/details.aspx?id=52283) <br>
Dataset document : [Reference document](https://nanopdf.com/download/this-document-microsoft-research_pdf)

There are a total of 12 classes, approximately 50 sequences for each class were used for training and 10 sequences were used for the test. Data sampling is used to solve the inherent problem of human motion recognition, the difference in sequence length for each motion. The odd-numbered class(total 6 classes) was trained by creating an artificial imbalance situation according to the experimental ratio to the number of balanced classes. 



### Method
<img src="https://github.com/seungjae-won/feature_generator__human_motion/blob/master/figure/proposed_method.PNG" align="left" height="300" width="800" >
<br><br><br><br><br><br><br><br><br><br><br><br><br><br>


### Results(Acciracu)
| Imbalane rate | 2 : 1  | 3 : 1 | 5 : 1 | 10 : 1 |
| :---------------------: |------------------------:|----------------------:|-----------------------:|------------------------:|
| Original LSTM      | 0.8483 ± 0.021 | 0.8050 ± 0.023 | 0.7733 ± 0.024 | 0.6300 ± 0.008 |
| Over-sampling(smoth)      | 0.8467 ± 0.008 | 0.8300 ± 0.008 | 0.7917 ± 0.025 | 0.6433 ± 0.009 |
| Weight-balancing      | **0.8917 ± 0.016** | 0.8100 ± 0.023 | 0.7333 ± 0.011 | **0.6633 ± 0.008** |
| **Proposed method**      | 0.8850 ± 0.006 | **0.8517 ± 0.006** | **0.8133 ± 0.004** | 0.6350 ± 0.014 |

                                                * Mean accuracy and standard deviation of 5 experiments

| Method | Average improvement | 
| :---------------------: |------------------------:|
| Over-sampling(smoth)  | 0.0137 |
| Weight-balancing      | 0.0104 |
| **Proposed method**      | **0.0321** |

<img src="https://github.com/seungjae-won/feature_generator__human_motion/blob/master/figure/confusion_matrix.PNG" align="left" height="300" width="800" >
<br><br><br><br><br><br><br><br><br><br><br><br><br><br>

### Discussion
1. Features that have passed LSTM may not be suitable
2. Concerns about problems with different sequence lengths for each motion of human motion data
3. Regardless of the data imbalance, if there is not much data itsself, the performance is severely degraded.

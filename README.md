## Generation of human behavior using ACGAN
<br>
Using the ACGAN idear feature generator
<br><br>
<img src="https://github.com/seungjae-won/Generation-of-human-behavior-using-ACGAN/blob/master/ex_result/ACGAN/figure/real_motion_data.gif" height="300" width="300>
<br><br><br><br><br><br><br><br><br><br><br><br>


<h3>Abstract</h3>
Class imbalance problem of data degrades classification performance. The same goes for the field of human motion recognition. Through a feature generator using ACGAN idea, I'm trying to improve the data imbalance problem in the field of human motion recognition. To compare performance through over-sampling and weight balancing, which are used to solve traditional data imbalance problems. "MSRC-12" provided by Microsoft is used as the dataset (Other datasets will be used in the future)


### Dataset
Download-dataset : [MSRC-12 - download](https://www.microsoft.com/en-us/download/details.aspx?id=52283) <br>
Dataset document : [Reference document](https://nanopdf.com/download/this-document-microsoft-research_pdf)

There are a total of 12 classes, approximately 50 sequences for each class were used for training and 10 sequences were used for the test. Data sampling is used to solve the inherent problem of human motion recognition, the difference in sequence length for each motion. The odd-numbered class(total 6 classes) was trained by creating an artificial imbalance situation according to the experimental ratio to the number of balanced classes. 

### Discussion
1. Features that have passed LSTM may not be suitable
2. Concerns about problems with different sequence lengths for each motion of human motion data
3. Regardless of the data imbalance, if there is not much data itsself, the performance is severely degraded.
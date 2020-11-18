 -- This is an ongoing project and not the final result. --

<h1>Generation of human behavior using ACGAN</h1>

<br><br>
<h3>   &nbsp&nbsp&nbsp&nbsp &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp &nbsp&nbsp Real motion data   
 &nbsp&nbsp&nbsp&nbsp &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp &nbsp&nbsp&nbsp&nbsp&nbsp  Fake motion data </h3>
<p>
<img src="https://github.com/seungjae-won/Generation-of-human-behavior-using-ACGAN/blob/master/ex_result/ACGAN/figure/real_motion_data.gif" align="left" height="300" width="300" >
<img src="https://github.com/seungjae-won/Generation-of-human-behavior-using-ACGAN/blob/master/ex_result/ACGAN/figure/fake_motion_data.gif" align="middle" height="300" width="300" >
 </p>

<h3>Abstract</h3>
Class imbalance problem of data degrades classification performance. The same goes for the field of human motion recognition. Through a feature generator using ACGAN idea, I'm trying to improve the data imbalance problem in the field of human motion recognition. To compare performance through over-sampling and weight balancing, which are used to solve traditional data imbalance problems. "MSRC-12" provided by Microsoft is used as the dataset (Other datasets will be used in the future)


### Dataset
Download-dataset : [MSRC-12 - download](https://www.microsoft.com/en-us/download/details.aspx?id=52283) <br>
Dataset document : [Reference document](https://nanopdf.com/download/this-document-microsoft-research_pdf)

There are a total of 12 classes, approximately 50 sequences for each class were used for training and 10 sequences were used for the test. Data sampling is used to solve the inherent problem of human motion recognition, the difference in sequence length for each motion. The odd-numbered class(total 6 classes) was trained by creating an artificial imbalance situation according to the experimental ratio to the number of balanced classes. 


### Method
<p>
<img src="https://github.com/seungjae-won/Generation-of-human-behavior-using-ACGAN/blob/master/figure/method_figure.PNG" align="left" height="300" width="800" >
<img src="https://github.com/seungjae-won/Generation-of-human-behavior-using-ACGAN/blob/master/figure/method_figure2.PNG" align="left" height="200" width="800" >
<br>
 </p>
 
<h3>Discussion</h3>
<h4>1. Features that have passed LSTM may not be suitable</h4>
<h4>2. Concerns about problems with different sequence lengths for each motion of human motion data</h4>
<h4>3. Regardless of the data imbalance, if there is not much data itsself, the performance is severely degraded.</h4>

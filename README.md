

<h1>Generation of human motion through a single image transformation of the whole motion time sequence</h1>

<br><br>
<h3>   &nbsp&nbsp&nbsp&nbsp &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp &nbsp&nbsp Real motion data   
 &nbsp&nbsp&nbsp&nbsp &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp &nbsp&nbsp&nbsp&nbsp&nbsp  Generated motion data </h3>
<p>
<img src="https://github.com/seungjae-won/Generation-of-human-behavior-using-ACGAN/blob/master/ex_result/ACGAN/figure/real_motion_data.gif" align="left" height="300" width="300" >
<img src="https://github.com/seungjae-won/Generation-of-human-behavior-using-ACGAN/blob/master/ex_result/ACGAN/figure/fake_motion_data.gif" align="middle" height="300" width="300" >
 </p>
 
 <h3> ALL Generated results </h3>
https://github.com/seungjae-won/Generation_of_human_motion_through_a_single_image_transformation_of_the_motion_time_sequence/blob/master/ex_result/ACGAN/figure/README.md


<h3>Abstract</h3>
In the field of human behavior generation, the X,Y,Z axis of the skeleton data is changed to R,G,B channels respectively to convert to image generation problems. Convert the full Time sequence of actions to a single image for each point and image generation for each class of data through ACGAN. Having novelty in taht the creation of human behavior has been resolved by converting it into an image.


### Dataset
Download-dataset : [MSRC-12 - download](https://www.microsoft.com/en-us/download/details.aspx?id=52283) <br>
Dataset document : [Reference document](https://nanopdf.com/download/this-document-microsoft-research_pdf)

Microsoft Research Cambridge-12 (MSRC-12) besture data was used in the experiment. A total of 12 classes exist, and about 60 time sequence data exist for each class. There was a problem that the length of time sequence for each class was very different. For smooth model learning, the time sequence for each class was extracted to a total of 256 through Average sampling. In addition, 60 data were thought to be insufficient for learning to generate appropriate behavior, so 600 data for each class was used through data copy.


### Method
<p>


<img src="https://github.com/seungjae-won/Generation-of-human-behavior-using-ACGAN/blob/master/figure/xyzTOrgb.PNG" align="left" height="300" width="800" >
 <br><br> <br><br>
<img src="https://github.com/seungjae-won/Generation-of-human-behavior-using-ACGAN/blob/master/figure/jointTOimage.PNG" align="left" height="300" width="800" >
 <br><br> <br><br>

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
 </p>
 
<h3>Discussion</h3>
<h4>1. Information loss inevitably occurs when converting the entire time sequence of human behavioral data into an image, and we need to think about how to minimize it.</h4>
<h4>2. Concerns about problems with different sequence lengths for each motion of human motion data</h4>
<h4>3. Performance improvement is needed to the extent that each operation can be distinguished in detail.</h4>

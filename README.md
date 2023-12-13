# listen_to_your_heart <br>
#### Final Project Github Repo for Listen To Your Heart Group Project (CS5100 FA23)
_____________________

The Project focuses on developing eXplainable AI (XAI) methods to try and help explain/understand these state-of-the-art models. We accomplish this by using LSTM and CNN Machine Learning Models to make accurate predictions about congestive heart failure using ECGs and explaining the Neural Network output/prediction using the LIME Segment eXplainable AI tool. Thus, eliminating the ambiguity around most network predictions. <br>

_____________________________
## Steps To Run Code  <br>

### Data Load Set Up  <br>
> Load data from PyhsioNet  <br>
- The RR data data to be analyzed has already been placed into the listen_to_your_heart repo, however if a re-run is required, kindly run in the LSTM environment <br>
* Run the download.ipynb <br>
```
NOTE: All models have been pre-trained for efficiency purposes.
Training of the CNN and LSTM can be skipped if desired.
```
### CNN Code Set Up  <br>
#### Create Environment <br>
 
> :bulb: **Tip:** Create Environment using Python Version 3.8 (myenv) <br>
> :bulb: **Tip:** Use Windows Platform <br>

#### Run the following code: <br>
<ol>
  <li> Activate the environment: conda activate myenv </li>
  <li> Install the required packages: conda install --file requirements.txt -c conda-forge</li>
  <li> Run the generate_cnn.ipynb Jupyter Notebook </li>
</ol>

### LSTM Model Code Set Up 
#### Create Environment <br>
> :bulb: **Tip:** Use Windows Platform <br>
#### Run the following code: <br>
<ol>
  <li> Activate the environment: conda activate myenv </li>
  <li> Install the required packages: conda install --file lstm-requirements.txt -c conda-forge</li>
  <li> Run the lstm.ipynb Jupyter Notebook </li>
</ol>

### Lime Segmentation Implementation 
> :bulb: **Tip:** Use Windows Platform <br>

#### Run the following code: <br>
<ol>
  <li> Activate the environment: conda activate myenv </li>
  <li> Git clone https://github.com/rush17m/LIMESegment </li>
  <li> Install the required packages: conda install --file requirements_update.txt </li>
  Run:
  <li> Run the explain.ipynb Jupyter Notebook </li>
</ol>







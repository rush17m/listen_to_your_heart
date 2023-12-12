# listen_to_your_heart <br>
#### Final Project Github Repo for Listen To Your Heart Group Project (CS5100 FA23)
_____________________

The Project focuses on developing eXplainable AI (XAI) methods to try and help explain/understand these state-of-the-art models. We accomplish this by using LSTM and CNN Machine Learning Models to make accurate predictions about congestive heart failure using ECGs and explaining the Neural Network output/prediction using the LIME Segment eXplainable AI tool. Thus, eliminating the ambiguity around most network predictions.
_____________________________
## Steps To Run Code  <br>

### Data Load Set Up  <br>
> Load data from PyhsioNet  <br>
* Run the download.ipynb <br>

### CNN Code Set Up  <br>
#### Create Environment <br>
> :bulb: **Tip:** Create Environment using Python Version 3.8 <br>
> :bulb: **Tip:** Use Platform: osx-arm64. <br>

#### Run the following code: <br>
<ol>
  <li> Create Environment : conda create -n my_cnn_env -c conda-forge --platform osx-arm64 python=3.8 </li>
  <li> Activate the environment: conda activate my_lstm_env </li>
  <li> Install the required packages: conda install --file requirements.txt -c conda-forge</li>
  <li> Run the generate cnn.ipynb Jupyter Notebook </li>
</ol>

### LSTM Model Code Set Up 
#### Create Environment <br>
> :bulb: **Tip:** Create Environment using Python Version 3.10.13 <br>
> :bulb: **Tip:** Use Platform: osx-arm64. <br>
#### Run the following code: <br>
<ol>
  <li> Create Environment: conda create -n my_lstm_env -c conda-forge --platform osx-arm64 python=3.10 </li>
  <li> Activate the environment: conda activate my_lstm_env</li>
  <li> Install the required packages: conda install --file lstm-requirements.txt -c conda-forge</li>
  <li> Run the lstm.ipynb Jupyter Notebook </li>
</ol>

### Lime Segmentation Implementation 
> :bulb: **Tip:** Create Environment using Python Version 3.8 <br>
> :bulb: **Tip:** Use Platform: osx-arm64. <br>

#### Run the following code: <br>

<ol>
  <li> Create Environment: conda create -n my_lime_env -c conda-forge --platform osx-arm64 python=3.8 </li>
  <li> Activate the environment: conda activate my_lime_env </li>
  <li> Install the required packages: listen_to_your_heart/LIMESegment/requirements.txt </li>
  Run:
  <li> Run the explain.ipynb Jupyter Notebook </li>
</ol>

```python
  from Utils.explanations import LIMESegment
  explanations = LIMESegment(ts, model, model_type, distance, window_size, cp, f)
```




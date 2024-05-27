# UniMEL: A Unified Framework for Multimodal Entity Linking with Large Language Models

This repository is the official implementation for the paper titled "UniMEL: A Unified Framework for Multimodal Entity Linking with Large Language Models".

```html
<p align="center">
  <img src="framework.png" alt="unimel" width="640">
</p>
```





## Usage

#### Step 1: Install and set up environment

```python
pip install -r requirements.txt
conda create -n unimel python==3.8.18
conda activate unimel
```



#### Step 2: Dataset

You may download WikiMEL and RichpediaMEL from https://github.com/seukgcode/MELBench and WikiDiverse from https://github.com/wangxw5/wikiDiverse.

#### Step 3: Train

We provide a trained checkpoint "checkpoint-2200.zip", you just need to unzip it to the current directory and record its path.

If you want to train a new checkpoint, please refer to peft (https://github.com/huggingface/peft) or swift (https://github.com/modelscope/swift).



#### Step 4: Run

```
cd UniMEL
bash run.sh 0 wikidiverse  # for wikidiverse
```



## Code Structure

```python
├─code
│  │  main.py
│  │  
│  └─untils
│      │  dataset.py
│      │  functions.py
│      │  
│      └─__pycache__
│              dataset.cpython-38.pyc
│              functions.cpython-38.pyc
│              
└─config
        wikidiverse.yaml
│  framework.png
│  README.md
│  requirements.txt
│  run.sh
```

## Results

#### Main results

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Model Performance Table</title>
<style>
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th, td {
    border: 1px solid black;
    padding: 8px;
    text-align: center;
  }
  th {
    background-color: #f2f2f2;
  }
</style>
</head>
<body>
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="4">Richpedia</th>
      <th colspan="4">WikiMEL</th>
      <th colspan="4">Wikidiverse</th>
    </tr>
    <tr>
      <th>Top-1</th>
      <th>Top-5</th>
      <th>Top-10</th>
      <th>Top-20</th>
      <th>Top-1</th>
      <th>Top-5</th>
      <th>Top-10</th>
      <th>Top-20</th>
      <th>Top-1</th>
      <th>Top-5</th>
      <th>Top-10</th>
      <th>Top-20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BERT</td>
      <td>31.6</td>
      <td>42.0</td>
      <td>47.6</td>
      <td>57.3</td>
      <td>31.7</td>
      <td>48.8</td>
      <td>57.8</td>
      <td>70.3</td>
      <td>22.2</td>
      <td>53.8</td>
      <td>69.8</td>
      <td>82.8</td>
    </tr>
    <tr>
      <td>BLINK</td>
      <td>30.8</td>
      <td>38.8</td>
      <td>44.5</td>
      <td>53.6</td>
      <td>30.8</td>
      <td>44.6</td>
      <td>56.7</td>
      <td>66.4</td>
      <td>22.4</td>
      <td>50.5</td>
      <td>68.4</td>
      <td>76.6</td>
    </tr>
    <tr>
      <td>ARNN</td>
      <td>31.2</td>
      <td>39.3</td>
      <td>45.9</td>
      <td>54.5</td>
      <td>32.0</td>
      <td>45.8</td>
      <td>56.6</td>
      <td>65.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DZMNED</td>
      <td>29.5</td>
      <td>41.6</td>
      <td>45.8</td>
      <td>55.2</td>
      <td>30.9</td>
      <td>50.7</td>
      <td>56.9</td>
      <td>65.1</td>
      <td>-</td>
      <td>39.1</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>JMEL</td>
      <td>29.6</td>
      <td>42.3</td>
      <td>46.6</td>
      <td>54.1</td>
      <td>31.3</td>
      <td>49.4</td>
      <td>57.9</td>
      <td>64.8</td>
      <td>21.9</td>
      <td>54.5</td>
      <td>69.9</td>
      <td>76.3</td>
    </tr>
    <tr>
      <td>MEL-HI</td>
      <td>34.9</td>
      <td>43.1</td>
      <td>50.6</td>
      <td>58.4</td>
      <td>38.7</td>
      <td>55.1</td>
      <td>65.2</td>
      <td>75.7</td>
      <td>27.1</td>
      <td>60.7</td>
      <td>78.7</td>
      <td>89.2</td>
    </tr>
    <tr>
      <td>HieCoAtt</td>
      <td>37.2</td>
      <td>46.8</td>
      <td>54.2</td>
      <td>62.4</td>
      <td>40.5</td>
      <td>57.6</td>
      <td>69.6</td>
      <td>78.6</td>
      <td>28.4</td>
      <td>63.5</td>
      <td>84.0</td>
      <td>92.6</td>
    </tr>
    <tr>
      <td>GHMFC</td>
      <td>38.7</td>
      <td>50.9</td>
      <td>58.5</td>
      <td>66.7</td>
      <td>43.6</td>
      <td>64.0</td>
      <td>74.4</td>
      <td>85.8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>MMEL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>71.5</td>
      <td>91.7</td>
      <td>96.3</td>
      <td>98.0</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>CLIP</td>
      <td>60.4</td>
      <td>96.1</td>
      <td>98.3</td>
      <td>99.2</td>
      <td>36.1</td>
      <td>81.3</td>
      <td>92.8</td>
      <td>98.3</td>
      <td>42.4</td>
      <td>80.5</td>
      <td>91.7</td>
      <td>96.6</td>
    </tr>
    <tr>
      <td>DRIN</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>65.5</td>
      <td>91.3</td>
      <td>95.8</td>
      <td>97.7</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DWE</td>
      <td>67.6</td>
      <td>97.1</td>
      <td>98.6</td>
      <td>99.5</td>
      <td>44.7</td>
      <td>65.9</td>
      <td>80.8</td>
      <td>93.2</td>
      <td>47.5</td>
      <td>81.3</td>
      <td>92.0</td>
      <td>96.9</td>
    </tr>
    <tr>
      <td>DWE+</td>
      <td>72.5</td>
      <td>97.3</td>
      <td>98.8</td>
      <td>99.6</td>
      <td>72.8</td>
      <td>97.5</td>
      <td>98.9</td>
      <td>99.7</td>
      <td>51.2</td>
      <td>91.0</td>
      <td>96.3</td>
      <td>98.9</td>
    </tr>
    <tr>
      <td>UniMEL (ours)</td>
      <td>94.8</td>
      <td>97.9</td>
      <td>98.3</td>
      <td>98.8</td>
      <td>94.1</td>
      <td>97.2</td>
      <td>98.4</td>
      <td>98.9</td>
      <td>92.9</td>
      <td>97.0</td>
      <td>99.5</td>
      <td>99.8</td>
    </tr>
  </tbody>
</table>
</body>
</html>

```


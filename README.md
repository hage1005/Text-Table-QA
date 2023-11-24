# Text-Table-QA
This is the project containing source code for 11711 HW4
## Requirements
```
python==3.7
torch==1.7.1+cu110
transformers==4.21.1
sentence-transformers==2.2.2
openai==1.3.5
```
## Data prepare
Download all data from [here](https://drive.google.com/file/d/1aVoBWvAE2BBaO5a27xHpgOqKGWzUV0K5/view?usp=sharing) . 

Then `unzip Data.zip` .

Directly use `train.row.json`, `dev.row.json` and `test.row.json` since we only focus on the reader part.

## Training

Downloader reader [checkpoint](https://drive.google.com/file/d/1IWHY-_kLNyHKZBxenX-RDBwDwqjiD2Zg/view?usp=share_link) and put it under reader1 directory

Fill in your openai key to get generated passages in `read.py`.
```
client = OpenAI(
    api_key="Your API",
)
```
### Train reader
```
bash read.sh
```
## Testing
Since training is very time-consuming (~10 hours) and requires additional money budget, you can directly run test scripts with checkpoint.
```
bash read_dev.sh
bash read_test.sh
```
## Acknowledge
Parts of our code are adapted from existing source code for the paper [S3HQA](https://arxiv.org/abs/2305.11725)
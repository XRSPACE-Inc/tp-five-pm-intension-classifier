# tp-five-pm-intension-classifier

This project aims to correct typo errors. For example:

- Incorrect: "鍋水液是忠華電信董事長"
- Correct: "郭水義是中華電信董事長"

## Usage

To use the project, simply run the following script:
python infer_task_multi.py

## Training Data

The training data is sourced from CHT news, where some characters are intentionally replaced to create incorrect
sentences.

## Model Path

The pre-trained model can be accessed from the following link:
[Model Path](https://huggingface.co/google-bert/bert-base-multilingual-uncased/tree/main)

## References

1. [基於BERT與混淆發音對影片字幕轉換之專業詞校正](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id=%22109CCU00392012%22.&searchmode=basic)

2. [基於BERT的ASR糾錯](https://blog.csdn.net/qq_27590277/article/details/107398826)

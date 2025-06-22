# Enhancing Multilingual AI Analyzing and Modeling Educational Content with the FineWeb-C


This project uses pretrained BERT models to classify the quality of educational texts using data from huggingfaces fineweb-c and fineweb2. It compares the performance of deep learning approaches with a traditional readability score (LIX). We implement fine-tuning of BERT, custom loss functions, and evaluation metrics to explore model behavior in imbalanced data scenarios.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Installation
The project uses version python 3.13

Clone the repo:
```bash
git clone https://github.com/Shadoughcake/Enhancing-Multilingual-AI-Analyzing-and-Modeling-Educational-Content-with-the-FineWeb-C.git

pip install -r requirements.txt

 ```
 

## Usage

To train the model:
```bash
python train.py --model bert-base-uncased --lr 1e-5 --epochs 100

```




## Data

The dataset is located in `data/` and consists of educational texts labeled by quality. You can generate the LIX scores using `lix_score.py`.







## Model

We fine-tune a pretrained BERT model by adding a classification head and use dropout for regularization. For imbalanced data, we include class weighting and a custom L1-penalized loss to encourage semantically close predictions.





## Results

| Model | Accuracy | F1-score (weighted) | L1-score |
|-------|----------|------------------|----|
| danish-bert-botxo | 66.58% | 0.6525 | 0.4121|
| bert-base-multilingual-cased	| 58.65% | 0.5930 | 0.4851 |



Visualizations, confusion matrices, or charts can go here.




## License

This project is licensed under the MIT License. See `LICENSE` for details.



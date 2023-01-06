# Model Card
For additional information, see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model used for this projekt is a `RandomForestClassifier` from `scikit-learn`. All the hyperparameters are default parameters of the classifier.

## Intended Use
The model is used for binary classification: It predicts whether employees' salary is above $50,000 (classes: `<=50000` and `>50000`).

## Training Data
The data utilized for training this model came from the Census Bureau, and consists of salary information: https://archive.ics.uci.edu/ml/datasets/census+income

For both training and evaluation, categorical features of the data are encoded using `OneHotEncoder` and the target is transformed using `LabelBinarizer`.

## Evaluation Data
The original dataset is first pre-processed and then split into training (80%) and evaluation data (20%).

## Metrics
Precision, recall, and F_beta score were used as metrics for evaluating the model's performance. The model achieves the following result:
* Precision: 0.7321428571428571
* Recall: 0.6431372549019608
* F_beta: 0.6847599164926931

## Ethical Considerations
Some bias may be embedded within the data, particularly around race and ethnicity. This should be checked before depending on a model using demographic data as its inputs.

## Caveats and Recommendations
The dataset is a outdated sample and cannot adequately be used as a statistical representation of the population. It is recommended to use the dataset for training purpose on ML classification or related problems only.

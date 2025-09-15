# The minimal model

Machine Learning analysis for the dataset from Kaggle https://www.kaggle.com/datasets/madaraweerasingha/cinnamon-quality-classification

The dataset is straightforward and provides sufficient information for classification. With this in mind, the objective was to design the simplest possible model, guided by a detailed analysis of feature importance. In practical applications, this strategy lowers classification costs by reducing the number of variables required.

Given the dataset’s balanced nature and clear feature separability, a bootstraping strategy as a lightweight alternative to k-fold cross-validation was adopted. This approach efficiently validated model robustness while prioritizing interpretability and computational simplicity.

Three key findings emerged from the analysis:

- Volatile Oil is the most important feature. Using only this variable as discriminant, it is possible to classify low-quality samples without error.

- For classifying intermediate- and high-quality groups, it is essential to use a logistic regression model with the features Volatile Oil, Ash, Coumarin and Acid Insoluble Ash.

- Despite chromium’s low correlation with other variables—suggesting potential complementary value—its inclusion reduces model robustness.


# Reproducibility

Create a conda environment (conda version 25.1.1)

```
conda env create -f environtment.yaml
conda activate cinnamon-classifier
```

# Create HTML and PDF files

Running these commands html and pdf file will be created (quarto version 1.8.24)

```
quarto render minimalCinnamon.ipynb --to html
quarto render minimalCinnamon.ipynb --to pdf
```
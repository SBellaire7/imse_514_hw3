# imse_514_hw3
Homework assignment for IMSE 514. Uses logistic regression to classify instances of the Wisconsin Breast Cancer Diagnostic Dataset.

The file HW3_Bottom_up.m performs a bottom-up search through the entire feature space to identify the best features for the model one-by-one.
After running the bottom-up search, the HW3_Kfold.m script can then be run (with the desired features' indices entered in line 20 of the code).

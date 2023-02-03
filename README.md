# GSV Training Data Replication Repository README

This document summarizes the included datasets (uploaded to SDR) and code (located here in Github) to replicate and reproduce tables in Hwang et al.'s (2023) paper "Curating Training Data for Reliable Large-Scale Visual Data Analysis: Lessons from Identifying Trash in Street View Imagery". 

## Data
**single_image.csv**: This dataset includes all of the images that have single image ratings from Mechanical Turk or coding sessions and all images with Trueskill ratings or ML predictions. Each row corresponds to an image, with columns indicating the answer provided in each of these surveys and/or the relevant Trueskill rating and ML prediction. 

**pairs.csv**: This dataset includes all of the images that were rated as pairs in Mechanical Turk or coding sessions. Each row corresponds to a pair of images. 

For both datasets, the accompanying codebook (**SMR_Data_Codebook.xlsx**) details each column’s values and their meanings.

## Scripts
**results-input.Rmd** uses single_image.csv and pairs.csv to generate all of the tables and figures in the paper.

For similar datasets but other input data, **results-anydata.Rmd** generates generic reliability measures.

PCA results and cosine similarity for images with discrepancies, as described in the paper, are generated using **pca_image_feature_analysis.ipynb**. 

Class Activation Maps (CAMs) are generated using **cams.py**.

Discrepancies input for PCA/cosine similarity/CAMs can be generated by user from single_image.csv.

Note, image names do not indicate the location, but users can request the associated location from the authors.



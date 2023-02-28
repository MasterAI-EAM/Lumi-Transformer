# UNSW AI4Science - MasterAI
##  Solar Transformer

This repository contains code for Solar Transformer for electroluminescence (EL) images. The model is based on the transformer architecture and is designed to process EL images of solar cells. 

### Background

EL imaging is a technique used to study solar cells. It involves capturing images of solar cells using a camera sensitive to the near-infrared region of the electromagnetic spectrum. These images show the distribution of charge carriers in the solar cell, which is related to the efficiency of the cell.

The solar-transformer model is designed to process EL images and predict the efficiency of the solar cell. It is based on the transformer architecture, which has been shown to be effective for processing sequential data such as natural language text.

### Dependencies

The code is written in Python and requires the following dependencies:

PyTorch (version 1.9.0 or later)  
NumPy  
Pandas  
Matplotlib  
scikit-learn  

### Usage  

The code is organized as follows:

data.py: contains code for loading and preprocessing the EL images  
model.py: contains code for the solar-transformer model  
evaluate.py: contains code for evaluating the model on a test set  

### Results

The solar-transformer model achieves state-of-the-art performance on the EL image dataset, with an accuracy of 91.7% on a classfication test. (defective or functional)

### Acknowledgments

This work was supported by the Univeristy of New South Wales. We would also like to thank the support of GreenDyanmics Pty. Ltd.

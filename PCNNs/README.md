# Introduction

This repository contains the code and resources for predicting vegetation phenology using an innovative Physical Constraint Neural Networks (PCNNs) model. The project aims to enhance the accuracy of vegetation phenology predictions by integrating machine learning techniques with physical mechanisms, addressing the limitations of traditional methods that often rely solely on vegetation indices from satellite imagery.

# Project Overview
This repository contains the code and data for the PCNNs model, which uses the Moderate-Resolution Imaging Spectroradiometer (MODIS) dataset to predict vegetation phenology for four vegetation types in North America. The model incorporates meteorological variables and is validated by field observations from PhenoCam and the USA National Phenology Network (USA-NPN) spanning from 2001 to 2021.


# Installation and Setup
Before starting with the project, ensure you have the following prerequisites:

Python 3.X: Make sure you have Python installed on your system.
Required Libraries: Install the necessary libraries in requirement.txt
MODIS dataset

data: Contains the MODIS dataset and field-observed data from PhenoCam and USA-NPN.
model: Houses the PCNNs model architecture and training code.
prediction code and weight: Provides instructions for running predictions using the trained model.
train and test code: Contains code for evaluating the model's performance using field-observed data.

#clone the repository including submodules
>git clone --recursive https://github.com/caomy7/PCNNs.git
>cd PCNNs

#create new virtual environment and install requirements
>python -m venv venv
>source venv/bin/activate              #linux
>venv/Scripts/activate.bat                #windows
>pip install -r requirements.txt

#download pretrained models
>python pretrained_models.py

#run
>python main.py




# Visualization
Use the Jupyter notebooks in the notebooks directory to visualize the results and assess the model's performance.

# Citation
If you find this project useful, please cite our paper:
Cao, M., & Weng, Q. (2024). Embedded physical constraints in machine learning to enhance vegetation phenology prediction. GIScience & Remote Sensing, 61(1). https://doi.org/10.1080/15481603.2024.2426598

# Contact Information
For questions or further clarification about the project, please reach out via email at 22037133r@connect.polyu.hk. Your feedback and suggestions are welcome!



# PCNN
# PCNN

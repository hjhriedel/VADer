<!-- LOGO AND TITLE-->
<!-- <p align="right"><img src="https://github.com/ISMD-TU-Darmstadt/VADer/blob/main/utils/img/sdu-color-transparent.png" alt="Logo" width="80" height="80"></p> -->

<h1 align="center">
  <br>

  </center>
  <img src="https://github.com/ISMD-TU-Darmstadt/VADer/blob/main/utils/img/sdu-color-transparent.png" alt="SDU" width="500">
  <br>
  <br>
  <b>VADer</b>
  <br>
  <sub><sup>Virtual Axle Detector</sup></sub>
  <br>
</h1>

# Introduction
This is an implementation of the paper "Virtual Axle Detector based on Analysis of Bridge Acceleration Measurements by Fully Convolutional Network" using Python, TensorFlow and SciPy. The model detects train axles when passing over accelerometers installed anywhere on bridges.

The repository contains code for:
- Model creation with TensorFlow
- Custom metrics
- Training
- Evaluation 
- Data transformation

Data used in the article can be found here: https://doi.org/10.5281/zenodo.6782318 <br>
Script for data transformation from *.mat to *.txt is also available in the repository on zenodo above.

# Code Overview
The code is organised as follows:
- **configs:** Configurations for the training. The configurations correspond to the variations of gamma as in the above paper.
- **data_loader:** Folder for different variants of data loaders and transformation configurations. The data loader here is for training with all data in memory.
- **models:** Script for model creation and script for our custom metrics.
- **trainers:** Trainer for logging experiment on comet.ml.
- **utils:** Miscellaneous utility functions.
- **evaluate.py:** Script for evaluating models on test data and plotting results.
- **train.py:** Script for initiating training for each configuration sequentially.
- **transformData.py:** Script for transform the data from *.txt to final input.

# Affiliation and Funding
The project was developed in the Structural Dynamics Unit of the Institute for Structural Mechanics and Design, TU Darmstadt, Germany.

The research project ZEKISS (www.zekiss.de) is carried out in collaboration with the German railway company DB Netz AG, the Wölfel Engineering GmbH and the GMG Ingenieurgesellschaft mbH. It is funded by the mFund (mFund, 2020) promoted by the The Federal Ministry of Transport and Digital Infrastructure.

The research project DEEB-INFRA (www.deeb-infra.de) is carried out in collaboration with the the sub company DB Campus from the Deutschen Bahn AG, the AIT GmbH, the Revotec zt GmbH and the iSEA Tec GmbH. It is funded by the mFund (mFund, 2020) promoted by the The Federal Ministry of Transport and Digital Infrastructure.

<div style="background-color: white" align="center">
    <img align="center" src="https://github.com/ISMD-TU-Darmstadt/VADer/blob/main/utils/img/tud_logo.jpg" alt="TU Darmstadt" width="130" hspace="20"/>
    <img align="center" src="https://github.com/ISMD-TU-Darmstadt/VADer/blob/main/utils/img/Logo%20ISM%2BD%20Bildmarke.png" alt="ISMD" width="130" hspace="20"/>
    <img align="center" src="https://github.com/ISMD-TU-Darmstadt/VADer/blob/main/utils/img/mFUND.JPG" alt="mFUND" width="130" hspace="20"/>
    <img align="center" src="https://github.com/ISMD-TU-Darmstadt/VADer/blob/main/utils/img/BMVI_Fz_2017_Office_Farbe_de.png" alt="BMVI" width="100" hspace="20"/>
</div>

# Seismic Inversion with 1D Temporal Convolutional Network (TCN)

This project implements a 1D Temporal Convolutional Network (TCN) for seismic inversion, designed to predict subsurface models such as Acoustic Impedance (AI) or elastic properties from seismic data. The code is structured to train on two synthetic datasets: SEAM and Marmousi. It includes functions for preprocessing, training, and testing the model. The final results include visualizations and performance metrics.

## Project Structure

```bash
├── core/
│   ├── datasets.py         # Seismic dataset handling and preprocessing
│   ├── model1D.py          # MustafaNet TCN model architecture
│   ├── utils.py            # Utility functions (standardization, extraction)
├── data/
│   ├── poststack_seam_seismic.npy    # SEAM seismic data
│   ├── seam_elastic_model.npy        # SEAM elastic model data
│   ├── marmousi_synthetic_seismic.npy # Marmousi seismic data
│   ├── marmousi_Ip_model.npy         # Marmousi AI model data
├── saved_models/
│   ├── model_seam_1D.pth   # Trained model weights
├── README.md               # Project documentation
├── train_test.py           # Main script for training and testing
└── data.zip                # Compressed data (will be extracted automatically)

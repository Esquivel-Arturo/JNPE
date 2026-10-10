# Joint Neural Posterior Estimation and Model Selection for Chemical Reaction Networks

This repo contains all the code, simulation scripts, and data processing pipelines used to produce all results and figures shown in the preprint "Joint Neural Posterior Estimation and Model Selection for Chemical Reaction Networks".

The pipeline uses a CNN-Transformer architecture combined with BayesFlow Normalizing Flows to jointly infer mechanism classes and kinetic parameters from partially observed temporal data.

## Structure

The main folder contains:

*   `setup_env.sh`: File called by all other slurm files to set up the environment used for the project (see requirements file in this folder for the dependencies needed).
*   `Data/`: Containing scripts for data generation:
    - `submit_sim.sh`: Slurm file used to submit data generation scripts to the computing cluster.
    - `static_sim.py`: Used to produce training (or testing) data from Mechanisms 1, 2, 3, and 5, considered for the various simulation studies in the paper.
    - `static_sim_M14_15_16.py`: Used to produce training data from Mechanisms 14, 15, and 16. This script was kept separate from the one above as this data was only used for the real data application in this project. 
*   `SMC_ABC/`: Containing two files used for our SMC-ABC implementation discussed in the paper: 
    - `submit_ABC_SMC.sh`: Slurm file used to submit our SMC-ABC to the computing cluster, utilizing multiple cores.
    - `ABC-SMC.py`: Our SMC-ABC implementation. It produces all related baseline figures and results shown in the paper.
*   `neural_posterior_inference/`: Containing all files and codes used for implementations of our neural inference pipeline: 
    - `submit_train.sh`: Slurm file used to submit any of the training scripts included in this folder, using GPU resources from the computing cluster.
    - `submit_inference.sh`: Slurm file used to submit any of the inference and evaluation scripts included within this folder.
    - `Simulation_study/`: Folder containing the code we used to produce all the results for the various settings considered in our simulation study: 
        - `train_M1_M3_JNPE.py`: Main script used to train our pipeline on the main 3 models from the paper (M1, M2, M3). It needs minor adjustments to remove S and train only on P for the settings when S is assumed unobserved.
        - `train_M1_M5_JNPE.py`: Analogous to the file above, but extending the pipeline to consider 4 models (M1, M2, M3, M5). 
        - `infer_M1_M3_JNPE.py`: Main script used to carry out inference with our pipeline on the main 3 models from the paper. It needs minor adjustments to remove S and make inference only from P for the settings when S is assumed unobserved. The variable `T_keep` can be modified to make inference on a random number of time points other than 20. Used to produce all simulation results and figures when considering the three main models.   
        - `infer_M1_M5_JNPE.py`: Analogous to the file above, but extending the pipeline to make inference on 4 models. 
    - `real_data_application/`: Folder containing the training and inference scripts to use our pipeline on real experimental data from a protodeboronation (taken from Table S22 in Ser et al., 2025):
        - `train_M14_M16_JNPE_for_real_data.py`: Training script for the real data application.
        - `inference_M14_M16_JNPE_real_data.py`: Inference script evaluating the empirical data.


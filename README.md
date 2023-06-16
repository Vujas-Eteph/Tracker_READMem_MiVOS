# VOTS 2023 Submission of READMem-MiVOS

## Setting up the environment:
Create a conda environment and install the packages as in the text file "Conda_environment.txt" (Need to filter the list)

Download the propagation model of MiVOS:
cd MiVOS
python download_model.py

## Evaluate on the VOTS 2023 dataset
run : vot evaluate --workspace workspace_VOTS_2023/ Tracker_READMem_MiVOS

## Get analysis
run : vot analysis --workspace workspace_VOTS_2023/ Tracker_READMem_MiVOS

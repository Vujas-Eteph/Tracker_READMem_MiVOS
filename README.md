# VOTS 2023 Submission of READMem-MiVOS

:fire: TODO : 
- "Conda_environment.txt" ( Filter the list to the essential packages only)
- Check that the commands work
- Don't use hard paths, find an alternative
- Add the results of the framework

## Setting up the environment:
Create a conda environment and install the needed packages.

```console
conda create --name Tracker_READMem_MiVOS python=3.10
conda install --name Tracker_READMem_MiVOS --file Conda_environment.txt
conda activate Tracker_READMem_MiVOS
```

## Setting up the tracker
Clone this repository and download the propagation model of MiVOS
```console
git clone https://github.com/Vujas-Eteph/Tracker_READMem_MiVOS
cd Tracker_READMem_MiVOS/MiVOS
python download_model.py
```

## Evaluate on the VOTS 2023 dataset
Supposing you already have the [VOT toolkit](https://votchallenge.net/howto/overview.html), [Trax package](https://github.com/votchallenge/trax) and [the integration](https://github.com/votchallenge/integration) installed.
Set up the workspace with the vots_2023 stack and integrate/test the tracker

```console
cd ../../
vot initialize vots2023 --workspace workspace_VOTS_2023
cp Tracker_READMem_MiVOS/trackers.ini workspace_VOTS_2023/trackers.ini
cd workspace_VOTS_2023
ln -s ../Tracker_READMem_MiVOS/ Tracker_READMem_MiVOS
```
Before going further, adapt the paths:
- in [trackers.ini](https://github.com/Vujas-Eteph/Tracker_READMem_MiVOS/blob/9d7143069f4d1c6038b48b4617246f093ebfc85a/trackers.ini#L14)
- and in [READMem_Tracker.py](https://github.com/Vujas-Eteph/Tracker_READMem_MiVOS/blob/9d7143069f4d1c6038b48b4617246f093ebfc85a/READMem_Tracker.py#LL27C1-L30C123) where  the path of the MiVOS propagation model and the configuration file should be changed.

Now test the tracker on 4 small sequences.
```console
vot test Tracker_READMem_MiVOS
```
You should get the following output message: ```Test concluded successfully```

Now let's evaluate the tracker on the vots_2023 stack
```console
cd ..
vot evaluate --workspace workspace_VOTS_2023 Tracker_READMem_MiVOS
```

After a while (approx. 35 hours), we can run the analysis and pack the results to sent to the server:
```console
vot analysis --workspace workspace_VOTS_2023 Tracker_READMem_MiVOS
vot pack --workspace workspace_VOTS_2023 Tracker_READMem_MiVOS
```

## History
Here is a [link](https://github.com/Vujas-Eteph/Tracker_READMem_MiVOS/blob/main/History.md) to the number of attempts and their specificity - *i.e.*, what change we made to the original READMem_miVOS tracker that made the performance decrease or increase.

## Credits
- [MiVOS](https://github.com/hkchengrex/MiVOS) for the core-architecture 
- [VOT challenge](https://www.votchallenge.net/) for providing the dataset

# Exhaustive and useful information:
## Official to VOT
- The official VOT Challenge website: https://www.votchallenge.net/
- The official VOT Challenge Support section (Always refer to this page when in trouble): https://www.votchallenge.net/howto/integration_multiobject.html
- The official VOT ToolKit (Python 3 version) GitHub page (don't forget to look up issues (sometimes usefull information)): https://github.com/votchallenge/toolkit 
- The official VOT Challenge Support FORUM (Only for discussions): https://groups.google.com/g/votchallenge-help
- Official terminology (under Key concepts): https://github.com/votchallenge/toolkit/blob/master/docs/overview.rst
- More information on when submitting the results to the VOT server: https://github.com/votchallenge/toolkit/issues/102

--------

If you find this work helpful/useful, please consider citing the original paper:
```bibtex
@misc{vujasinović2023readmem,
      title={READMem: Robust Embedding Association for a Diverse Memory in Unconstrained Video Object Segmentation}, 
      author={Stéphane Vujasinović and Sebastian Bullinger and Stefan Becker and Norbert Scherer-Negenborn and Michael Arens and Rainer Stiefelhagen},
      year={2023},
      eprint={2305.12823},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
<br clear="left"/>



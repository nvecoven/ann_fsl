# Feature selection with deep neural networks

This repository provides code allowing to compute feature relevance, as described in **"Nets versus trees for feature rankingand gene network inference"**

## Networks definition
All the code required to build the neural networks is included in the *NetworkConstruction* folder.

## Required software
The code has been built using Tensorflow-2.0 and scikit-learn, which are the only two dependencies for this repository.

## Usage
One can instantiate the class FSNET, defined in *FSNET.py* and train the model to get the feature importances. A complete example is provided through *artificial_launcher.py*. To run it proceed as follows:
- Generate artificial datasets : *python ./generate_datasets.py*
- Train the model and get feature importances : "python ./artificial_launcher.py"

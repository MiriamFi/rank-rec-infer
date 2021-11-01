# Inferring geo-location from recommendations
- Uses Rankrec for generating recommendations
- Uses sklearn for generting classifer models for inferring location

## Setup
- Install the necessary packages
- Make sure to have included the datasets
- For MovieLens experiments, generate user features
- For BookCrossing experiments, apply data preprocessing
- Run experiments

The first step is to intall the necessary packages, which can be found in ```requirements.txt```.

The next step is to include the datasets to be used. In order to replicate the experiments, add a ```data``` folder, which contains the MovieLens 100K dataset in a subfolder ```ml-100k``` and the BookCrossing dataset in the foldder ```book-crossing```.

In order to run the movielens experiments, user features (gender, age, occupation, location) must be generated. This is done by running the script in ```gen_features.py```. Make sure to have the right settings (in the top of the file) for the experiment.

When running the BookCrossing experiment, only a subset of the dataset is used. Therefore, some preprocessing is needed, and can be achieved by running the script in ```bx_prepro.py```. Make sure to have the right settings for the experiment.

Then, everything is ready for the experiments to run. A quick overview of the program files:
- MovieLens experiment files: ```rfm_ex1_ar2.py```, ```rfm_ex1_ar5.py``` and ```rfm_ex1_ar10.py```. The last number reflects the number of location categories for the inference.
- BookCrossinf experiment files:  ```rfm_ex2_05.py``. The last number represents that the 50% most popular items were included in the subset.



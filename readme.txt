################################## Getting started ##################################

# This is a copy of the cefasrepres 'exampleML' repository for anyone struggling with github access rights. The parent 'exampleML' repository may be more up-to-date than this repository, so please only use this repo if the original is not visible.

# To run, please install the environment with conda. Open your conda prompt (i.e. miniforge prompt, anaconda prompt) and navigate to the 'reproducible_environment' folder (specific to your working folder) like so:
cd 'C:\Users\JR13\Desktop\exampleML\reproducible_environment'

# make the environment from the .yml file
conda env create -f python27_spy.yml

# enter the new environment
conda activate opencvenv

# We have used spyder to display the footage in this example. To load spyder, just type spyder. 
spyder

# Once in spyder, open python_code/1_extracting_from_video.py and please PRESS RUN FILE WITH THE GREEN TRIANGLE for each script. This should automatically set your working directory to where you downloaded the repository
# If you run individual lines of code, the working directory will not be set relative to the file it is in, so please ensure package_directory is set to C:\Users\JR13\C:\Users\JR13\Desktop\exampleML or your equivalent.
# execute the three files in exampleML\python_code in numerical order, following the instructions in the header of each file.


################################## Workflow ##################################
# Script 1: Patch extraction
# Execute 1_extracting_from_video.py . This will generate some unclassified image patches.

# Labelling
# Open extracted_training_data in a file window. Drag each swatch into the appropriate directory.
# Note that 'g' has a tail at the top right, but '9' is curved at the top right.
# Also, a horizontal bar is detected as a character. Delete this swatch as a false detection.

# Script 2: Model training
# Execute 2_model_training.py . This will teach a classifier.

# Script 3: Classification
# Execute 3_interpreting_video.py . This will use the classifier to interpret the video.

# The resulting classifications are in the output\output_data_interpretations directory

# I would recommend Adrian Rosebrock's pyimagesearch for further learning about computer vision and machine learning
# Much of the code in this repository has been lifted from various free tutorials at https://www.pyimagesearch.com/ as well as various stackexchange threads. 

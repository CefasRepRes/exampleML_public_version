# This is a copy of the cefasrepres 'exampleML' repository for anyone struggling with github access rights.

# To run, please install the environment with conda. Open your conda prompt (i.e. miniforge prompt, anaconda prompt) and navigate to the 'reproducible_environment' folder (specific to your working folder) like so:
cd 'C:\Users\JR13\Desktop\exampleML\reproducible_environment'

# make the environment from the .yml file
conda env export -f python27_spy.yml

# enter the new environment
conda activate opencvenv

# We have used spyder to display the footage in this example. To load spyder, just type spyder. 
spyder

# Once in spyder, execute the three files in exampleML\python_code in numerical order

# I would reccommend Adrian Rosebrock's pyimagesearch for further learning about computer vision and machine learning
# Much of the code in this repository has been lifted from various free tutorials at https://www.pyimagesearch.com/ as well as various stackexchange threads. 
# Project Name: Ultraviolet

# Author Name: Mason Dana
# Author Contact: mason.dana.2004@gmail.com



This project contains several portions, of which, I will briefly explain here.

For the main file, that can be found as "mainOptimizer.py" under the file "BestResultsLPStatement". 
    This all refers to the main python function using Statement analysis on the LogPrecis datasets.

    Notably, the best results from this can be found in the "all_trails_report.txt" file under "OptimizerOutputFinal" in the "BestREsultsLPStatement" folder. This was the only trial ran during this summer project.

    The "output.log" folder under the "BestResultsLPStatement" folder also contains this information in a neatly organized format.

Furthermore, outside this folder, we have the "hf_cache" which includes information like the actual models.

Then, the "OptimizerOutputFinal" folder contains the working directory's (currently "mainOptimizerLPToken.py", used with token anaylsis) output.

Nexts, the venv is the virtual environment. This can be activated via the function:
    "source venv/bin/activate".

    This contains the working directory's python environment. Ensure you are in the working directory "../Ultraviolet" before running.

For the last folder, the wandb folder contains optional information about the Weights and Balances graphing/recording software for results.


Then, we have several files. I have kept a .env file with a HF_token formatted like: "HF_Token = XXXXXXXXX". I also have a gitignore file to prevent this from being uploaded to GitHub.

I also have the main working file directly in the Ultraviolet folder, currently called "mainOptimizerLPToken.py" for the token analysis.

Up next is the output.log file of the current main files output. This is when running with a specific command (can find in the main file).

I then have the README.md file, standard requirements.txt file, and finally my .JSON files.



This is a brief summary of the contents in this project. To run the program, use the commented out instructions found in the main.py files (like "mainOptimizer.py").

Additionally, a HuggingFace account is required, with access to the Llama 3 - 8B model.
For recording purposes, a WandB account is also required (if desired to record these graphs).



Also, if the venv is not included, please create one and download the requirements.txt file.
If the hf_cache is not included, you will need to download the model.
If the wandb file is not included, you will possibly need to set that up.

These may not be included due to size of the files.

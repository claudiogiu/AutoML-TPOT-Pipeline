# AutoML Pipeline via TPOT

## Introduction  

This repository is designed for implementing an AutoML pipeline via TPOT. The pipeline is trained on a dataset sourced from the research conducted by KOKLU M., SARIGIL S., and OZBEK O. (2021) in their paper *The use of machine learning methods in classification of pumpkin seeds (Cucurbita pepo L.)* (Genetic Resources and Crop Evolution, 68(7), 2713-2726, DOI: [10.1007/s10722-021-01226-0](https://doi.org/10.1007/s10722-021-01226-0)).  

The dataset used for training is publicly available at [this link](https://www.muratkoklu.com/datasets/).  

TPOT adopts evolutionary algorithms to automate the search and optimization of machine learning pipelines. Through a genetic programming approach, it iteratively evaluates, selects, and refines combinations of estimators and feature transformation steps to maximize predictive performance on a given task.

## Getting Started 

To set up the repository properly, follow these steps:  

**1.** **Create the Data Directory**  
   - Before running the pipeline, create a `data/` folder in the project root.  
   - Inside `data/`, create two subdirectories:  
     - `raw/`: This will store the unprocessed dataset.  
     - `processed/`: The data will be split into **training and test sets** and saved here.
  
**2. Set Up the Python Environment**  
 
   - Create and activate a virtual environment:  

     ```sh
     python3 -m venv venv
     source venv/bin/activate  # On Windows use: venv\Scripts\activate 
     ```

   - Install dependencies from `requirements.txt`:  

     ```sh
     pip install -r requirements.txt 
     ``` 

**3. Execute the Pipeline with Makefile**  
   - The repository includes a **Makefile** to automate execution of scripts in the `src/` folder.  
   - Run the following command to execute the full workflow:  

     ```sh
     make run_all  
     ```  
   
   - This command sequentially runs the following modules:
     - `load_data.py`: Ingests the data.
     - `preprocess.py`: Handles data preparation for pipeline initialization.   
     - `train_pipeline.py`: Utilizes TPOT, with the search space of data transformation steps and estimators constrained by the `config.json` file, and saves the highest-performing pipeline setup to the `models/` directory.  
     - `evaluate_pipeline.py`: Computes evaluation metrics to assess pipeline performance on the held-out test set. 


## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository.  

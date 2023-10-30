# NeuralSurvivalClustering :zap:

NeuralSurvivalClustering (NSC) is an innovative tool for benchmarking a variety of survival analysis techniques. The benchmarking utilizes the comprehensive TCGA data set.

## Acknowledgements :clap:
This project has evolved from the original NSC methodology as introduced by Jeanselme. By leveraging the core principles of this technology and integrating the TCGA data set, we've been able to enhance both the performance and the analytical depth of the NSC. For a deeper dive into the original work, please refer to the following resources:

- :mag: [NSC GitHub Repository](https://github.com/Jeanselme/NeuralSurvivalClustering)
- :scroll: [Original NSC Paper](https://proceedings.mlr.press/v174/jeanselme22a/jeanselme22a.pdf)

We wholeheartedly appreciate the substantial efforts of the original authors in creating the NSC technology, the invaluable foundation for our project.

## Technical Specifications :wrench:
- Development Environment: torch 1.7.1+cu110 (cudnn8.0)

## Software Prerequisites :books:
Ensure to use the following versions of the libraries:

- numpy==1.24.1
- pandas==2.0.1
- matplotlib==3.7.1
- scipy==1.10.1

This benchmark has been successfully replicated in a Windows environment. :computer:

## Process of Execution :runner:
- Initiate the data preprocessing by executing example/dataprocessing.ipynb. :arrow_forward:
- Subsequently, in the terminal, run `python main.py`. :arrow_forward:

For running in a Jupyter Notebook environment, please follow these steps:

- Start with the execution of dataprocessing as stated above.
- Afterwards, execute example/clustering_TCGA.ipynb. :arrow_forward: 
 

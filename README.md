# NeuralSurvivalClustering 💥

NeuralSurvivalClustering (NSC) offers benchmarking for various techniques used in survival analysis. The benchmarking dataset employed is the TCGA data.

## References 🔗
This work builds upon the NSC methodology, originally presented by Jeanselme. We have adopted the foundational principles of this technology and integrated the TCGA dataset for enhanced performance and analysis. For further details about the original work, please refer to the following resources:

- 🔎 [NSC GitHub Repository](https://github.com/Jeanselme/NeuralSurvivalClustering)
- 📄 [Original NSC Paper](https://proceedings.mlr.press/v174/jeanselme22a/jeanselme22a.pdf)

We acknowledge and appreciate the efforts of the original authors in developing the NSC technology, which has served as a stepping stone for our project.

## Development Environment 🛠️
- torch 1.7.1+cu110

## Dependencies 📚
Ensure to match the following library versions:

- numpy==1.24.1
- pandas==2.0.1
- matplotlib==3.7.1
- scipy==1.10.1

This benchmark has been reproduced in a Windows environment. 💻

## Execution Steps 🏃
- For data preprocessing, execute example/dataprocessing.ipynb. ▶️
- Then, in the terminal, run `python main.py`. ▶️

If you wish to run in a Jupyter Notebook environment, follow the steps below:

- First, execute the dataprocessing as mentioned above.
- Then, run clustering_TCGA.ipynb. ▶️

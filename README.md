# Mode Adaptive Graph Network (MAGN) 
This repository is based on traffic forecasting on the LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/ee57cd73a76bd927ffca3dda1dc3b9d4-Paper-Datasets_and_Benchmarks.pdf) that consists of three regions Greater Bay Area (GBA), Greater Los Angeles (GLA), and San Diego (SD). Our main code has been modified from the Github repo: [Link](https://github.com/liuxu77/LargeST). 

You can download all data from the provided [link](https://www.kaggle.com/datasets/liuxu77/largest).

The Variational Mode Decomposition method is proposed by Konstantin Dragomiretskiy and Dominique Zosso  [Link](https://ieeexplore.ieee.org/document/6655981) to determine the modes of the signals.

In our work, we make learnable parameters for VMD and unfold this iterative algorithm. 

## Preprocessing 
To preprocessing the raw data, please refer to github repo [Link](https://github.com/liuxu77/LargeST). We set the sampling time for 15 minutes during preprocessing.
### 0.1 Process Traffic Flow Data of CA
In a jupyter notebook `process_ca_his.ipynb` in the folder `data/ca` to process and generate a cleaned version of the flow data. Please go through this notebook.

### 0.2 Generate Traffic Flow Data for Training
Please go to the `data` folder, and use the command below to generate the flow data for model training in our manuscript.
```
python generate_data_for_training.py --dataset ca --years 2019
```
The processed data are stored in `data/ca/2019`. We also support the utilization of data from multiple years. For example, changing the years argument to 2018_2019 to generate two years of data.

### 0.3 Generate Other Sub-Datasets
We describe the generation of the GLA dataset as an example. Please first go through all the cells in the provided jupyter notebook `generate_gla_dataset.ipynb` in the folder `data/gla`. Then, use the command below to generate traffic flow data for model training.
```
python generate_data_for_training.py --dataset gla --years 2019
```
It will generate the files in `/data/gla/2019/his.npz` and other supporting files. Copy `his.npz` file and place in main directory. 
```
main
|-- uvmd.py
|-- src |-- models |-- astgcn.py
```
## Main Code
The main architecture consists of two main compoenents: decomposition of spatiotemporal data and deep neural network:

### 1. Unfolding of Variational Mode Decomposition (UVMD)
The data for decomposition is arranged in the order of (time, nodes, features). The features in our work are concatenated such as counts, time of the day, day of the week. The output features of this decomposition will be (counts,modes,time of the day, day of the week). 

It is a torch based code that supports GPU and unfolds the VMD and using the mode specific bandwidth constraint. 
```
python uvmd.py --K 13 --layers 1
```
After training copy the `his.npz` file from unfold folder and replace it with original file in the preprocessed folder.  
### 2. Neural Network 
 We use the flow data from 2019 in our training and evaluation in our paper.  The backbone of the deep neural network is based on [ASTGCN](https://github.com/guoshnBJTU/ASTGCN-2019-pytorch).

#### Training 


Execute the Python file in the terminal

To run the ASTGCN,
```
python experiments/astgcn/main.py --device cuda:0 --model_name astgcn --dataset GLA --years 2019 --bs 4 --input_dim 15 
```


#### Evaluation
We evaluate MAE, RMSE, and MAPE error metrics on prediction horizons 1 to 12. 

```
python experiments/astgcn/main.py --device cuda:0 --model_name astgcn --dataset GLA --years 2019 --bs 4 --input_dim 15 --mode test
```
#### Citation
If you find our work helpful in your research work, please cite:
```
@misc{ahmad2025robustspatiotemporalforecastingusing,
      title={Robust Spatiotemporal Forecasting Using Adaptive Deep-Unfolded Variational Mode Decomposition}, 
      author={Osama Ahmad and Lukas Wesemann and Fabian Waschkowski and Zubair Khalid},
      year={2025},
      eprint={2509.00703},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.00703}, 
}
```

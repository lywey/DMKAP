# DMKAP

This project was implemented using the Tensorflow framework.

Tensorflow 2.2.0, Python 3.7, cuda 10.2, cudnn 7.6.5



#### Running DMKAP

1. You will first need to request access for [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/), [Single-level Clinical Classification Software(CCS) for ICD-9-CM](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp), and [Multi-level CCS](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp).

2. Use "process_mimic.py" to process MIMIC-III dataset and generate a suitable training dataset.

3. Use "build_trees.py" to build files that contain the ancestor information of each medical code.

4. Use "process_treeseq.py" to generate knowledge sequences for model input. Use "process_label.py" to generate the labels.

5. Use "process_gcnlabel.py" to generate the labels required by the GCN embedding module, and use "process_adjacencylist.py" to generate the adjacency matrix required by the GCN embedding module.

6. Use "train.py" to generate GCN Embedding in the "gcn_embedding" folder. Before to execute this algorithm, it is necessary to install these required packages shown in the file named ' requirements.txt '. You can also install all the required packages by just using one command :

   ```
   $ pip install -r requirements.txt
   ```

7. Run DMKAP by using "run.py".
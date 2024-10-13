# Code for GloCOM: A Short Text Neural Topic Model via Global Clustering Context

## Preparing Libraries

1. Python 3.10.14
2. Install the following libraries
    ```
    numpy==1.26.3
    scipy==1.10.1
    sentence-transformers==2.7.0
    torch==2.4.1+cu124
    torchvision==0.19.1+cu124
    gensim==4.3.3
    scikit-learn==1.5.1
    tqdm==4.66.5
    ```
    
2. Install Java
3. Download [this Java jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to `./evaluations` and rename it to `palmetto.jar`
4. Download and extract [this processed Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to `./evaluations/wiki_data` as an external reference corpus.

    Here is the folder structure:
    ```
        |- evaluations
            | - wiki_data
                | - wikipedia_bd/
                | - wikipedia_bd.histogram
            |- ...
            |- palmetto.jar
    ```

## Running
To run and evaluate our model, run the following command:
```
python run.py --model GloCOM --num_topics 50 --data_dir data/SearchSnippets 
```

You can also specify additional arguments when running the model:

```
--aug_coef <float> # Default: 0.5 - Coefficient for augmentation 
--prior_var <float> # Default: 0.1 - Prior variance
--weight_loss_ECR <float> # Default: 60.0 - Weight for ECR loss
```

## KNNTM Running

We provide KNNTM OT distances and code in [this link](https://drive.google.com/file/d/1ArmApnG2exZNp8zZ-F-bUVEGj7Qi2F92/view?usp=sharing). Unzip the file and the folder structure should be like this:

    ```
    |- data
        | - SearchSnippets
            | - KNNTM/
                | - M_coo.npz
                | - M_cos.npz
    ```

To run the KNNTM model

```
python run.py --model KNNTM --num_topics 50 --data_dir data/SearchSnippets
```

You can also specify additional arguments when running the model:
```
--alpha <float> # Default: 1.0
--num_k <int> # Default: 30 
--eta <float> # Default: 0.2 
--rho <float> # Default: 0.6 
--p_epochs <int> # Default: 20
```

## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for the evaluation of topic coherence.

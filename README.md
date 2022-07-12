# MSdocTr-Lite
 
Table of contents:
 
1. [Datasets](#Datasets)
 

 

## Datasets
This section is dedicated to the datasets used in the paper: download and formatting instructions are provided 
for experiment replication purposes.

### IAM

#### Details

IAM corresponds to english grayscale handwriting images (from the LOB corpus).
We provide a script to format this dataset for the commonly used split for result comparison purposes.
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 6,482 |     976    | 2,915 |
| paragraph |  747  |     116    |  336  |

#### Download



- Register at the [FKI's webpage](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- Download the dataset [here](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database) 
- Move the following files into the folder Datasets/raw/IAM/
    - formsA-D.tgz
    - formsE-H.tgz
    - formsI-Z.tgz
    - lines.tgz
    - ascii.tgz



 
 

### Format the datasets

- Comment/Uncomment the following lines from the main function of the script "format_datasets.py" according to your needs and run it

```
if __name__ == "__main__":

    # format_IAM_line()
    # format_IAM_paragraph()

```

- This will generate well-formated datasets, usable by the training scripts.


2. [generate dataset](#Datasets)
```
python3 /home/marwa/MSdocTr-Lite/raw/generate_iam_from_lines.py
```
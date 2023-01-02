# corpusCasM2
A corpus of annotated clinical texts. 

## Annotation

Annotation guides are available in the `annotation_guide` folder.
The annotation was performed collaborativelly by the students of  masters students from Université Paris Cité (see [annotators list](annotation_guide/annotators.md)). 

### Annotated Concepts 

Temporality
: mentions of temporal information (date, duration, frequency) and their scope (the portion of the text impacted by the temporality mention)

Medical entities
: problem, treatment, test

see annotation guides for more informations


## Dataset

- train_set: 423 documents, 8305 sentences
- test_set: 133 documents, 2545 sentences
- validation_set: 106 documents, 2122 sentences

## Usage

Using [Hugging Face datasets](https://huggingface.co/docs/datasets/index)

In your terminal: 
```bash
git clone https://github.com/aneuraz/corpusCasM2.git
```
Alternatively, you can download the content of the github repository and unzip it to a local directory. 

```bash
cd corpusCasM2 # or <PATH_TO_LOCAL_DIR>
```

Install the required libraries (ideally using a virtual environment such as conda): 
```bash
conda create --name corpuscas python=3
conda activate corpuscas
pip install -r requirements.txt
```

Then in python: 

```python
import datasets
corpusCasM2 = datasets.load_dataset('corpusCasM2')
```
NB: the dataset is not available on the Hugging Face Hub. Therefore, you need to have it in a local directory for this command to run. 
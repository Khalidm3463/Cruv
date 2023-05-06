# Cruv

## Installation and Setup

#### Clone this repository:
``` bash
git clone https://github.com/Khalidm3463/Cruv.git
```

#### Required Dependencies and Libraries

+ **pandas**
+ **numpy**
+ **spacy**
+ **scikit-learn**

#### Installation of Required Dependencies

``` python
pip install pandas numpy spacy scikit-learn
```

#### Download the spaCy English model

``` python
python -m spacy download en_core_web_sm
```

<hr>

## Dataset

- The dataset is named as **'news.csv'** and thus is stored in the Comma Seperated Values **(.csv)** format.
- The dataset schema

    | Field        | Data Type | Description                             |
    |--------------|-----------|-----------------------------------------|
    | title        | String    | Title of the article                    |
    | content      | String    | The text content of the article         |
    | published_at | String    | Date and time the article was published |
    | source       | String    | The source of the article               |
    | topic        | String    | The topic of the article                |

<hr>

## Usage
1. Head on to the directory where you downloaded these files in.

2. Open the terminal in that corresponding directory.

    - If you are using python3, use the following command:
    ``` python
    python3 news.py
    ```
    - If you are using lower versions of python, use the following command:
    ```
    python news.py
    ```
3. The program then summarizes the content and stores the summary in the **'summary.csv'** file in the same directory.
- Note
   - The program uses the **'news.csv'** dataset therefore we don't need to mention its name as an argument.
   - The program may take several minutes to run.

<hr>

## Extras
- I have also added a Python Notebook named 'news.ipynb' for this project.



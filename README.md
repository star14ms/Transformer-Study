# Study Transformer

Study of Transformer Architecture

## Requirements

- Conda (optional)
- Python 3.7 or higher
- Pytorch Lightning

## Installation

Using conda
```cmd
conda env create -f environment.yml -n new_environment_name
```

Using pip
```cmd
pip install -r requirements.txt
```

## Usage

```cmd
python data/*/tokenizer.py
python train.py
python test.py
```

## Sample Datasets

1. [addition](data/addition/addition.txt)
2. [date](data/date/date.txt)

### 1. data/addition/addition.txt
```txt
16+75  _91  
52+607 _659 
75+22  _97  
63+22  _85  
795+3  _798 
706+796_1502
8+4    _12  
84+317 _401 
9+3    _12  
6+2    _8   
...
```

### 2. data/date/date.txt

TODO: make tokenizer and model for this dataset
```txt
september 27, 1994           _1994-09-27
August 19, 2003              _2003-08-19
2/10/93                      _1993-02-10
10/31/90                     _1990-10-31
TUESDAY, SEPTEMBER 25, 1984  _1984-09-25
JUN 17, 2013                 _2013-06-17
april 3, 1996                _1996-04-03
October 24, 1974             _1974-10-24
AUGUST 11, 1986              _1986-08-11
February 16, 2015            _2015-02-16
...
```

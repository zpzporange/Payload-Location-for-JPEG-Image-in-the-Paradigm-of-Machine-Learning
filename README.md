#  Payload Location for JPEG Image in the Paradigm of Machine Learning

This repository contains the source code for the paper Payload Location for JPEG Image in the Paradigm of Machine Learning by  Tong Qiao<sup>†</sup>, Pan Zeng<sup>†</sup>, Ben Niu, Yanli Chen, and Xiangyang Luo<sup>*</sup>

### Installation

We support `python3`. To install the dependencies run:

```python
pip install -r requirements.txt
```

###  Input & Output

**Input**  : A set of stego JPEG images with the same dimensions and embedded with the same stego-key,
and the corresponding stego-key
**Output ** : A location map that indicates the embedding positions in the input images,
represented by a binary matrix with the same size as the input images

### Dataset

**Bossbase**. This dataset can be download [here](http://agents.fel.cvut.cz/stegodata/).

#### Additional notes

Citation:
# RO Diacritics module

**RO Diacritics** is a straightforward diacritics restoration module for Romanian Language

```python
from ro_diacritics import restore_diacritics
print(restore_diacritics("fara poezie, viata e pustiu"))
```

or correcting a pandas dataframe:

```python
from ro_diacritics import restore_diacritics
df['text-diacritice'] = df['text'].apply(restore_diacritics)
```

## Installing

```console
$ python -m pip install ro-diacritics
```
or

```console
$ pip install ro-diacritics
```

## Requirements

 * torch and torchtext
 * numpy
 * nltk and scikit-learn (for training)
 * needs nltk.download('punkt') for tokenization

## References

- Ruseti, S., Cotet, T. M., & Dascalu, M. (2020). Romanian Diacritics Restoration Using Recurrent Neural Networks. arXiv preprint arXiv:2009.02743.
- https://github.com/teodor-cotet/DiacriticsRestoration

# Readme
## Environment
### pip
`pip install -r requirements.txt`


## AssertFood
### NLTK Corpus

    import nltk
    nltk.download()
    
    >>> wordnet [ENTER]

### Google Inception

    from assert_food.views import maybe_download_and_extract, make_word_set
    maybe_download_and_extract()
    make_word_set()
    
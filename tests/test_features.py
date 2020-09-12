from reddit.processing.features import TextTokenizer

def test_tokenizer_lemmatizes(pretokenized_dataset):
    tokenizer = TextTokenizer(variable='text')
    tokenized = tokenizer.transform(pretokenized_dataset)
    assert tokenized[0] == 'hurry'
    assert tokenized[1] == 'finish'
    assert tokenized[2] == ''
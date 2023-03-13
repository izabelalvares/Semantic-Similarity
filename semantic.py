#compulsory task 1

import spacy

# run en_core_web_md
nlp = spacy.load('en_core_web_md')


sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# cat, monkey, banana and my own example

tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# cat and monkey have higher similarity than fruit and animal suggesting it's because they're both animals
# apple and apple have highest similarity (1) - because they are the same thing
# banana monkey has higher similarity than banana cat showing that spacy knows that monkeys eat bananas

# run en_core_web_sm

nlp = spacy.load('en_core_web_sm')


sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

#this time when using the en_core_web_sm model for semantic similarity, I received the following warning message:
# The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger,
# parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small models,
# e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, or use one of the larger models instead if available.

#en_core_web_sm results in lower similarity values compared to en_core_web_md even though it's analysing the same data
#interestingly, web_md ranks 'where did my dog go' lower in similarity compared to 'I will name my dog diana'..
# however the converse is true for web_sm


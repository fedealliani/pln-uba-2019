from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

classifiers = {
    'lr': LogisticRegression,
    'svm': LinearSVC,
    'mnnb':MultinomialNB
}


def feature_dict(sent, i):
    """Feature dictionary for a given sentence and position.

    sent -- the sentence.
    i -- the position.
    """
    word = sent[i]
    previousWord = ""
    nextWord = ""
    if i==0:
        if len(sent)>1:
            previousWord = "<s>"
            nextWord = sent[i+1]
        else:
            previousWord = "<s>"
            nextWord = "</s>"
    elif i==len(sent)-1:
        if i-1>=0:
            previousWord = sent[i-1]
            nextWord = "</s>"
        else:
            previousWord = "<s>"
            nextWord = "</s>"
    else:
        previousWord = sent[i-1]
        nextWord = sent[i+1]  
    return features(word,previousWord,nextWord)

def features(word,previousWord,nextWord):
        return {
            'lower': word.lower(),
            'istitle': word.istitle(),
            'isupper':word.isupper(),
            'isnumber':word.isnumeric(),
            'p_lower': previousWord.lower(),
            'p_istitle': previousWord.istitle(),
            'p_isupper':previousWord.isupper(),
            'p_isnumber':previousWord.isnumeric(),
            'n_lower': nextWord.lower(),
            'n_istitle': nextWord.istitle(),
            'n_isupper':nextWord.isupper(),
            'n_isnumber':nextWord.isnumeric(),
        }

class ClassifierTagger:
    """Simple and fast classifier based tagger.
    """

    def __init__(self, tagged_sents, clf='lr'):
        """
        clf -- classifying model, one of 'svm', 'lr' (default: 'lr').
        """
        self.pipeline = Pipeline([
            ('vect', DictVectorizer()),
            ('clf', classifiers[clf]())
        ])
        self.tagged_sents = list(tagged_sents)
        self.known_words = []
        self.fit(self.tagged_sents)

    def fit(self, tagged_sents):
        """
        Train.

        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        
        tagged_sents = list(tagged_sents)
        for sent in tagged_sents:
            for word,_ in sent:
                if word not in self.known_words:
                    self.known_words.append(word)

        X = []
        y_true = []

        for sent in self.tagged_sents:
            for i,_ in enumerate(sent):
                y_true.append(sent[i][1])
                X.append(feature_dict([i[0] for i in sent],i))
                
        self.pipeline.fit(X,y_true)

    def tag_sents(self, sents):
        """Tag sentences.

        sent -- the sentences.
        """
        return [self.tag(sent) for sent in sents]


    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        X_test = [feature_dict(sent,i) for i,_ in enumerate(sent)]
        y_pred = self.pipeline.predict(X_test)
        return y_pred

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        if w in self.known_words:
            return False

        return True
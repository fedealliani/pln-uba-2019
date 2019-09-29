from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

classifiers = {
    'lr': LogisticRegression,
    'svm': LinearSVC,
}


def feature_dict(sent, i):
    """Feature dictionary for a given sentence and position.

    sent -- the sentence.
    i -- the position.
    """
    word = sent[i][0]
    previousWord = ""
    nextWord = ""
    if i==0:
        if len(sent)>1:
            previousWord = "<s>"
            nextWord = sent[i+1][0]
        else:
            previousWord = "<s>"
            nextWord = "<s>"
    elif i==len(sent)-1:
        if i-1>=0:
            previousWord = sent[i-1][0]
            nextWord = "<s>"
        else:
            previousWord = "<s>"
            nextWord = "<s>"
    else:
        previousWord = sent[i-1][0]
        nextWord = sent[i+1][0]            
    return {
            'w': word.lower(),
            'wt': word.istitle(),
            'wu':word.isupper(),
            'wd':word.isnumeric(),
            'pw': previousWord.lower(),
            'nw': nextWord.lower(),
            'nwt': nextWord.istitle(),
            'nwu':nextWord.isupper(),
            'nwd':nextWord.isnumeric(),
        }


class ClassifierTagger:
    """Simple and fast classifier based tagger.
    """

    def __init__(self, tagged_sents, clf='lr'):
        """
        clf -- classifying model, one of 'svm', 'lr' (default: 'lr').
        """
        self.tagged_sents = list(tagged_sents)
        self.X = []
        self.y_true = []

        for sent in self.tagged_sents:
            for i,_ in enumerate(sent):
                self.y_true.append(sent[i][1])
                self.X.append(feature_dict(sent,i))

        self.features = list(zip(self.X, self.y_true))
        self.fit(self.tagged_sents)
        # WORK HERE!!

    def fit(self, tagged_sents):
        """
        Train.

        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        self.vect = DictVectorizer()
        self.vect.fit(self.X)
        self.clf = MultinomialNB()
        X2 = self.vect.transform(self.X)
        self.clf.fit(X2, self.y_true)

    def tag_sents(self, sents):
        """Tag sentences.

        sent -- the sentences.
        """
        return [self.tag(sent) for sent in sents]


    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        import pdb; pdb.set_trace()
        X_test = [feature_dict(sent,i) for i,_ in enumerate(sent)]
        X2_test = self.vect.transform(X_test)
        y_pred = self.clf.predict(X2_test)
        import pdb; pdb.set_trace()
        return y_pred
        # WORK HERE!!

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        # WORK HERE!!

    def features(self,word,previousWord,nextWord):
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
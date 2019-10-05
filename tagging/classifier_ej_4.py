
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
            'ends_with_ista': word[-4:]=='ista',
            'ends_with_esta': word[-4:]=='esta',
            'ends_with_asta': word[-4:]=='asta',
           #'ends_with_ar': word[-2:]=='ar',
            #'ends_with_er': word[-2:]=='er',
            #'ends_with_ir': word[-2:]=='ir',
             #'ends_with_mente':word[-5:]=='mente',
            'ends_with_ción':word[-4:]=='ción',
            'ends_with_sión':word[-4:]=='sión',
            'ends_with_ada':word[-3:]=='ada',
            'ends_with_ido':word[-3:]=='ido',
            'ends_with_ado':word[-3:]=='ado',
            'istitle': word.istitle(),
            'isupper':word.isupper(),
            'isnumber':word.isnumeric(),
            'p_lower': previousWord.lower(),
            'p_istitle': previousWord.istitle(),
            'p_isupper':previousWord.isupper(),
            'p_isnumber':previousWord.isnumeric(),
            'p_ends_with_ista': previousWord[-4:]=='ista',
            'p_ends_with_esta': previousWord[-4:]=='esta',
            'p_ends_with_asta': previousWord[-4:]=='asta',
            #'p_ends_with_ar': previousWord[-2:]=='ar',
            #'p_ends_with_er': previousWord[-2:]=='er',
            #'p_ends_with_ir': previousWord[-2:]=='ir',
            #'p_ends_with_mente':previousWord[-5:]=='mente',
            'p_ends_with_ción':previousWord[-4:]=='ción',
            'p_ends_with_sión':previousWord[-4:]=='sión',
            'p_ends_with_ada':previousWord[-3:]=='ada',
            'p_ends_with_ido':previousWord[-3:]=='ido',
            'p_ends_with_ado':previousWord[-3:]=='ado',
            
            'n_lower': nextWord.lower(),
            'n_istitle': nextWord.istitle(),
            'n_isupper':nextWord.isupper(),
            'n_isnumber':nextWord.isnumeric(),
            #'n_ends_with_ar': nextWord[-2:]=='ar',
            #'n_ends_with_er': nextWord[-2:]=='er',
            #'n_ends_with_ir': nextWord[-2:]=='ir',
            #'n_ends_with_mente':nextWord[-5:]=='mente',
            'n_ends_with_ista': nextWord[-4:]=='ista',
            'n_ends_with_esta': nextWord[-4:]=='esta',
            'n_ends_with_asta': nextWord[-4:]=='asta',
            'n_ends_with_ción':nextWord[-4:]=='ción',
            'n_ends_with_sión':nextWord[-4:]=='sión',
            'n_ends_with_ción':nextWord[-4:]=='ción',
            'n_ends_with_sión':nextWord[-4:]=='sión',
            'n_ends_with_ada':nextWord[-3:]=='ada',
            'n_ends_with_ido':nextWord[-3:]=='ido',
            'n_ends_with_ado':nextWord[-3:]=='ado',
        }

class ClassifierTaggerEj4:
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
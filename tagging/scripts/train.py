"""Train a sequence tagger.

Usage:
  train.py [options] -c <path> -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: badbase]:
                  badbase: Bad baseline
                  base: Baseline
                  classifier: ClassifierTagger
                  classifier_ej_4: ClassifierTaggerEj4
  -t <classifier_type> Classifier type [default: lr]:
                  lr: Logistic Regression
                  mnnb: MultinomialNB
                  svm: LinearSVC
  -c <path>     Ancora corpus path.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tagging.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger, BadBaselineTagger
from tagging.classifier import ClassifierTagger
from tagging.classifier_ej_4 import ClassifierTaggerEj4

models = {
    'badbase': BadBaselineTagger,
    'base': BaselineTagger,
    'classifier':ClassifierTagger,
    'classifier_ej_4':ClassifierTaggerEj4,
}


if __name__ == '__main__':
    opts = docopt(__doc__)
    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader(opts['-c'], files)
    sents = corpus.tagged_sents()

    # train the model
    model_class = models[opts['-m']]
    clf = opts['-t']
    if clf==None:
      clf = 'lr'
    model = model_class(sents,clf)
    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
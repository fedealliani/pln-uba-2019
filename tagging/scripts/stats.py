"""Print corpus statistics.

Usage:
  stats.py -c <path>
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt
from collections import defaultdict

from tagging.ancora import SimpleAncoraCorpusReader


class POSStats:
    """Several statistics for a POS tagged corpus.
    """
    def __init__(self, tagged_sents):
        """
        tagged_sents -- corpus (list/iterable/generator of tagged sentences)
        """
        # Convert nltk.collections.LazyMap to list
        self.corpus = list(tagged_sents)
        # Set number of tokens to 0
        self.numberOfTokens = 0
        # Create Dictionary Token -> Number of appearances
        self.tokenToNumberOfAppearances = defaultdict(lambda:0)
       
        # Create Dictionary Token -> [Tag1,Tag2,...]
        self.tokenToTags = defaultdict(lambda:set([]))
        
        # Create Dictionary Tag -> Number of appearances
        self.tagsToNumberOfAppearances = defaultdict(lambda:0)
        
        # Create Dictionary Tag -> {Token1:Number of Appearances,Token2:Number of Appearances,..}
        self.tagToTokens_ToNumberOfAppearances = defaultdict(lambda:defaultdict(lambda: 0))
        

        # Go through the sentences to count the words
        for phrase in self.corpus:
            for token, tag in phrase:
                self.tokenToTags[token].add(tag)
                self.tagToTokens_ToNumberOfAppearances[tag][token] +=1
                self.tokenToNumberOfAppearances[token] +=1
                self.numberOfTokens +=1
                self.tagsToNumberOfAppearances[tag] +=1

    def sent_count(self):
        """Total number of sentences."""
        return len(self.corpus)

    def token_count(self):
        """Total number of tokens."""
        return self.numberOfTokens
                
    def words(self):
        """Vocabulary (set of word types)."""
        return list(self.tokenToNumberOfAppearances.keys())

    def word_count(self):
        """Vocabulary size."""
        return len(self.words())

    def word_freq(self, w):
        """Frequency of word w."""                    
        return self.tokenToNumberOfAppearances[w]

    def unambiguous_words(self):
        """List of words with only one observed POS tag."""
        return [token for token, tags in self.tokenToTags.items() if len(tags)==1]

    def ambiguous_words(self, n):
        """List of words with n different observed POS tags.

        n -- number of tags.
        """      
        return [token for token, tags in self.tokenToTags.items() if len(tags)==n]

    def tags(self):
        """POS Tagset."""
        return list(self.tagsToNumberOfAppearances.keys())

    def tag_count(self):
        """POS tagset size."""
        return len(self.tags())

    def tag_freq(self, t):
        """Frequency of tag t."""
        return self.tagsToNumberOfAppearances[t]

    def tag_word_dict(self, t):
        """Dictionary of words and their counts for tag t."""
        return dict(self.tagToTokens_ToNumberOfAppearances[t])


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data with <path> because with -c not working
    corpus = SimpleAncoraCorpusReader(opts['<path>'])
    sents = corpus.tagged_sents()

    # compute the statistics
    stats = POSStats(sents)

    print('Basic Statistics')
    print('================')
    print('sents: {}'.format(stats.sent_count()))
    token_count = stats.token_count()
    print('tokens: {}'.format(token_count))
    word_count = stats.word_count()
    print('words: {}'.format(word_count))
    print('tags: {}'.format(stats.tag_count()))
    print('unambiguous words:{}'.format(len(stats.unambiguous_words())))
    print('')

    print('Most Frequent POS Tags')
    print('======================')
    tags = [(t, stats.tag_freq(t)) for t in stats.tags()]
    sorted_tags = sorted(tags, key=lambda t_f: -t_f[1])
    print('tag\tfreq\t%\ttop')
    for t, f in sorted_tags[:10]:
        words = stats.tag_word_dict(t).items()
        sorted_words = sorted(words, key=lambda w_f: -w_f[1])
        top = [w for w, _ in sorted_words[:5]]
        print('{0}\t{1}\t{2:2.2f}\t({3})'.format(t, f, f * 100 / token_count, ', '.join(top)))
    print('')

    print('Word Ambiguity Levels')
    print('=====================')
    print('n\twords\t%\ttop')
    for n in range(1, 10):
        words = list(stats.ambiguous_words(n))
        m = len(words)
        
        # most frequent words:
        sorted_words = sorted(words, key=lambda w: -stats.word_freq(w))
        top = sorted_words[:5]
        print('{0}\t{1}\t{2:2.2f}\t({3})'.format(n, m, m * 100 / word_count, ', '.join(top)))

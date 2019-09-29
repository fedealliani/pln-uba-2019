from collections import defaultdict
from tagging.scripts.stats import POSStats

class BadBaselineTagger:

    def __init__(self, tagged_sents, default_tag='nc0s000'):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        default_tag -- tag for all words.
        """
        self._default_tag = default_tag

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        return self._default_tag

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return True


class BaselineTagger:

    def __init__(self, tagged_sents, default_tag='nc0s000'):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        default_tag -- tag for unknown words.
        """
        self._default_tag = default_tag
        self.tagged_sents = list(tagged_sents)

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        if not hasattr(self, 'stats'):
            self.stats = POSStats(self.tagged_sents)

        if self.unknown(w):
            return self._default_tag
        
        tagsForW = list(self.stats.tokenToTags[w])
        tagsForW = sorted(tagsForW)
        tagMoreFrequent = tagsForW[0]
        numberOfAppearancesForTagMoreFrequent = self.stats.tagsToNumberOfAppearances[tagMoreFrequent]
        for tag in tagsForW:
            if self.stats.tagsToNumberOfAppearances[tagMoreFrequent] > numberOfAppearancesForTagMoreFrequent:
                tagMoreFrequent = tag
                numberOfAppearancesForTagMoreFrequent = self.stats.tagsToNumberOfAppearances[tagMoreFrequent]        

        return tagMoreFrequent

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        if not hasattr(self, 'stats'):
            self.stats = POSStats(self.tagged_sents)

        if len(self.stats.tokenToTags[w]) == 0:
            return True

        return False

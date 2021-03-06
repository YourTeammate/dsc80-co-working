
import os
import pandas as pd
import numpy as np
import requests
import time
import re

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """

    r = requests.get(url)
    raw_text = r.text
    start_index = re.search(r'(\*\*\* START OF.+\*\*\*)', raw_text).span()[1]
    end_index = re.search(r'(\*\*\* END OF.+\*\*\*)', raw_text).span()[0]
    
    return raw_text[start_index:end_index].replace('\r\n', '\n')
    
# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """

    text_list = re.split('(\W)', re.sub(r'\n', ' ', re.sub(r'\n{2,}', ' \x03 \x02 ', "\n\n\n" + book_string + "\n\n\n")))

    return [i for i in text_list if (i != ' ' and i != '')][1:-1]
    
# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        words = pd.Series(tokens).value_counts()
        num_unique = len(words)

        def assign_prob(row):
            return 1/num_unique

        ser = pd.Series(words.index).apply(assign_prob)
        ser.index = words.index
        return ser
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        if pd.Series(words).isin(self.mdl.index).all():
            result = 1
            for i in words:
                result *= self.mdl[i]
            return result
        else:
            return 0
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        return ' '.join(np.random.choice(list(self.mdl.index), p=self.mdl.values, size=M))

            
# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        ser = pd.Series(tokens).value_counts()
        tot_cnts = ser.sum()

        def assign_prob(row):
            return ser[row] / tot_cnts

        res = pd.Series(ser.index).apply(assign_prob)
        res.index = ser.index
        return res
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        if pd.Series(words).isin(self.mdl.index).all():
            result = 1
            for i in words:
                result *= self.mdl[i]
            return result
        else:
            return 0
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        return ' '.join(np.random.choice(list(self.mdl.index), p=self.mdl.values, size=M))
        
    
# ---------------------------------------------------------------------
# Question #5,6,7,8
# ---------------------------------------------------------------------

class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N-1, tokens)
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        return [tuple(tokens[i:i + self.N]) for i in range(len(tokens) - self.N + 1)]
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """

        # ngram counts C(w_1, ..., w_n)
        ngrams_ct = pd.Series(ngrams).value_counts()
        # n-1 gram counts C(w_1, ..., w_(n-1))
        n1grams = [x[:-1] for x in ngrams]
        n1grams_ct = pd.Series(n1grams).value_counts()

        # Create the conditional probabilities
        def assign_prob(row):
            return ngrams_ct[row] / n1grams_ct[row[:-1]]

        prob = pd.Series(ngrams_ct.index).apply(assign_prob)
        # prob.index = ngrams_ct.index
        
        # Put it all together
        ans = pd.DataFrame()
        ans['ngram'] = pd.Series(ngrams_ct.index)
        n1grams_list = [x[:-1] for x in list(ngrams_ct.index)]
        ans['n1gram'] = pd.Series(n1grams_list)
        ans['prob'] = prob

        return ans
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4)*(1/2)*(1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        ngrams_wd = self.create_ngrams(words)
        if len(ngrams_wd) == 0:
            if len(words) == 0:
                return 0
            else:
                return self.prev_mdl.probability(words)

        if pd.Series(ngrams_wd).isin(self.mdl['ngram']).all():
            result = 1
            temp_mdl = self.mdl.set_index('ngram')
            for i in ngrams_wd:
                result *= temp_mdl['prob'].get(i)
            prev_prob = self.prev_mdl.probability(ngrams_wd[0][:-1])
            return result * prev_prob
        else:
            return 0

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """

        # Use a helper function to generate sample tokens of length `length`
        def token_sample(M):
            if M < self.N:
                return self.prev_mdl.sample(M).split()

            tokens = token_sample(M-1)
            if len(tokens) > M-1:
                return tokens

            last_n1gram = tuple(tokens[-(self.N - 1):])
            pool = self.mdl[self.mdl['n1gram'] == last_n1gram]
            # print(tokens[-(self.N - 1):])
            if pool.shape[0] == 0:
                return tokens + ['\x03', '\x02'] + self.prev_mdl.sample(self.N - 1).split()

            cur_token = np.random.choice(pool['ngram'].values, p=pool['prob'].values, size=1)

            return tokens + [cur_token[-1][-1]]

        # Transform the tokens to strings
        tks = ['\x02'] + token_sample(M)[:M]
        ans = ' '.join(tks)

        return ans


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_book'],
    'q02': ['tokenize'],
    'q03': ['UniformLM'],
    'q04': ['UnigramLM'],
    'q05': ['NGramLM']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True

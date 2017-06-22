"""
    Author: Brandon Chambers

    TCSS 435 A.I.
    Programming Assignment #3 - Natural Language Processing
    5/20/2017
    
        This program will fit a tri-gram language model to English and then use it to generate new
    English text.  
        A unigram model consists of a single probability distribution P(W), over the 
    set of all words.  
        A bigram model consists of two probability distributions: P(W(0)) and 
    P(W(i) | W(i-1)).  The first distribution is just the probability of the first word in a 
    document.  The second distribution is the probability of seeing a word W(i) given that the 
    previous word was W(i-1).
        A trigram model consists of three probability distributions: P(W(0)), P(W(1) | W(0)), and
    P(W(i) | W(i-1),W(i-2)).  The first distribution is, as above, the probability of the first 
    word in the document.  The next distribution is the probability of the second word given the 
    first one.  And the third distribution is the probability of the ith word given the two 
    preceding words. 
        Given a set of documents, in this case various novels/short stories, your job in this 
    assignment is to fit a trigram model of English.  It's recommended that you do this by using a
    hash table in which you hash on word W(i-1).  It also contains a pointer to a second level of 
    linked lists that link the words that appeared at position W(i).
"""

import sys, copy, math, random, time

# Globals
_testers_ = dict()
_uniforms_ = dict()
_unigrams_ = dict()
_bigrams_ = dict()
_trigrams_ = dict()
unknown = "<?>"
vocab_len = 0
total_len = 0
# Global Constants
START = "*START*"
STOP = "*STOP*"
NEW_TEXT_WORD_COUNT = 1000
MAX_NUMBER_TO_PRINT = 40
# Hardcoded lambda values, from my testing these provide the best/lowest perplexity
UNIF_P = 0.0025  # Uniform Probability
UNIG_P = 0.0175  # Unigram Probability
BI_P = 0.03  # Bigram Probability
TRI_P = 0.95  # Trigram Probability


def uniform_model():
    """

    :return: uniformed dictionary
    """
    print "Building Uniformity Dictionary{...}",
    sys.stdout.flush()
    uni_dict = {k: 1.0 / vocab_len for k in _unigrams_}
    print "Task Complete!"
    return uni_dict


def unigram_model():
    """

    :return: dictionary of unigrams
    """
    print "Converting to Unigrams, Building Dictionary{...}",
    sys.stdout.flush()
    unigrams = {k: float(_unigrams_[k]) / total_len for k in _unigrams_}
    print "Task Complete!"
    return unigrams


def bigram_model(_t):
    """

    :param _t:
    :return:
    """
    train_lst = copy.deepcopy(_t)
    print "Performing Bigram Modeling-->\n    Appending to Bigram List[...]",
    sys.stdout.flush()
    bigrams = []  # temp list of bigrams, ultimately stored in global dictionary
    train_lst.insert(0, START)
    for i in xrange(len(train_lst) - 1):
        bigrams.append((train_lst[i], train_lst[i + 1]))
    bigrams.sort()

    print "Task Complete!\n    Moving Bigrams to Dictionary{...}",
    sys.stdout.flush()
    word1 = bigrams[0][0]
    word2 = bigrams[0][1]
    tmp_lst = []
    word1_cnt = 0
    word2_cnt = 0
    for ch in bigrams:
        if ch[0] == word1:
            word1_cnt += 1
            if ch[1] == word2:
                word2_cnt += 1
            else:
                tmp_lst.append(((word1, word2), word2_cnt))
                word2 = ch[1]
                word2_cnt = 1
        else:  # capture the last word.
            _bigrams_[(word1,
                       word2)] = float(word2_cnt) / word1_cnt
            for pair in tmp_lst:
                _bigrams_[pair[0]] = float(pair[1]) / word1_cnt
            word1 = ch[0]
            word2 = ch[1]
            tmp_lst = []
            word1_cnt = 1
            word2_cnt = 1

    # Put the result in global dictionary
    _bigrams_[(word1, word2)] = float(word2_cnt) / word1_cnt
    print ("Task Complete!\n    Bigram Count: %d" % len(_bigrams_))
    return _bigrams_


def trigram_model(_t):
    """

    :param _t:
    :return:
    """
    train_lst = copy.deepcopy(_t)
    print "Performing Trigram Modeling-->\n    Appending to Trigram List[...]",
    sys.stdout.flush()
    trigrams = []  # temp list of trigrams, ultimately stored in global dictionary
    train_lst.insert(0, START)
    train_lst.insert(0, START)
    for i in xrange(len(train_lst) - 2):
        trigrams.append((train_lst[i], train_lst[i + 1], train_lst[i + 2]))
    trigrams.sort()

    print "Task Complete!\n    Moving Trigrams to Dictionary{...}",
    sys.stdout.flush()
    word1 = trigrams[0][0]
    word2 = trigrams[0][1]
    word3 = trigrams[0][2]
    tmp_lst = []
    wrd2_cnt = 0
    wrd3_cnt = 0
    for ch in trigrams:
        if ch[0] == word1:
            if ch[1] == word2:
                wrd2_cnt += 1
                if ch[2] == word3:
                    wrd3_cnt += 1
                else:
                    tmp_lst.append(((word1, word2, word3), wrd3_cnt))
                    word3 = ch[2]
                    wrd3_cnt = 1
            else:
                _trigrams_[(word1, word2, word3)] = float(wrd3_cnt) / wrd2_cnt
                for triple in tmp_lst:
                    _trigrams_[triple[0]] = float(triple[1]) / wrd2_cnt
                word2 = ch[1]
                word3 = ch[2]
                tmp_lst = []
                wrd2_cnt = 1
                wrd3_cnt = 1
        else:
            _trigrams_[(word1, word2, word3)] = float(wrd3_cnt) / wrd2_cnt
            word1 = ch[0]
            word2 = ch[1]
            word3 = ch[2]
            tmp_lst = []
            wrd2_cnt = 1
            wrd3_cnt = 1

    # Put the result in global dictionary
    _trigrams_[(word1, word2, word3)] = float(wrd3_cnt) / wrd2_cnt
    print ("Task Complete!\n    Trigram Count: %d" % len(_trigrams_))
    return _trigrams_


def tuple_tostring(_t):
    """

    :param _t:
    :return:
    """
    sb = ""  # string builder
    for i in xrange(len(_t)):
        if i == 0:
            sb += str(_t[len(_t) - 1]) + " "
        elif i == len(_t) - 1:
            sb += str(_t[0]) + " "
        else:
            sb += str(_t[len(_t) - 1 - i]) + " "
    return sb


def print_grams(_u, _b, _t):
    unigram_lst = [(k, _u[k]) for k in _u]
    unigram_lst.sort(key=lambda x: x[1])
    print "----------------------------------------------\n"

    print "\nPrinting most common n-grams..."
    print "\n----------------------------------------------"

    print "Some of the most common unigrams:\n"
    for i in xrange(30):
        print "  ",
        print unigram_lst[len(unigram_lst) - 1 - i]
    print "----------------------------------------------\n"

    bigram_lst = [(k, _b[k]) for k in _b]
    bigram_lst.sort(key=lambda x: x[1])
    most_common_bigrams = filter(lambda x: x[1] > 0.85, bigram_lst)
    print "\n----------------------------------------------"

    print "Some of the most common bigrams (p > 0.85):\n"
    for i in xrange(MAX_NUMBER_TO_PRINT):
        print "  ",
        print most_common_bigrams[len(most_common_bigrams) - 1 - i]
    print "----------------------------------------------\n"

    trigram_lst = [(k, _t[k]) for k in _t]
    trigram_lst.sort(key=lambda x: x[1])
    most_common_trigrams = filter(lambda x: x[1] > 0.85, trigram_lst)
    print "\n----------------------------------------------"
    print "Some of the most common trigrams (p > 0.85):\n"
    for i in xrange(MAX_NUMBER_TO_PRINT):
        print "  ",
        print most_common_trigrams[len(most_common_trigrams) - 1 - i]
    print "----------------------------------------------\n"


def mymain(to_test, to_train):
    """
    Driver function.
    :param to_test: 
    :param to_train: 
    :return: 
    """
    global vocab_len, total_len, _uniforms_, _unigrams_, _bigrams_, _trigrams_
    _t_sb = ""  # tester string builder
    train_sb = ""  # trainer string builder
    strings = ""

    start_time = time.time()

    gatekeeper = 8.09
    # check that hardcoded total probability equals 1.0
    if (UNIF_P + UNIG_P + BI_P + TRI_P) * gatekeeper != gatekeeper:
        print "Error: Hardcoded lambda values must sum to 1.0."

    print "\n\n\t\t******************* Begin NGram Model Training ********************\n" \
          "\t-----------------------------------------------------------------------------------" \
          "\n\nOpening Training File...",
    sys.stdout.flush()
    for _file in to_train:
        train_line = open(_file, "r")
        strings = train_line.readline()
        while strings != "":
            train_sb += strings
            strings = train_line.readline()
        train_line.close()
    print "Task Complete!"

    print "Appending Tokenized Training Text to List[...]",
    sys.stdout.flush()
    train_sb = train_sb.lower()
    train_lst = "".join(train_sb.split("\n")).split(" ")
    if len(train_lst) == 0:
        return

    sorted_train_lst = copy.deepcopy(train_lst)
    sorted_train_lst.sort()
    print "Task Complete!"

    print "Adding NGram Counts to Dictionary{...}",
    sys.stdout.flush()
    _unigrams_[unknown] = 0
    cur_elem = sorted_train_lst[0]
    for item in sorted_train_lst:
        if item != cur_elem:
            cur_elem = item
        if cur_elem not in _unigrams_:
            _unigrams_[cur_elem] = 0
        _unigrams_[cur_elem] += 1
    print "Task Complete!"

    # To avoid destructive deletion of keys, create a buffer of keys to be deleted after the loop.
    print "Replacing Rare Words with <?> Token...",
    sys.stdout.flush()
    to_delete = []
    for key in _unigrams_:
        if _unigrams_[key] < 5:
            _unigrams_[unknown] += _unigrams_[key]
            to_delete.append(key)
    for key in to_delete:
        del _unigrams_[key]

    train_lst = map(lambda x: unknown if x in to_delete else x, train_lst)
    print "Task Complete!"

    vocab_len = len(_unigrams_)
    total_len = len(train_lst)

    print "\n----------------------------------------------"
    print ("    Vocabulary Size: %d" % vocab_len)
    print ("    Total Words: %d" % total_len)
    print "----------------------------------------------\n"

    print "\n----------------------------------------------"
    print "Running Probability Models:"
    _uniforms_ = uniform_model()
    _unigrams_ = unigram_model()
    _bigrams_ = bigram_model(train_lst)
    _trigrams_ = trigram_model(train_lst)

    print_grams(_unigrams_, _bigrams_, _trigrams_)
    print "Training Complete!\n"

    print "----------------------------------------------"
    print "Beginning Testing:"
    print "Opening test file...",
    sys.stdout.flush()
    if to_test != "0":
        testfd = open(to_test, "r")
        strings = testfd.readline()
        while strings != "":
            _t_sb += strings
            strings = testfd.readline()
        testfd.close()
    print "Done!"

    print "Creating testing text token list[...]",
    sys.stdout.flush()
    _t_sb = _t_sb.lower()
    test_lst = "".join(_t_sb.split("\n")).split(" ")
    if len(test_lst) == 0:
        return
    sorted_test_lst = copy.deepcopy(test_lst)
    sorted_test_lst.sort()
    print "Done!"

    print "Creating counting dictionary{...}",
    sys.stdout.flush()
    _testers_[unknown] = 0
    cur_elem = sorted_test_lst[0]
    for item in sorted_test_lst:
        if item != cur_elem:
            cur_elem = item
        if cur_elem not in _testers_:
            _testers_[cur_elem] = 0
        _testers_[cur_elem] += 1
    print "Done!"

    print "\nReplacing all infrequent words with <?> token...",
    sys.stdout.flush()
    to_delete = []
    for key in _testers_:
        if _testers_[key] < 5:
            _testers_[unknown] += _testers_[key]
            to_delete.append(key)
    for key in to_delete:
        del _testers_[key]
    test_lst = map(lambda x: unknown if x in to_delete else x, test_lst)
    print "Done!"

    print "Replacing words not in vocabulary with <?> token...",
    sys.stdout.flush()
    for i in xrange(len(test_lst)):
        if test_lst[i] not in _unigrams_:
            test_lst[i] = unknown
    print "Done!"

    new_unigram_lst = []
    new_bigram_lst = []
    new_trigram_lst = []

    print "\nRebuilding NGram Model Lists-->"
    print "  Creating Unigram List[...]",
    sys.stdout.flush()
    new_unigram_lst = copy.deepcopy(test_lst)
    print "Done!"

    print "  Creating Bigram List[...]",
    sys.stdout.flush()
    test_lst_copy = copy.deepcopy(test_lst)
    test_lst_copy.insert(0, START)
    for i in xrange(len(test_lst_copy) - 1):
        new_bigram_lst.append((test_lst_copy[i], test_lst_copy[i + 1]))
    print "Done!"

    print "  Creating Trigram List[...]",
    sys.stdout.flush()
    test_lst_copy = copy.deepcopy(test_lst)
    test_lst_copy.insert(0, START)
    test_lst_copy.insert(0, START)
    for i in xrange(len(test_lst_copy) - 2):
        new_trigram_lst.append((test_lst_copy[i], test_lst_copy[i + 1], test_lst_copy[i + 2]))
    print "Done!"

    uniform_len = len(test_lst)

    print "\nBegin Linear Interpolation Perplexity Calculation-->"
    print ("  Probabilities: %0.4f %0.4f %0.4f %0.4f" % (UNIF_P, UNIG_P, BI_P, TRI_P))

    sigma = 0.0
    uniform_prob = 0.0
    unigram_prob = 0.0
    bigram_prob = 0.0
    trigram_prob = 0.0
    pow10 = 0
    for i in xrange(uniform_len):
        if test_lst[i] in _uniforms_:
            uniform_prob = _uniforms_[test_lst[i]]
        if new_unigram_lst[i] in _unigrams_:
            unigram_prob = _unigrams_[new_unigram_lst[i]]
        if new_bigram_lst[i] in _bigrams_:
            bigram_prob = _bigrams_[new_bigram_lst[i]]
        if new_trigram_lst[i] in _trigrams_:
            trigram_prob = _trigrams_[new_trigram_lst[i]]

        weighted_val = (UNIF_P * uniform_prob + UNIG_P * unigram_prob
                        + BI_P * bigram_prob + TRI_P * trigram_prob)
        if weighted_val == 0.0:
            continue
        sigma += math.log(weighted_val)
        uniform_prob = 0.0
        unigram_prob = 0.0
        bigram_prob = 0.0
        trigram_prob = 0.0
        pow10 += 1

    perplexity = math.exp(-sigma / uniform_len)
    print "  Done..." + ("\n\tPerplexity = %0.6f" % perplexity) + "%"

    total_time = time.time() - start_time
    print "\nElapsed execution time:\n" + ("\t-- %0.4f sec --" % total_time)

    print "\nWriting 1000 words to new text files.\n  OUTPUT1.txt for training files <= 3" \
          "\n  OUTPUT2.txt for training files > 3."
    if len(to_train) <= 3:
        with open('OUTPUT1.txt', 'w') as out_file:
            out_file.write("  Training Files <= 3\n  2 Sherlock Holmes files...\n")
            for i in xrange(NEW_TEXT_WORD_COUNT / 3):
                out_file.write(tuple_tostring(random.choice(new_trigram_lst)))
                if i % 7 == 0 and i != 0:
                    out_file.write("\n")
    if len(to_train) > 3:
        with open('OUTPUT2.txt', 'w') as out_file:
            out_file.write("  Training Files > 3\n  All 6 files...\n")
            for i in xrange(NEW_TEXT_WORD_COUNT / 3):
                out_file.write(tuple_tostring(random.choice(new_trigram_lst)))
                if i % 7 == 0 and i != 0:
                    out_file.write("\n")
    print 'Done!\n'

    print "\t--------------------------------------END----------------------------------------"
    print "\t    *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage:",
        print ("python %s " % sys.argv[0]), " [testfile] [trainingfile(s)]"
        exit(0)
    mymain(sys.argv[1], sys.argv[2:])

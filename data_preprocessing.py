import nltk

nltk.data.path.append('.')


# Split the data into sentences
def split_to_sentences(data):
    sentences = data.splitlines()
    # Additional clearing
    # - Remove leading and trailing spaces from each sentence
    # - Drop sentences if they are empty strings.
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    return sentences


# Tokenize the sentences into words
def tokenize_sentences(sentences):
    # Initialize the list of lists of tokenized sentences
    tokenized_sentences = []
    # Go through each sentence
    for sentence in sentences:
        # Convert to lowercase letters
        sentence = sentence.lower()
        # Convert into a list of words
        tokenized = nltk.word_tokenize(sentence)
        # append the list of words to the list of lists
        tokenized_sentences.append(tokenized)
    return tokenized_sentences


# Get the tokenized data
def get_tokenized_data(data):
    # Get the sentences by splitting up the data
    sentences = split_to_sentences(data)
    # Get the list of lists of tokens by tokenizing the sentences
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences


# Count words You won't use all the tokens (words) appearing in the data for training. Instead, you will use the more
# frequently used words.
def count_words(tokenized_sentences):
    word_counts = {}
    # Loop through each sentence
    for sentence in tokenized_sentences:
        # Go through each token in the sentence
        for token in sentence:
            # If the token is not in the dictionary yet, set the count to 1
            if not token in word_counts:
                word_counts[token] = 1
            # If the token is already in the dictionary, increment the count by 1
            else:
                word_counts[token] += 1
    return word_counts


# Function that takes in a text document and a threshold count_threshold.
def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    # Initialize an empty list to contain the words that
    # appear at least 'minimum_freq' times.
    closed_vocab = []
    # Get the word counts of the tokenized sentences
    # Use the function that you defined earlier to count the words
    word_counts = count_words(tokenized_sentences)
    for word, cnt in word_counts.items():
        # check that the word's count
        # is at least as great as the minimum count
        if cnt >= count_threshold:
            # append the word to the list
            closed_vocab.append(word)
    return closed_vocab


# Function replaces all the words not in closed vocabulary by token <unk>
def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    # Place vocabulary into a set for faster search
    vocabulary = set(vocabulary)

    # Initialize a list that will hold the sentences
    # after less frequent words are replaced by the unknown token
    replaced_tokenized_sentences = []

    # Go through each sentence
    for sentence in tokenized_sentences:

        # Initialize the list that will contain
        # a single sentence with "unknown_token" replacements
        replaced_sentence = []
        # for each token in the sentence
        for token in sentence:  # complete this line
            # Check if the token is in the closed vocabulary
            if token in vocabulary:
                # If so, append the word to the replaced_sentence
                replaced_sentence.append(token)
            else:
                # otherwise, append the unknown token instead
                replaced_sentence.append(unknown_token)

        # Append the list of tokens to the list of lists
        replaced_tokenized_sentences.append(replaced_sentence)

    return replaced_tokenized_sentences


# Pre-process data
def preprocess_data(train_data, test_data, count_threshold, unknown_token="<unk>",
                    get_words_with_nplus_frequency=get_words_with_nplus_frequency,
                    replace_oov_words_by_unk=replace_oov_words_by_unk):

    # Get the closed vocabulary using the train data
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)

    # For the train data, replace less common words with "<unk>"
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary, unknown_token)

    # For the test data, replace less common words with "<unk>"
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token)

    return train_data_replaced, test_data_replaced, vocabulary



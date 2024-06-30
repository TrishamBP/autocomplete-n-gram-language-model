import random
import nltk

nltk.data.path.append('.')

from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import count_n_grams, get_suggestions

## Step 1 - Load and Pre Process the Data
# Load the data
with open("./data/en_US.twitter.txt", "r", encoding="utf-8") as f:
    data = f.read()
# Split the data into training sets and test sets
tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]
# Preprocess the train and test data
minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data,
                                                                        test_data,
                                                                        minimum_freq)


## Autocomplete
def get_user_input_suggestions(vocabulary, n_gram_counts_list, k=1.0):
    while True:
        user_input = input("Enter a sentence (or 'q' to quit): ").lower().strip()
        if user_input == 'q':
            break

        # Tokenize the user input
        tokens = nltk.word_tokenize(user_input)

        # Get suggestions
        suggestions = get_suggestions(tokens, n_gram_counts_list, vocabulary, k)

        # Sort suggestions by probability (highest first) and print them
        sorted_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)

        print("\nSuggestions:")
        for i, (word, prob) in enumerate(sorted_suggestions, 1):
            print(f"{i}. {word} (probability: {prob:.6f})")

        print("\n")


# In your main code, after preprocessing:
n_gram_counts_list = []
for n in range(1, 5):  # Assuming you want to use 1-gram to 4-gram models
    n_gram_counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(n_gram_counts)

# Call the function
get_user_input_suggestions(vocabulary, n_gram_counts_list)

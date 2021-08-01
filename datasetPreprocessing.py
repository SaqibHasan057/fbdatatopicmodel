import pandas as pd
import re
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize




def load_dataset(filename):
    dataset = pd.read_csv(filename,encoding="utf-8")
    messages = dataset["Message"]
    messages = messages.dropna()
    messages = messages.tolist()
    print("Dataset loaded!!")
    return messages



def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_url(string):
    return re.sub(r"http\S+", "", string)

def remove_bangla_number(string):
    return re.sub(r"^[১২৩৪৫৬৭৮৯০]+$", "", string)


def remove_punctuation(string):
    return re.sub(r"[\p{P}]", "", string)



def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    filtered_text = " ".join(filtered_text)
    return filtered_text







## lemmatize string
def lemmatize_word(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    # provide context i.e. part-of-speech
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
    lemmas = " ".join(lemmas)
    return lemmas



def text_preprocessing(arr):
    ##Convert to lower case
    x = [item.lower() for item in arr]
    print("Converted to lowercase!!")

    ##Remove numbers
    x = [re.sub(r'\d+', '', i) for i in x]

    print("Numbers removed!!")

    #Remove bangla numbers
    x = [remove_bangla_number(i) for i in x]

    print("Bangla numbers removed!!")

    ##Remove emojis
    x = [remove_emoji(i) for i in x]

    print("Emojis removed!!")

    ##Remove urls
    x = [remove_url(i) for i in x]

    print("URLs removed!!")

    ##Remove punctuations
    #translator = str.maketrans(string.punctuation + '—…'+"।", ' ' * (len(string.punctuation) + 3))
    #x = [i.translate(translator) for i in x]
    x = [remove_punctuation(i) for i in x]

    print("Punctuations removed!!")

    ##Remove whitespaces
    x = [" ".join(i.split()) for i in x]

    print("Whitespaces removed!!")

    ##Remove stopwords
    x = [remove_stopwords(i) for i in x]

    print("Stopwords removed!!")

    ##Lemmatization
    x = [lemmatize_word(i) for i in x]

    print("Lemmatization done!!")

    count = 1
    for i in x:
        print(count,i)
        count+=1


    print("Text processing done!!")

    return x

def save_processed_dataset(filename,x):
    f = open(filename,"wb")
    pickle.dump(x,f)
    f.close()
    print("Processed dataset saved!!")








original_file = "dataset.csv"
new_file = "processed_dataset.pkl"

arr = load_dataset(original_file)
arr = text_preprocessing(arr)
save_processed_dataset(new_file,arr)
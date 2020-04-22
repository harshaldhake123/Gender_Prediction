import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


def loadDataset(filename, label):
    # clean the datasets using RegEx
    pattern = re.compile(r'([a-zA-Z]{3,})+')
    fh = open(filename, encoding="utf-8")
    names = []
    for x in fh:
        if re.search(pattern, x):  # check the pattern for each name
            names.append((re.search(pattern, x).group(1), label))  # tuple of (name,label)
    return names  # return the list of tuples of a particular category

    # to concat the labelled categories,shuffle elements and splitting names and labels


# load each file as  lists of tuples(name,gender) after cleaning the data

# building corpus by:
# 1.merging both data lists
# 2.shuffling all data
# 3.separating the names and genders ->
#       a.last 4 characters of name go in names
#       b.labels go in separate list of genders
def buildCorpus(set1, set2):
    corpus = list(set1)  # 1st list goes into corpus
    corpus.extend(set2)  # second list extends the corpus
    random.shuffle(corpus)  # shuffling
    random.shuffle(corpus)
    names = []
    genders = []
    for name, label in corpus:
        names.append(name[-4:].lower())  # getting features by considering last 4 name characters
        genders.append(label)  # separate list for  categories
    return names, genders
    # return total data in two separate lists of names and genders(original data was
    # separated as all names as per gender
    # while now data is segregated as all names and all genders)


def trainTestAlgo(names, genders):
    # split the total name and gender data into train and test sets
    # vectorizer to create a vocabulary of features
    # fit-> create a vocabulary of words and assign a numeric value for each word: Eg.{'nita': 1916, 'nish': 1915}
    # create a bag of words ie a sparse matrix where each row is a record and column specify the vocabulary,
    # value is 1 is the word is present in  the record else 0
    # from NaiveBayes:  BernoulliNB algorithm for binary classification, else for multiple labels use MultinomialNB
    # train the algorithm using features from bag of words and genders from the training set of data
    nametrain, nametest, gendertrain, gendertest = train_test_split(names, genders, test_size=0.1)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(nametrain)
    bow = vectorizer.transform(nametrain)
    algo = BernoulliNB()
    algo.fit(bow, gendertrain)

    errCount = 0
    for name, gender in zip(nametest, gendertest):
        name_bow = vectorizer.transform([name])  # each testname creates a bow
        # print(name_bow)
        result = algo.predict(name_bow)  # predict the gender
        # print(result)
        if gender != result[0]:
            errCount += 1
    print("Train Data:", len(nametrain), "\nTest Data:", len(nametest), "\nTesting Accuracy: ",
          100 - float((errCount / len(nametest)) * 100).__round__(2), '%', "\nErrors:", errCount,
          )
    while True:
        input("Enter a name to predict:[0  to exit]")


def main():
    mnames = loadDataset("Indian-Male-Names.csv", 'M')  # list of tuples
    fnames = loadDataset("Indian-Female-Names.csv", 'F')  # list of tuples
    names, gender = buildCorpus(mnames, fnames)
    trainTestAlgo(names, gender)


main()


from LoadDataSet import documents
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

stopwordPath = "/home/satya/Downloads/stopwords.txt"
stopwordsFile = open(stopwordPath,'r')

stopwords = list(stopwordsFile.read().split('\r\n'))

print "STOPWORDS"
print stopwords

# A LIST OF LIST OF LIST
bagOfWords = list()

def tokenizer(eachDoc):
    bagOfWords_eachDoc = list()

    for eachSentence in eachDoc:
        bagOfWords_eachSentence = list()
        bagOfWords_eachSentence = eachSentence.lower().encode("ascii","ignore").split()

        bagOfWords_eachSentence_withoutStopWords = [word for word in bagOfWords_eachSentence if word not in stopwords]

        bagOfWords_eachDoc.append(bagOfWords_eachSentence_withoutStopWords)

    bagOfWords.append(bagOfWords_eachDoc)


for eachDoc in documents:
    tokenizer(eachDoc)


#BAG OF WORDS WITH STOPWORDS REMOVED
print bagOfWords


sentencesWithStopwordsRemoved = list()

def sentenceGenerator(docList):
    sentenceList = list()
    for eachSentenceList in docList:
        sentence = ' '.join(eachSentenceList)
        sentenceList.append(sentence)

    sentencesWithStopwordsRemoved.append(sentenceList)

for eachDocumentList in bagOfWords:
    sentenceGenerator(eachDocumentList)

#LIST OF LIST FOR EACH DOCUMENT IN SENTENCE FORM
print sentencesWithStopwordsRemoved

listOf_tf = list() #CONTAINS ALL TF SCORES,LOCAL

vectorizer = CountVectorizer()



for eachSentenceList in sentencesWithStopwordsRemoved:
    vectorizer.fit(eachSentenceList)
    tf_forEachDoc = vectorizer.transform(eachSentenceList).toarray()

    listOf_tf.append(tf_forEachDoc)



#FINAL LIST
sortedSuperList = list()

for i in xrange(listOf_tf.__len__()):

    sortedSubList = list()

    for eachTfScoreList in listOf_tf[i]:
        sortedList = sorted(eachTfScoreList,reverse=True)
        if sortedList.__len__()>50:
            sortedSubList.append(sortedList[:50])

        else:
            zeros = list()
            for i in xrange(50):
                zeros.append(0)
            sortedList.extend(zeros)
            sortedSubList.append(sortedList[:50])

    sortedSuperList.append(sortedSubList)




#FINAL 2D ARRAY
listOfArrays = list()


ex = np.array(sortedSuperList[0])


for eachDocList in sortedSuperList:
    array = np.array(eachDocList)
    listOfArrays.append(array)


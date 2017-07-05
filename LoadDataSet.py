from xml.dom.minidom import parse
import xml.dom.minidom

corpus_xmlPath = "/home/satya/Downloads/bc3.1.0/corpus.xml"
DOMTree = parse(corpus_xmlPath)

#print DOMTree

collection = DOMTree.documentElement

docs = collection.getElementsByTagName("DOC")
#print docs.length

documents = list()

for eachDoc in docs:
    tempList = list()

    subject = eachDoc.getElementsByTagName("Subject")[0]
    #print subject.childNodes[0].data
    tempList.append(subject.childNodes[0].data)

    sentences = eachDoc.getElementsByTagName("Sent")
    #print sentences

    for eachSentence in sentences:
        #print eachSentence.childNodes[0].data
        tempList.append(eachSentence.childNodes[0].data.encode("ascii","ignore"))

    documents.append(tempList)

print documents



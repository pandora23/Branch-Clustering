
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus import stopwords, words
from itertools import product
from nltk import FreqDist
import pickle
import nltk
import json
import urllib2
import urllib
import re
import Queue
import searchEngine


#triple extraction
def tripleExtraction(text, stopWords):
    text = [word for word in text.split() if word.lower() not in stopWords]
    triples = list()
    for word1 in text:
        for word2 in text:
            for word3 in text:
                if word1 != word2 and word1 != word3 and word2 != word3:
                    newTriple = (word1, word2, word3)
                    triples.append(newTriple)
    return triples

#test triple extraction
sentence = "two fish.n.01 are in a tank one looks to the other and asks how do you drive this thing"
stopwords = nltk.corpus.stopwords.words('english')
words = nltk.corpus.words.words('en')
otherWords = ['index', 'account', 'may']
triples = tripleExtraction(sentence, stopwords)
#print(triples)

#get number of search results
##def showHits(word1, word2):
##    cx = '015484707549482425261:b5q1t9m6bpa'
##    key = 'AIzaSyCUegTaNsC6224cy4VNKOSWr8isztlaqh4'
##    #url = 'https://www.googleapis.com/customsearch/v1?key=INSERT_YOUR_API_KEY&cx='
##    #url = 'http://stackoverflow.com/questions/1657570/google-search-from-a-python-app'
##    searchString = '+' + word1 + ' +' + word2
##    query = urllib.urlencode({'q': searchString})
##    #url2 = 'http://www.google.com/search?%s' % query
##    url2 = 'https://www.googleapis.com/customsearch/v1?key=AIzaSyCUegTaNsC6224cy4VNKOSWr8isztlaqh4&cx=015484707549482425261:b5q1t9m6bpa&%s' % query
##    url = 'http://ajax.googleapis.com/ajax/services/search/web?v=1.0&%s' % query
##    print(url2)
##    search_response = urllib2.urlopen(url2)
##    search_results = search_response.read().decode("utf8")
##    results = json.loads(search_results)
##    data = results['searchInformation']
##    print('Total results: %s' % data['totalResults'])
##    return data['totalResults']



class taxonomyTree(object):
    def __init__(self):
        self.children = list()
        self.childTreeLinks = list()
        self.data = ""
        self.parent = None
        


def buildTaxonomyUsingWebMining(parent, rootString, stopwords, words, htmlWords, stringSoFar, depth, targetDepth):
    
    
    if depth > targetDepth:
        return

    newDepth = depth + 1
    
    node = taxonomyTree()
    
    node.data = rootString

    resultsToFind = 50
    
    results = searchEngine.getSearchResultsBing(stringSoFar, resultsToFind)

    rootWords = nltk.word_tokenize(stringSoFar)
    print("rootwords:")
    print(rootWords)
    
    #set of each word
    wordSets = list()

    #raw data for files
    
    allData = []

    numTitles = len(results)
    print("Number of results: " + str(numTitles))
   
    #go through each result url and get the text
    for result in results:
        #print(results['formattedUrl'])
        print(result)
        accepted = False
        try:
            #go to site and get webpage

            try:
                #url = "https://" + result
                url = result
                #print(url)
                page = urllib2.urlopen(url).read()

            except:
                #url = "http://" + result
                page = urllib2.urlopen(url).read()
                

            #print(url)

            accepted = True
            
        except:
            
            pass
            

        #find indexes of rootwords
        if accepted == True:

            print("accepted")

            page = nltk.clean_html(page)

            page = re.sub('[;:.,`\"\'-]','',page)

            page = nltk.word_tokenize(page)
            
            pageTokens = []
            
            for word in rootWords:
                print(word)
                #grab all indexes
                indexList = []
                i = -1
                try:
                    while True:
                        i = page.index(word, i+1)
                        indexList.append(i)
                except:
                    pass

                print(indexList)
                #DISTANCE TO TARGET
                distanceFromTarget = 5
                
                try:
                    for index in indexList:
                        if index < distanceFromTarget:
                            pageTokens.append(page[:(index+distanceFromTarget)])
                        if (index + distanceFromTarget) >= len(page):
                            pageTokens.append(page[(index-distanceFromTarget):])
                        else:
                            pageTokens.append(page[(index-distanceFromTarget):(index+distanceFromTarget)])
                except:
                    pass
            print(pageTokens)

            docTokens = []

            try:
                #print("PT")
                #print(pageTokens)
                
                
                if pageTokens != []:
                    for set1 in pageTokens:
                        for token in set1:
                            #print(type(token))
                            if isinstance(token, str):
                                if token.lower() not in stopwords and token.lower() and token.lower() in words:
                                    docTokens.append(token.lower())
                    
            except:
                pass
                
            if docTokens != []:
                wordSets.append(set(docTokens))
                allData.append(docTokens)
            
#REPLACE WITH TFIDF


    #words common to all
    try:
        commonWords = set.intersection(*wordSets)
    except:
        print('none')
        commonWords = []
        
    allwordslist = list()

    for lists in allData:
        for token in lists:
            allwordslist.append(token)

    #get frequency distribution
    fdist = FreqDist(allwordslist)

    print(depth);
    
    if(depth == 0):
        branchFactor = 5;
    else:
        branchFactor = 50;
    
    node.children=fdist.keys()[:branchFactor]

    print("children at depth " + str(depth) + " :")
    print(node.children)
    #begin tree induction
    for child in node.children:
          
          string = child + " " + stringSoFar
          #print(string)
          branch = buildTaxonomyUsingWebMining(node, child, stopwords, words, otherWords, string, newDepth, targetDepth)
          node.childTreeLinks.append(branch)


    return node

tree = buildTaxonomyUsingWebMining(None, "tank", stopwords, words, otherWords, "tank", 0, 2)


#pickle the tree
output = open('fish5.pkl','wb')

pickle.dump(tree,output)




featureVector = []

#breadth first extraction of features
level = 0

#initialize word vector for tree
treeWordFrequencyVector = [0 for x in range(len(words))]

que = Queue.Queue()

que.put(tree)

while que.empty() == False:
    nextChild = que.get()

    print("child at level: " + str(level))
    print(nextChild.data)

    try:
        wordIndex = words.index(nextChild.data)
        treeWordFrequencyVector[wordIndex] = treeWordFrequencyVector[wordIndex] + 1
    except:
        pass
    
    for link in nextChild.childTreeLinks:
        if link != None:
            que.put(link)


def depthFirstTreeIterate(treeNode):

    print(treeNode.data)
    yield treeNode.data
    if treeNode.childTreeLinks != []:
        for child in treeNode.childTreeLinks:
            depthFirstTreeIterate(child)
    return


#branch seperations


    

#cluster trees, unsupervised

#attributes

#1.  word frequency vector for entire tree
#  [x1...xn]
#try to use nltk.corpus.words as vector

#synset based clustering

#hyponym

#clustering ambiguity:  find a feature that will allow one to cluster ambiguous and non-ambiguous words

#linkage itself is based on occuring on a branch for a given word

#distances between words of the children.



#lixical diversity

#visualize clustering

#mixmatch joke clustering based on linkage patterns

#2  x1 =

#sentiment





#test
##fishtank = showHits("fish", "tank")
##fishdrive = showHits("fish", "drive")
##tankdrive = showHits("tank", "drive")
##fishfish = showHits("fish", "fish")
##drivedrive = showHits("drive", "drive")
##tanktank = showHits("tank", "tank")
##total1 = fishtank / (fishfish + tanktank)
##total2 = fishdrive / (fishfish + drivedrive)
##total3 = tankdrive / (tanktank + drivedrive)
##
##print("fish tank : " + fishtank)
##print("fish drive : " + fishdrive)
##print("tank drive : " + tankdrive)
##print(fishfish)
##print(drivedrive)
##print(tanktank)
##print("total1 " + total1)
##print("total2 " + total2)
##print("total3 " + total3)
##
##    

def getSemanticSimilarity(word1, word2):
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    sem1,sem2=wn.synsets(word1),wn.synsets(word2)
    scoreMax = 0
    for i,j in list(product(sem1,sem2)):
        #score = i.res_similarity(j, brown_ic)
        scoreMax = score if scoreMax < score else scoreMax
    print(scoreMax)
    return scoreMax

#test heuristic
def getHumorScoreUsingTriples(triples):
    count = 0
    for triple in triples:
        #similarity between X and Y
        sim1 = getSemanticSimilarity(triple[0], triple[1])
        #sim between X and Z
        sim2 = getSemanticSimilarity(triple[0], triple[2])
        #between Y and Z
        sim3 = getSemanticSimilarity(triple[1], triple[2])
        print(triple)
        print(sim1, sim2, sim3)
        
##getHumorScoreUsingTriples(triples)
##
##
##words = nltk.corpus.cmudict.words()
##
##
##
##z =[w for w in words if len(w) >= 12]
##
##
##for a in range(100):
##    print(z[a])
##    print(wn.synsets(z[a]))


#test



#phonetic dictionary to check if things are phonetically related

entries = nltk.corpus.cmudict.entries()

def phoRelated(word1, word2, phoneticDict):
    phonetics1 = [pron for x,pron in entries if x == word1][0]
    phonetics2 = [pron for x,pron in entries if x == word2][0]
    n = len(phonetics1)
    if n >= 1:
        for i in xrange(len(phonetics2)):
            #print(phonetics1)
            #print(phonetics2[i:i+n])
            if phonetics1 == phonetics2[i:i+n]:
                #print(phonetics1)
                return True

#test
#phoRelated('bark', 'ark', entries)
#getSemanticSimilarity('bark', 'ark')


#test python template

def testTemplate(word1, word2, word3):
    phoRelated1 = phoRelated(word1, word3, entries)
    #semRelated1 = getSemanticSimilarity(word1, word3)
    phoRelated2 = phoRelated(word2, word3, entries)
    #semRelated2 = getSemanticSimilarity(word2, word3)
    if phoRelated1:
        if phoRelated2:
            print(word1, word2, word3)


#test
for entry1 in entries[0:10]:
    for entry2 in entries[0:10]:
        for entry3 in entries[0:10]:
            if entry1 != entry2 and entry1 != entry3 and entry2 != entry3:
                #print(entry1[0], entry2[0], entry3[0])
                testTemplate(entry1[0], entry2[0], entry3[0])

    

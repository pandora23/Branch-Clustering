
import pickle



class taxonomyTree(object):
    def __init__(self):
        self.children = list()
        self.childTreeLinks = list()
        self.data = ""
        self.parent = None


def depthFirstTreeIterate(treeNode):
    if treeNode != None:
        if treeNode.childTreeLinks != []:
            print('next')
            for child in treeNode.childTreeLinks:
                for node in depthFirstTreeIterate(child):
                    yield node
        yield treeNode.data


#test grab the pickly file

testPk = open('fish6.pkl','rb')

tree = pickle.load(testPk)

print(tree)

treeTraversal = depthFirstTreeIterate(tree)

for node in treeTraversal:
    print(node)


def calcDistanceBetweenBranches(A,B):
    #print(A,B)
    commonWords = A.intersection(B)
    return len(set(commonWords))

    
def clusterBranchesUsingMaxCommonWords(numCluster, root):
    
    #extract words from children
    branchesData = list()
    subtrees = list()
    for branch in root.childTreeLinks:
        subtrees.append(branch)
        subtree = depthFirstTreeIterate(branch)
        branchSet = []
        for nodeData in subtree:
            branchSet.append(nodeData)
        branchSet = set(branchSet)
        #print(branchSet)
        branchesData.append(branchSet)

    print(branchesData)
    #establish base vector

    
    #knn: naively just grab the first set of branches
    centers = branchesData[:numCluster]

    print(centers)
    
    print(len(branchesData))


    
    
    numIter = 100

    for i in range(numIter):
        groupings = [[] for x in range(numCluster)]
        
        for branch in branchesData:
            #get closest center
            minDist = 0
            index = -1
            for center in centers:
                distance = calcDistanceBetweenBranches(branch,center)
                #commonWords = branch.intersection(center)
                #count = len(set(commonWords))
                if distance >= minDist:
                    minDist = distance
                    index = centers.index(center)
            
            groupings[index].append(branch)
            
##        print("CENTERS:")
##        print(centers)
##        print("GROUPINGS:")
##        print(groupings)

        #reassign centroids
        for group in groupings:
            index = groupings.index(group)
            print(index)
            minDist = 0
            candidate = None
            for branch in group:
                #calculate distance to other branches in group
                totalDist = 0
                for branch2 in group:
                    totalDist = totalDist + calcDistanceBetweenBranches(branch,branch2)
                if totalDist >= minDist:
                    minDist = totalDist
                    candidate = branch
            print(centers)
            centers[index] = candidate
            
                
        
    for group in groupings:
        print("Next cluster:")
        print(group)
            


        
    print("centers:")
    print(centers)
    




        
    #group branchSets into two groups based upon maximizing number of shared words
    #identify most disjoint branches
##    for branchSetA in branchesData:
##        for branchSetB in branchesData:
##            if branchSetA != branchSetB:
##                commonWords = branchSetA.intersection(branchSetB)
##                print(len(commonWords))
##                print(branchSetA, branchSetB)

def returnSubTreeGivenWordSet(wordSet, tree):
    print(1)

def subTreeCombination(treeA, treeB, rootWord):
    print(2)
    

testPk = open('fish5.pkl','rb')
tree = pickle.load(testPk)
clusterBranchesUsingMaxCommonWords(5, tree)

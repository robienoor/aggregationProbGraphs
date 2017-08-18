import numpy as np
import itertools
import warnings, math
from itertools import product, chain
warnings.filterwarnings('ignore')

def getInOutArgs(argMtx):
    sumArgs = argMtx.sum(axis=0)

    inArgs = np.argwhere(sumArgs == 0)
    inArgs = (inArgs.tolist())
    inArgs = [i[0] for i in inArgs]

    attacked = argMtx[inArgs, :]

    outArgs = (np.unique(np.where(attacked>0)[1])).tolist()

    return inArgs, outArgs

def calculateGroundedExtension(argMtx):
    argTypes = np.array(range(0, argMtx.shape[0]))
    ext = []
    terminate = False

    while not terminate:
        inArgs, outArgs = getInOutArgs(argMtx)

        if len(inArgs) > 0:
            ext.extend(list(argTypes[inArgs]))
            argsDelete = inArgs + outArgs
            argMtx = np.delete(argMtx, argsDelete, axis = 0)
            argMtx = np.delete(argMtx, argsDelete, axis = 1)
            argTypes = np.delete(argTypes, argsDelete)

        else:
            break

        sums = np.sum(argMtx.sum(axis=0))

        # If we find that the resulting graph (having deleted current in and out args) is got no more attacks in it then add 
        # whatever is leftover to the extension
        if sums == 0:
            ext.extend(list(argTypes))
            terminate = True

    return ext

def generatePermutations(posArgs, negArgs):

    posPerms = np.array(list(itertools.product([0,1], repeat=len(posArgs)*len(negArgs))))
    negPerms = np.array(list(itertools.product([0,1], repeat=len(negArgs)*len(posArgs))))

    allPermsList = [list(chain(*i)) for i in product(posPerms, negPerms)]

    posIdxs = []
    negIdxs = []

    currPos = 0
    for x in range(len(posArgs)):
        start = currPos + len(posArgs)
        posIdxs.extend(range(start, start + len(negArgs)))
        currPos += (len(posArgs) + len(negArgs))


    currNeg = len(posArgs)*(len(posArgs) + len(negArgs))
    for x in range(len(negArgs)):
        start = currNeg
        negIdxs.extend(range(start, start + len(posArgs)))
        currNeg += (len(posArgs) + len(negArgs))

    allPerms = np.zeros(shape=(len(allPermsList), (len(posArgs)+len(negArgs))**2))
    allPermsList = np.array(allPermsList)

    allPerms[:,posIdxs] = allPermsList[:,0:(len(posArgs)*len(negArgs))]
    allPerms[:,negIdxs] = allPermsList[:,(len(posArgs)*len(negArgs)):]

    return allPerms


def calculateProbabilityDistribution(posArgs, negArgs, rating):

    nargs = len(posArgs + negArgs)
    allPermutations = generatePermutations(posArgs, negArgs)

    # Determine the Polarity of the Post
    if rating == '-':
        groundedExtension = negArgs
    elif rating == '+':
        groundedExtension = posArgs
    else:
        groundedExtension = []


    # Iterate over the set of graphs that are possible (excluding circular attacks and same polarity attacks) to see which one's 
    # have a grounded extension matching the polarti
    acceptedGraphs = []
    for graph in allPermutations:
        attMtx = np.vstack( np.array_split(np.array(graph), nargs))
        ext = calculateGroundedExtension(attMtx)
        if set(groundedExtension) == set(ext):
            acceptedGraphs.append(graph.tolist())

    # Convert acceptedGraphs into a numpy array and return
    acceptedGraphs = np.array(acceptedGraphs)

    return acceptedGraphs, allPermutations


def generateGraphsGivenSetsOfArgs(posArgs, negArgs):

    superAllPermutations = []

    # This is with all the memebers in
    allPermutations = generatePermutations(posArgs, negArgs)
    superAllPermutations.extend(allPermutations)

    totalArgs = posArgs + negArgs
    noArgs = len(totalArgs)
    removeList = list(range(1, len(totalArgs)))

    # start removing members
    for x in removeList:

        removeargs = list(itertools.combinations(totalArgs, x))

        for args in removeargs:
            bigCopy = np.copy(allPermutations)

            for arg in args:
                colsDel = np.arange(0, (noArgs * noArgs) - 1, noArgs) + arg
                rowsDel = np.arange(0, noArgs) + (noArgs * arg)
                allrc = np.concatenate((colsDel, rowsDel), axis=0)
                print(bigCopy[allrc])
                bigCopy[:, allrc] = np.inf

            superAllPermutations.extend(bigCopy)

    return superAllPermutations

# Same as generateGraphsGivenSetsOfArgs except rearranged order so graphs with less args sit at bottom of list
def generateGraphsGivenSetsOfArgsRearranged(posArgs, negArgs):

    superAllPermutations = []

    # This is with all the memebers in
    allPermutations = generatePermutations(posArgs, negArgs)
    superAllPermutations.extend(allPermutations)

    totalArgs = posArgs + negArgs
    noArgs = len(totalArgs)
    removeList = list(range(1, len(totalArgs)))

    # start removing members
    for x in removeList:

        removeargs = list(itertools.combinations(totalArgs, x))

        for args in removeargs:
            bigCopy = np.copy(allPermutations)

            for arg in args:
                colsDel = np.arange(0, (noArgs * noArgs) - 1, noArgs) + arg
                rowsDel = np.arange(0, noArgs) + (noArgs * arg)
                allrc = np.concatenate((colsDel, rowsDel), axis=0)
                print(bigCopy[allrc])
                bigCopy[:, allrc] = np.inf

            superAllPermutations.extend(bigCopy)

    return superAllPermutations

def getGraphPolarity(graph, posArgs, negArgs):

    nargs = int((math.sqrt(len(graph))))
    attMtx = np.vstack(np.array_split(np.array(graph), nargs))
    groundedExt = calculateGroundedExtension(attMtx)

    if not groundedExt: #empty list
        return 'n'
    if set(groundedExt) <= set(posArgs):
        return '+'
    if set(groundedExt) <= set(negArgs):
        return '-'


def getGraphPolarityMixedGraphSize(g, posArgs, negArgs):

    attMtx =  np.vstack(np.array_split(np.array(g), len(posArgs+negArgs)))
    noArgs = len(attMtx)

    v = [np.inf]*noArgs

    infsFound = (attMtx == v)
    infsSum =  np.sum(infsFound,axis=1)
    existingArgs = np.where(infsSum != noArgs)

    smallG = attMtx[:,existingArgs]
    smallG = smallG[existingArgs,:]

    x = existingArgs[0].tolist()
    posFound = list(set(x).intersection(posArgs))
    negFound = list(set(x).intersection(negArgs))

    allFound = posFound + negFound

    smallG = smallG.flatten()
    smallG =  np.vstack(np.array_split(np.array(smallG.flatten()), len(existingArgs[0])))

    groundedExt = calculateGroundedExtension(smallG)
    groundedExt = [allFound[i] for i in groundedExt]

    if not groundedExt: #empty list
        return 'n'
    if set(groundedExt) <= set(posFound):
        return '+'
    if set(groundedExt) <= set(negFound):
        return '-'



def getGroundedExtensionMixedGraphSize(g, posArgs, negArgs):

    attMtx =  np.vstack(np.array_split(np.array(g), len(posArgs+negArgs)))
    noArgs = len(attMtx)
    allArgs = posArgs + negArgs

    v = [np.inf]*noArgs

    infsFound = (attMtx == v)
    infsSum =  np.sum(infsFound,axis=1)
    existingArgs = np.where(infsSum != noArgs)

    smallG = attMtx[:,existingArgs]
    smallG = smallG[existingArgs,:]

    x = existingArgs[0].tolist()
    posFound = list(set(x).intersection(posArgs))
    negFound = list(set(x).intersection(negArgs))

    allFound = posFound + negFound

    smallG = smallG.flatten()
    smallG =  np.vstack(np.array_split(np.array(smallG.flatten()), len(existingArgs[0])))

    groundedExt = calculateGroundedExtension(smallG)
    groundedExt = [allFound[i] for i in groundedExt]

    groundedExtensionFullSize = set(groundedExt).intersection(allArgs)

    return list(groundedExtensionFullSize)


def arrangeEveryGragph(everyGraph):

    # This returns a True False Array where the condition matches
    everyGraphInfCountPerRowBools = everyGraph == np.inf
    # This is a summing of Trues
    everyGraphInfCounPerRowInts = (np.array(everyGraphInfCountPerRowBools)).sum(axis=1)
    # Appending the counts to the end of every graph array so can rearrange
    reArrangedEveryGraph = np.c_[everyGraph, everyGraphInfCounPerRowInts]
    # Sort the everygraph by the inf Count
    reArrangedEveryGraph = reArrangedEveryGraph[reArrangedEveryGraph[:, -1].argsort()]
    # Delete final column (count infs col) as we dont need it anymore
    reArrangedEveryGraph = np.delete(reArrangedEveryGraph, -1, axis=1)

    return reArrangedEveryGraph

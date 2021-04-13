from matplotlib import pyplot as plt
import numpy as np

def readSeg(filepath):
    """ Read segmentation data fromat from Berkeley Segmentation DataSet .

    Args:
        filepath (SEG format): SEG file path.
        
    Returns:
        2d np.array: label map
    """
    f = open(filepath,"r")
    data = []
    metadata = {}

    #get metadata from seg file
    for line in f:
        
        if("data" in line):
            next(f)
            break
        metadata[line.split()[0]] = line.split()[1]
    #get the segmentation data
    
    for line in f:
        data.append([int(i) for i in line.split()])
    
    #parse the data to create the label map
    labelMap = np.zeros((int(metadata['height']), int(metadata['width'])))

    """
    from berkeley document
        <s> <r> <c1> <c2>

    All values start counting at 0.  <s> is the segment number; <r> is the
    row; <c1> and <c2> are column numbers.  The line means that columns
    [<c1>..<c2>] of row <r> belong to segment <s>.

    """
    for values in data:
        labelMap[values[1],values[2]:values[3]+1] = values[0]

    plt.imshow(labelMap)
    plt.show()

    return labelMap


def readSegEdge(filepath):
    """ Read segmentation data fromat from Berkeley Segmentation DataSet .

    Args:
        filepath (SEG format): SEG file path.
        
    Returns:
        2d np.array: edge map
    """
    f = open(filepath,"r")
    data = []
    metadata = {}

    #get metadata from seg file
    for line in f:
        
        if("data" in line):
            next(f)
            break
        metadata[line.split()[0]] = line.split()[1]
    #get the segmentation data
    
    for line in f:
        data.append([int(i) for i in line.split()])
    
    #parse the data to create the label map
    edgeMap = np.zeros((int(metadata['height']), int(metadata['width'])))
    labelMap = np.zeros((int(metadata['height']), int(metadata['width'])))
    labelMap2 = np.zeros((int(metadata['height'])+1, int(metadata['width'])))
    labelMap3 = np.zeros((int(metadata['height']), int(metadata['width'])+1))
    """
    from berkeley document
        <s> <r> <c1> <c2>

    All values start counting at 0.  <s> is the segment number; <r> is the
    row; <c1> and <c2> are column numbers.  The line means that columns
    [<c1>..<c2>] of row <r> belong to segment <s>.

    """
    for values in data:
        labelMap[values[1],values[2]:values[3]+1] = values[0]
        labelMap2[values[1]+1,values[2]:values[3]+1] = values[0]
        labelMap3[values[1],values[2]+1:values[3]+1] = values[0]

    #fast edge label detection
    mask1 = labelMap != labelMap2[:-1,:]
    mask2 = labelMap != labelMap3[:,:-1]
    edgeMap = (labelMap * mask1) + (labelMap * mask2)
    
    #normalize data
    mask = edgeMap > 0
    edgeMap = mask * 1

    # plt.imshow(edgeMap)
    # plt.show()
    # print(filepath)
    return {'label':edgeMap, "id":int(filepath.split('.')[0].split('/')[-1])}

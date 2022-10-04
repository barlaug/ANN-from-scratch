# Helper functions for matrix/vector operations

def transpose(m):
    """Takes the transpose of a NxM matrix
    params:
        m: NxM double list, real valued
    returns:
        MxN double list, transposed of m 
    """
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

def printlist(l):
    """Prints list as an array, or prints the input if it is a real number"""
    if type(l) == list:
        if type(l[0]) == list: # l is a column vector
            for row in l:
                print(row)
        else: # l is a row vector
            print(l)
    elif (type(l) == int) or (type(l) == float):
        print(l)
    else:
        print("Error: Input is not list or a number")

def dot(v1, v2):
    """Takes the dot product (inner product) of the inputs
    params:
        v1, v2: vectors, real valued
    returns:
        The dot product of v1,v2
    """
    if len(v1) != len(v2):
        print("Error: Vectors given as input are not of equal length")
        return 
    else:
        products = [i*j for (i,j) in zip(v1,v2)]
        return sum(products)

def flatten_matrix(m):
    """Flattens out an NxM matrix to a vector of length (NxM)"""
    return [el for row in m for el in row]


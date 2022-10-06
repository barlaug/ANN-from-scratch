# TODO: Remove functions that are not used. Improve readability. Make if statement in diff() more general.

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

def make_rep_vector(length, value):
    """Makes a length long vector with only value vals in it"""
    return [value for i in range(length)]

def scalar_mult(m, s):
    """Returns the scalar product of m and s
    params:
        m: scalar, vector or matrix
        s: scalar
    returns:
        m*s, scalar, vector or matrix - depends on the shape of m
    """
    if type(m) == list:
        if type(m[0]) == list: # m is two dimentional (a matrix)
            res = []
            for row in m:
                res.append([s*el for el in row])
        else: # m is one dimentional (a vector)
            res = [s*el for el in m]      
    else: # m is a scalar
        res = m*s
    return res

def diff(m1, m2):
    """Takes the element-wise difference between m1 and m2 (i.e. m1 - m2)
    params:
        m1, m2: vectors or matrices, same size
    returns:
        m1 - m2
    """
    if (type(m1) == list) and (type(m2) == list):
        if len(m1) != len(m2):
            print("m1 and m2 must have same number of rows")
            return
        if (type(m1[0]) == list) and (type(m2[0]) == list): # both matrices, confimed same number of rows
            if len(m1[0]) != len(m2[0]): # assume that all rows are of equal length, MAYBE MAKE MORE ROBUST LATER
                print("m1 and m2 must have same number of columns")
                return
            # Size ok. Perform calc
            diff_mat = []
            for i in range(len(m1)):
                diff_row = []
                for j in range(len(m1[0])):
                    diff_row.append(m1[i][j] - m2[i][j])
                diff_mat.append(diff_row)
            return diff_mat
        else:
            diff_vec = [(m1[i] - m2[i]) for i in range(len(m1))]
            return diff_vec
    else:
        print("m1, m2 must be lists or list of lists")
        return



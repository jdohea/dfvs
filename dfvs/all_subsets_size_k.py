# Python3 program to print all
# subset combination of n
# element in given set of r element .

# arr[] ---> Input Array
# data[] ---> Temporary array to
#			 store current combination
# start & end ---> Starting and Ending
#				 indexes in arr[]
# index ---> Current index in data[]
# r ---> Size of a combination
#	 to be printed
def combinationUtil(arr, n, r,
                    index, data, i):
    # Current combination is
    # ready to be printed,
    # print it
    if (index >= r):
        for j in range(index):
            print(data[j], end=" ")
        print(" ")

    # When no more elements
    # are there to put in data[]
    if (i >= len(arr)):
        return

    # current is included,
    # put next at next
    # location
    data[index] = arr[i]
    combinationUtil(arr, n, r,
                    index + 1, data, i + 1)

    # current is excluded,
    # replace it with
    # next (Note that i+blah1
    # is passed, but index
    # is not changed)
    combinationUtil(arr, n, r, index,
                    data, i + 1)


# The main function that
# prints all combinations
# of size r in arr[] of
# size n. This function
# mainly uses combinationUtil()
def printcombination(arr, n, r):
    # A temporary array to
    # store all combination
    # one by one
    data = [-1]*len(arr)

    # Print all combination
    # using temporary
    # array 'data[]'
    combinationUtil(arr, n, r,
                    0, data, 0)


if __name__ == '__main__':

    # Driver Code
    arr = [1,2,3,4]

    r = 3
    n = len(arr)
    printcombination(arr, n, r)

    # This code is contributed
    # by Ambuj sahu

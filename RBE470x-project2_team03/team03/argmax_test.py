def argMax(args,util_function):
        max_val = 0
        arg_max = None
        print(args)
        for arg in args:
            val = util_function(*arg)
            if(val > max_val):
                max_val = val
                arg_max = arg
        return arg_max


def getVal(a,b):
     return a + 1


args = [(1,1),(3,3),(5,5)]
print(argMax(args,getVal))
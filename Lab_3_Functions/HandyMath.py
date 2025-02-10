def midpoint(begin_value, end_value):
    mdpt = (begin_value+end_value)/2
    return mdpt

def squareroot(number):
    return number**.5

def exponent(base, exponent):
    exp = base**exponent
    return exp

def max(num1, num2):
    return num1 if num1 >= num2 else num2

def min(num1, num2):
    return num1 if num1 <= num2 else num2

#takes two numbers x,y and a function name as argumennts 
#then returns a string "The function <function name>
#x,y = <function applied to x,y>"

def arguments(x, y, func):
    return  f"The function {func.__name__}({x}, {y}) = {func(x, y)}"

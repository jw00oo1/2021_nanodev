import test1

class A:
    def __init__(self, **kargs):
        print(kargs.get('name'))

tmpargs = test1.args
print(tmpargs)
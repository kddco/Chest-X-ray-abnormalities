def foo():
    print("start...")
    while True:
        throw = yield 10
        print("throw:",throw)
g = foo()
print(next(g))
print("*"*20)
print(next(g))
print(next(g))
"""This experiment is for an easily satisfiable formula of 3 variables"""

from SAT_experiment import run

# (x v y v z) ^ (x' v y v z) ^ (x v y' v z)
formula = [(    (0, True), (1, True), (2, True) ),
                ((0, False), (1, True), (2, True)),
                ((0, True), (1, False), (2, True))]

if __name__ == "__main__":
    satisfiable = run(formula)
    if satisfiable:
        print("FORMULA IS SATISFIABLE")
    else:
        print("FORMULA IS NOT SATISFIABLE")
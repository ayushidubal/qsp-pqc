op = open('outputs_2.txt')

lines = op.readlines()

costs_2 = []
times_2 = []
for l in lines:
    A = l.strip().split()
    costs_2.append(float(A[8]))
    times_2.append(float(A[11]))

print(costs_2)
print(times_2)

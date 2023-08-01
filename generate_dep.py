import numpy as np
import torch

direct_counter = """0,3185,331,0,18
0,0,122,813,0
0,270,0,2,5
0,0,132,0,377
0,17,0,0,0"""

counter = """0,0,0,3044,4390,2584,2068,280,37
0,0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0,0
0,0,0,0,2876,2582,1987,280,37
0,0,0,23,0,2584,1987,280,37
0,0,0,0,0,0,1987,279,36
0,0,0,50,72,49,0,193,27
0,0,0,0,0,0,1,0,37
0,0,0,0,0,0,1,0,0
"""

completion = "  0    0    0 3044 4390 2584 2068  280   37"

def read_counter(s):
    counter = []
    for _s in s.split():
        counter.append(list(map(int, _s.split(','))))
    return np.array(counter)

def read_completion(s):
    return np.array(list(map(int, s.split())))

# counter = read_counter(counter)
# direct_counter = read_counter(direct_counter)
# completion = read_completion(completion)

states = torch.load("/home/alice/orbit/logs/rl_games/heat/fffff_Jun02_05-36-41/nn/objective_selector.pth")
counter = states["counter"]
direct_counter = states["direct_counter"]
completion = states["completion_counter"]
known_objectives = ['init'] + states["known_objectives"]
# print('init')
for achv in known_objectives:
    print(achv)

print(counter)
print(completion)

n = counter.shape[0]
graph = np.zeros((n, n), dtype=int)

for i in range(n):
    for j in range(n):
        graph[i, j] = counter[i, j] > completion[j] * 0.99 and counter[j, i] == 0

def find_earliest(graph, u):
    nl = [[u], []]
    all_nodes = set([u])
    d = 0
    cnt = 0
    while len(nl[d]) > 0:
        nl[1 - d] = []
        for v in nl[d]:
            for x in range(n):
                if graph[x][v] and not x in all_nodes:
                    nl[1 - d].append(x)
                    all_nodes.add(x)
                    cnt += 1
        d = 1 - d
    return cnt

key_edges = []
for i in range(n):
    for j in range(n):
        if graph[i][j] == 1:
            earliest = find_earliest(graph, j)
            graph[i][j] = 0
            connected = find_earliest(graph, j) < earliest
            graph[i][j] = 1
            if connected:
                key_edges.append((i, j))
            # if connected:
            #     print(i, '->', j)
        else:
            connected = False
        print(f"{1 if connected else 0}", end='\n' if j == n - 1 else ',')
print(key_edges)

# print(direct_counter / completion[:, None])

while True:
    # i, j = map(int, input().split())
    # print(counter[i, j] / completion[j], counter[j, i] / completion[j])
    target = int(input())

    good = set([target])
    for i in range(n):
        if counter[i, target] >= completion[target] * 0.8:
            good.add(i)

    required = good
    
    for i in required:
        print(known_objectives[i])

    cur = 0
    visited = set()
    while cur != target:
        nxts = []
        for i in range(1, n):
            if i in required and i not in visited and all([u in visited for u in range(1, n) if graph[u, i]]):
                nxts.append((direct_counter[cur, i], i))

        nxts = sorted(nxts, reverse=True)
        nxt = nxts[0][1]
        print("next:", nxts[0][0], known_objectives[nxt])
        visited.add(nxt)
        cur = nxt
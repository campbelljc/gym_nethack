
VERBOSE = True
LOGGING = False

distances = {}
def distance_pt(A, B): #, metric='manhattan'):
    if (A, B) not in distances:
        val = abs(B[0]-A[0]) + abs(B[1]-A[1])
        distances[(A, B)] = val
        distances[(B, A)] = val
        return val
    return distances[(A, B)]
    #if metric is 'manhattan':
    #elif metric is 'euclidean':
    #    return math.hypot(B[0]-A[0], B[1]-A[1])

def dfs(start, passable_func, neighbor_func, min_neighbors=2, diag=False):
    # src: http://codereview.stackexchange.com/questions/78577/depth-first-search-in-python
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            valid_neighbors = [n for n in neighbor_func(*vertex, diag=diag) if passable_func(*n)]
            if len(valid_neighbors) < min_neighbors: # if this cell only has 1 non-diagonal neighbor...skip it
                continue
            stack.extend([n for n in valid_neighbors if n not in visited])
    return visited

def get_maximal_rectangle(map_width, map_height, rect_points):
    # src: http://stackoverflow.com/questions/7245/puzzle-find-largest-rectangle-maximal-rectangle-problem
    def updateCache(cache, matrixRow, MaxX):
        for m in range(MaxX):
            if not matrixRow[m]:
                cache[m] = 0
            else:
                cache[m] += 1
        return cache
    
    matrix = [[True if (i, j) in rect_points else False for j in range(map_height)] for i in range(map_width)]
    
    best_ll = (0, 0)
    best_ur = (-1, -1)
    best_area = 0

    MaxX = len(matrix[0])
    MaxY = len(matrix)

    stack = []
    cache = [0 for k in range(MaxX+1)]

    for n in range(MaxY):
        openWidth = 0
        cache = updateCache(cache, matrix[n], MaxX)
        for m in range(MaxX+1):
            if cache[m] > openWidth:
                stack.append((m, openWidth))
                openWidth = cache[m]
            elif cache[m] < openWidth:
                area = 0
                p = (-1, -1)
                while True:
                    p = stack.pop()
                    area = openWidth * (m - p[0])
                    if area > best_area:
                        best_area = area
                        best_ll = (p[0], n)
                        best_ur = (m - 1, n - openWidth + 1)
                    openWidth = p[1]
                    if cache[m] >= openWidth:
                        break
                openWidth = cache[m]
                if openWidth != 0:
                    stack.append(p)
    
    positions = set()
    for x in range(best_ll[0], best_ur[0]+1):
        for y in range(best_ur[1], best_ll[1]+1):
            positions.add((y, x))
    #print(positions)
    #input("?")
    return best_ll, best_ur, best_area, positions

def get_maximal_square(map_width, map_height, rect_points):
    # src: http://tech-queries.blogspot.ca/2011/09/maximum-size-square-sub-matrix-with-all.html
    
    matrix = [[1 if (i, j) in rect_points else 0 for j in range(map_height)] for i in range(map_width)]
    
    i, j = -1, -1
    s_mat = [[0 for j in range(map_height)] for i in range(map_width)]
    
    for i in range(map_width):
        s_mat[i][0] = matrix[i][0]
    for j in range(map_height):
        s_mat[0][j] = matrix[0][j]
            
    for i in range(1, map_width):
        for j in range(1, map_height):
            if matrix[i][j] == 1:
                s_mat[i][j] = min(s_mat[i][j-1], s_mat[i-1][j], s_mat[i-1][j-1]) + 1
            else:
                s_mat[i][j] = 0
 
    size = s_mat[0][0]
    max_i, max_j = 0, 0
    for i in range(map_width):
        for j in range(map_height):
            if size < s_mat[i][j]:
                size = s_mat[i][j]
                max_i = i
                max_j = j
    
    #positions = set()
    #for x in range(max_i-size+1, max_i+1):
    #    for y in range(max_j-size+1, max_j+1):
    #        positions.add((x, y))

    return (max_i, max_j), size #, positions

def is_straight_line_adjacent(initial, path, delta=2):
    max_dx, max_dy = 0, 0
    for pos in path:
        dx, dy = abs(pos[0] - initial[0]), abs(pos[1] - initial[1])
        if dx > max_dx:
            max_dx = dx
        if dy > max_dy:
            max_dy = dy
    straight = max_dx < delta or max_dy < delta
    return straight

def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

global log_str
log_str = ""
def save_log(*a, **k):
    global log_str
    log_str += str(a) + "\n"
def print_log():
    print(log_str)

# src : http://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
if VERBOSE:
    verboseprint = print
elif LOGGING:
    verboseprint = save_log
else:
    verboseprint = lambda *a, **k: None
from ast import Tuple
import cv2
import numpy as np
from functools import reduce

ALPHA = 10
BETA = 1
RHO = 0.05
Q = 1

# baseado em https://stackoverflow.com/a/67145008
def get_end_pnts(pnts, img):
    extremes = []
    for p in pnts:
        x = p[0]
        y = p[1]
        n = 0
        if (y > 0):
            n += img[y - 1,x]
            if (x > 0):
                n += img[y - 1,x - 1]
            if (x < img.shape[1] - 1):
                n += img[y - 1,x + 1]
        if (x > 0):
            n += img[y,x - 1]
            if (y < img.shape[0] - 1):
                n += img[y + 1,x - 1]
        if (x < img.shape[1] - 1):
            n += img[y,x + 1]
        if (y < img.shape[0] - 1):
            n += img[y + 1,x]
            if (x < img.shape[1] - 1):
                n += img[y + 1,x + 1]
        n /= 255
        if n == 1:
            extremes.append(p)
    return extremes

# can iterate over and add elements
class LimitedQueue:
    def __init__(self, list = [], limit = 10):
        self.limit = limit
        self.queue = list

    def add(self, item):
        if len(self.queue) == self.limit:
            self.queue.pop(0)
        self.queue.append(item)

    def __contains__(self, item):
        return item in self.queue

    def __iter__(self):
        return iter(self.queue)

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return str(self.queue)

    def __repr__(self):
        return repr(self.queue)

def detect_endpoints2(img):
    nz = cv2.findNonZero(255 - img)
    pts = np.squeeze(nz)
    ext = get_end_pnts(pts, img)
    return ext

def detect_endpoints3(img):
    # Set the end-points kernel:
    h = np.array([[1, 1, 1],
                  [1, 10, 1],
                  [1, 1, 1]])

    # Convolve the image with the kernel:
    imgFiltered = cv2.filter2D(img, -1, h)

    # Extract only the end-points pixels, those with
    # an intensity value of 110:
    endPointsMask = np.where(imgFiltered == 110, 255, 0)
    # The above operation converted the image to 32-bit float,
    # convert back to 8-bit uint
    endPointsMask = endPointsMask.astype(np.uint8)

    return [(x,y) for (x,y) in np.argwhere(imgFiltered == 110)]



# baseado em https://stackoverflow.com/a/72353635
def detect_endpoints(img):
    img2 = 255 - img

    # kernels to find endpoints in all 4 directions
    k1 = np.array(([0, 0, 0], [-1, 1, -1], [-1, -1, -1]), dtype="int")
    k2 = np.array(([0, -1, -1], [0, 1, -1], [0, -1, -1]), dtype="int")
    k3 = np.array(([-1, -1, 0],  [-1, 1, 0], [-1, -1, 0]), dtype="int")
    k4 = np.array(([-1, -1, -1], [-1, 1, -1], [0, 0, 0]), dtype="int")

#    k5 = np.array(([0, 0, 0], [0, 1, -1], [0, -1, -1]), dtype="int")
#    k6 = np.array(([0, -1, -1], [0, 1, -1], [0, 0, 0]), dtype="int")
#    k7 = np.array(([-1, -1, 0], [-1, 1, 0], [0, 0, 0]), dtype="int")
#    k8 = np.array(([0, 0, 0], [-1, 1, 0], [-1, -1, 0]), dtype="int")
#
#    k9 = np.array(([-1, -1, -1], [0, 1, -1], [-1, -1, -1]), dtype="int")
#    k10 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, 0, -1]), dtype="int")
#    k11 = np.array(([0, -1, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")
#    k12 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, -1, 0]), dtype="int")
#    k13 = np.array(([-1, 0, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")
#    k14 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, 0, -1]), dtype="int")

    convolutions = [k1, k2, k3, k4] # k5, k6, k7, k8, k9, k10, k11, k12, k13, k14]

    # perform hit-miss transform for every kernel
    out = reduce(lambda x, y: x + y,
        [cv2.morphologyEx(img, cv2.MORPH_HITMISS, k) for k in convolutions[1:]],
        cv2.morphologyEx(img, cv2.MORPH_HITMISS, convolutions[0]))

    # find points in white (255) and draw them on original image
    pts = np.argwhere(out == 255)

    return [(x,y) for (y,x) in pts if img[x,y] == 0]

def draw_endpoints(img, pts):
    img2 = img.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    for pt in pts:
        img2 = cv2.circle(img2, (pt[0], pt[1]), 2, (0,255,0), -1)
    return img2

def calc_visibility(img, i, j, max_val):
    h, w = img.shape
    img2 = np.array(img, dtype='int16')
    vals = (np.abs(img2[i-1, j-1] - img2[i+1, j-1]) if (0 < i < w-1 and j > 0) else 0,
            np.abs(img2[i-1, j+1] - img2[i+1, j+1]) if (0 < i < w-1 and j < w-1) else 0,
            np.abs(img2[i, j-1] - img2[i, j+1]) if (0 < j < w-1) else 0,
            np.abs(img2[i-1, j] - img2[i+1, j]) if (0 < i < h-1) else 0)
    return max(vals) / max_val

def calc_visibility_matrix(img):
    max_val = img.max()
    visibility_matrix = np.zeros(img.shape, dtype='double')
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            visibility_matrix[i,j] = calc_visibility(img, i, j, max_val)
    return visibility_matrix

# distance between a point and the closest endpoint
def dist(p0, endpoints):
    def _dist(p0, p1):
        return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    return min([_dist(p0, p1) for p1 in endpoints])

def calc_transition_for_ant(img, ant, pheromone_matrix, tabu_list, endpoints):
    r, s = ant
    range_x = range(max(0, r-1), min(img.shape[0], r+2))
    range_y = range(max(0, s-1), min(img.shape[1], s+2))
    points = [(u,v) for u in range_x for v in range_y if (u,v) not in tabu_list]
    denom = sum([pheromone_matrix[u,v]**ALPHA * dist((u,v), endpoints)**BETA for (u,v) in points])
    if denom == 0 or np.isnan(denom):
        return None
    probabilities = [(pheromone_matrix[i,j]**ALPHA * dist((i,j), endpoints)**BETA) / denom for (i,j) in points]
    chosen_idx = np.random.choice(len(points), p=probabilities)
    #print(points[chosen_idx])
    return points[chosen_idx]


def fitness(trail, visibility_matrix):
    mean = np.mean([visibility_matrix[p] for p in trail])
    dev = np.std([visibility_matrix[p] for p in trail]) or 1
    return mean / (dev * len(trail))

def pheromone_update(ants, tabu_lists, pheromone_matrix, visibility_matrix):
    pheromone_matrix *= (1 - RHO)
    for (ant, trail) in zip(ants, tabu_lists):
        i, j = ant
        pheromone_matrix[i,j] += fitness(trail, visibility_matrix)

def ant_colony_optimization(ants, visibility_matrix, pheromone_matrix, tabu_lists, endpoints, visited_matrix, limit=300):
    stopped_ants = []
    for iteration in range(limit):
        print(iteration)
        if (stopped_ants):
            ants_to_remove = sorted(stopped_ants, reverse=True)
            for idx in ants_to_remove:
                ants.pop(idx)
                tabu_lists.pop(idx)
            stopped_ants = []
        if len(ants) == 0:
            break
        for k in range(len(ants)):
            ant = ants[k]
            tabu_list = tabu_lists[k]
            new_point = calc_transition_for_ant(visibility_matrix, ant, pheromone_matrix, tabu_list, endpoints)
            if (new_point == None):
                stopped_ants.append(k)
                break
            ants[k] = new_point
            tabu_list.add(new_point)
            visited_matrix[new_point] = True
            if new_point in endpoints:
                stopped_ants.append(k)
                print(f"Endpoint found by ant {k}")
        pheromone_update(ants, tabu_lists, pheromone_matrix, visibility_matrix)
    return pheromone_matrix

def main():
    original_img = cv2.imread('peppers.bmp', flags=cv2.IMREAD_GRAYSCALE)
    sobel_img = cv2.imread('peppers-sobel-thin.bmp', flags=cv2.IMREAD_GRAYSCALE)
    visibility_matrix = calc_visibility_matrix(original_img)
    pheromone_matrix = np.array(visibility_matrix, dtype='double')
    visited_matrix = np.zeros(visibility_matrix.shape, dtype='bool')
    endpoints = detect_endpoints(255 - sobel_img)
    for endpoint in endpoints:
        visited_matrix[endpoint] = True
    print(f"{len(endpoints)} endpoints found")
    cv2.imwrite('res/endpoints.bmp', draw_endpoints(sobel_img, endpoints))
    cv2.imwrite('res/visibility_matrix.bmp', visibility_matrix*255)
    ants = endpoints.copy()
    tabu_list = [LimitedQueue([point]) for point in ants]
    ant_colony_optimization(ants, visibility_matrix, pheromone_matrix, tabu_list, endpoints, visited_matrix, 10)
    visited_img = np.where(visited_matrix == True, 0, 255)
    #cv2.imshow('pheromone matri', pheromone_matrix); cv2.waitKey(); cv2.destroyAllWindows()
    result = np.minimum(visited_img, sobel_img)
    cv2.imwrite('res/visited.bmp', visited_img)
    cv2.imwrite('res/pheromone_matrix.bmp', (pheromone_matrix / pheromone_matrix.max())*255)
    cv2.imwrite('res/pheromone_matrix_nonzero.bmp', np.array(np.where(pheromone_matrix > 0, 0, 255), 'uint8'))
    cv2.imwrite('res/result.bmp', result)
    return pheromone_matrix

if __name__ == '__main__':
    main()

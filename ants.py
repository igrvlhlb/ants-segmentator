from ast import Tuple
import cv2
import numpy as np
from functools import reduce
from numba import jit, float64, int64, typed, types
from numba.types import UniTuple

ALPHA = 10
BETA = 1
RHO = 0.05
Q = 1

INT_PAIR = UniTuple(int64, 2)


class Segmentator:
    def __init__(self, original_img, sobel_img):
        self.original_img = original_img
        self.sobel_img = sobel_img
        self.visibility_matrix = self.calc_visibility_matrix()
        self.pheromone_matrix = np.array(self.visibility_matrix, dtype='double')
        self.endpoints = self.detect_endpoints()
        self.ants = self.endpoints.copy()
        self.tabu_lists = [LimitedQueue([point]) for point in self.ants]


    def detect_endpoints(self):
        img = 255 - self.sobel_img

        k1 = np.array(([0, 0, 0], [-1, 1, -1], [-1, -1, -1]), dtype="int")
        k2 = np.array(([0, -1, -1], [0, 1, -1], [0, -1, -1]), dtype="int")
        k3 = np.array(([-1, -1, 0],  [-1, 1, 0], [-1, -1, 0]), dtype="int")
        k4 = np.array(([-1, -1, -1], [-1, 1, -1], [0, 0, 0]), dtype="int")

        convolutions = [k1, k2, k3, k4]

        # perform hit-miss transform for every kernel
        out = reduce(lambda x, y: x + y,
            [cv2.morphologyEx(img, cv2.MORPH_HITMISS, k) for k in convolutions[1:]],
            cv2.morphologyEx(img, cv2.MORPH_HITMISS, convolutions[0]))

        # find points in white (255) and draw them on original image
        pts = np.argwhere(out == 255)

        return [(x,y) for (y,x) in pts if img[x,y] == 0]

    def ant_colony_optimization(self, visited_matrix, limit=10):
        stopped_ants = []
        for iteration in range(limit):
            print(iteration)
            if (stopped_ants):
                ants_to_remove = sorted(stopped_ants, reverse=True)
                for idx in ants_to_remove:
                    self.ants.pop(idx)
                    self.tabu_lists.pop(idx)
                stopped_ants = []
            if len(self.ants) == 0:
                break
            for k in range(len(self.ants)):
                ant = self.ants[k]
                tabu_list = self.tabu_lists[k]
                new_point = self.calc_transition_for_ant(k)
                if (new_point == None):
                    stopped_ants.append(k)
                    break
                self.ants[k] = new_point
                tabu_list.add(new_point)
                visited_matrix[new_point] = True
                if new_point in self.endpoints:
                    stopped_ants.append(k)
                    print(f"Endpoint found by ant {k}")
            self.pheromone_update()
        return self.pheromone_matrix

    def calc_transition_for_ant(self, k):
        r, s = self.ants[k]
        range_x = range(max(0, r-1), min(self.original_img.shape[0], r+2))
        range_y = range(max(0, s-1), min(self.original_img.shape[1], s+2))
        points = [(u,v) for u in range_x for v in range_y if (u,v) not in self.tabu_lists[k]]
        denom = sum([self.pheromone_matrix[u,v]**ALPHA * self.dist((u,v))**BETA for (u,v) in points])
        if denom == 0 or np.isnan(denom):
            return None
        probabilities = [(self.pheromone_matrix[i,j]**ALPHA * self.dist((i,j))**BETA) / denom for (i,j) in points]
        chosen_idx = np.random.choice(len(points), p=probabilities)
        #print(points[chosen_idx])
        return points[chosen_idx]

    def pheromone_update(self):
        self.pheromone_matrix *= (1 - RHO)
        for k in range(len(self.ants)):
            i, j = self.ants[k]
            self.pheromone_matrix[i,j] += self.fitness(k)

    def fitness(self, k):
        i, j = self.ants[k]
        trail = self.tabu_lists[k]
        mean = np.mean([self.visibility_matrix[p] for p in trail])
        dev = np.std([self.visibility_matrix[p] for p in trail]) or 1
        return mean / (dev * len(trail))

    def calc_visibility(self, i, j, max_val):
        img = self.original_img
        h, w = img.shape
        img2 = np.array(img, dtype='int16')
        vals = (np.abs(img2[i-1, j-1] - img2[i+1, j-1]) if (0 < i < w-1 and j > 0) else 0,
                np.abs(img2[i-1, j+1] - img2[i+1, j+1]) if (0 < i < w-1 and j < w-1) else 0,
                np.abs(img2[i, j-1] - img2[i, j+1]) if (0 < j < w-1) else 0,
                np.abs(img2[i-1, j] - img2[i+1, j]) if (0 < i < h-1) else 0)
        return max(vals) / max_val

    def calc_visibility_matrix(self):
        img = self.original_img
        max_val = img.max()
        visibility_matrix = np.zeros(img.shape, dtype='double')
        h, w = img.shape
        for i in range(h):
            for j in range(w):
                visibility_matrix[i,j] = self.calc_visibility(i, j, max_val)
        return visibility_matrix

    @staticmethod
    @jit(float64(INT_PAIR, INT_PAIR), nopython=True, fastmath=True, cache=True)
    def _dist(p0, p1):
        return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    
    # distance between a point and the closest endpoint
    def dist(self, p0):
        return min([self._dist(p0, p1) for p1 in self.endpoints])

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


def draw_endpoints(img, pts):
    img2 = img.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    for pt in pts:
        img2 = cv2.circle(img2, (pt[0], pt[1]), 2, (0,255,0), -1)
    return img2

def main():
    original_img = cv2.imread('peppers.bmp', flags=cv2.IMREAD_GRAYSCALE)
    sobel_img = cv2.imread('peppers-sobel-thin.bmp', flags=cv2.IMREAD_GRAYSCALE)
    visited_matrix = np.zeros(original_img.shape, dtype='bool')
    segmentator = Segmentator(original_img, sobel_img)
    for endpoint in segmentator.endpoints:
        visited_matrix[endpoint] = True
    print(f"{len(segmentator.endpoints)} endpoints found")
    cv2.imwrite('res/endpoints.bmp', draw_endpoints(sobel_img, segmentator.endpoints))
    segmentator.ant_colony_optimization(visited_matrix)
    visited_img = np.where(visited_matrix == True, 0, 255)

    cv2.imwrite('res/visibility_matrix.bmp', segmentator.visibility_matrix*255)

    cv2.imwrite('res/visited.bmp', visited_img)

    cv2.imwrite('res/pheromone_matrix.bmp', (segmentator.pheromone_matrix / segmentator.pheromone_matrix.max())*255)
    
    cv2.imwrite('res/pheromone_matrix_nonzero.bmp', np.array(np.where(segmentator.pheromone_matrix > 0, 0, 255), 'uint8'))

    result = np.minimum(visited_img, sobel_img)
    cv2.imwrite('res/result.bmp', result)

if __name__ == '__main__':
    main()

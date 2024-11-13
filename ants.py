# /// script
# requires-python = ">=3.9,<=3.12"
# dependencies = [
#     "tqdm",
#     "opencv-python",
#     "numba",
#     "numpy",
# ]
# ///

import tqdm
import pathlib
import argparse

import cv2
import numpy as np
from functools import reduce
from numba import jit, float64, int64
from numba.core.types.containers import UniTuple


ALPHA = 10
BETA = 1
RHO = 0.05
Q = 10
VERBOSE = False

INT_PAIR = UniTuple(int64, 2)


@jit(INT_PAIR(INT_PAIR), nopython=True)
def swap(tup):
    return (tup[1], tup[0])


class Segmentator:
    def __init__(self, original_img, sobel_img, memsize=10):
        self.original_img = original_img
        self.sobel_img = sobel_img
        self.visibility_matrix = self.calc_visibility_matrix()
        self.pheromone_matrix = np.array(self.visibility_matrix, dtype="double")
        self.endpoints = self.detect_endpoints()
        self.ants = self.endpoints.copy()
        self.initial_endpoints = self.ants.copy()
        self.tabu_lists = [LimitedQueue([point], memsize) for point in self.ants]
        self.visited_matrix = np.zeros(original_img.shape, dtype="bool")

    def detect_endpoints(self):
        img = 255 - self.sobel_img

        k1 = np.array(([0, 0, 0], [-1, 1, -1], [-1, -1, -1]), dtype="int")
        k2 = np.array(([0, -1, -1], [0, 1, -1], [0, -1, -1]), dtype="int")
        k3 = np.array(([-1, -1, 0], [-1, 1, 0], [-1, -1, 0]), dtype="int")
        k4 = np.array(([-1, -1, -1], [-1, 1, -1], [0, 0, 0]), dtype="int")

        convolutions = [k1, k2, k3, k4]

        # perform hit-miss transform for every kernel
        out = reduce(
            lambda x, y: x + y,
            [cv2.morphologyEx(img, cv2.MORPH_HITMISS, k) for k in convolutions[1:]],
            cv2.morphologyEx(img, cv2.MORPH_HITMISS, convolutions[0]),
        )

        # find points in white (255) and draw them on original image
        pts = np.argwhere(out == 255)

        return [(x, y) for (y, x) in pts if img[y, x] == 255]

    def ant_colony_optimization(self, alpha=ALPHA, beta=BETA, rho=RHO, q=Q, limit=10):
        n_initial = len(self.ants)
        stopped_ants = []
        pbar = tqdm.tqdm(total=limit)
        for iteration in range(limit):
            pbar.desc = f"({len(self.ants)}/{n_initial} ants)"
            # if VERBOSE:
            # print(f'iteration: {iteration} ({len(self.ants)}/{n_initial} ants)')
            if stopped_ants:
                ants_to_remove = sorted(stopped_ants, reverse=True)
                for idx in ants_to_remove:
                    self.ants.pop(idx)
                    self.tabu_lists.pop(idx)
                    self.initial_endpoints.pop(idx)
                stopped_ants = []
            if len(self.ants) == 0:
                break
            for k in range(len(self.ants)):
                ant = self.ants[k]
                tabu_list = self.tabu_lists[k]
                new_point = self.calc_transition_for_ant(k, alpha, beta)
                if new_point == None:
                    stopped_ants.append(k)
                    break
                self.ants[k] = new_point
                tabu_list.add(new_point)
                self.visited_matrix[swap(new_point)] = True
                if new_point in self.endpoints:
                    stopped_ants.append(k)
                    print(f"Endpoint found by ant {k}")
            self.pheromone_update(rho, q)
            pbar.update(1)
        pbar.close()
        return self.pheromone_matrix

    def calc_transition_for_ant(self, k, alpha, beta):
        r, s = self.ants[k]
        initial_endpoint = self.initial_endpoints[k]
        range_x = range(max(0, r - 1), min(self.original_img.shape[0], r + 2))
        range_y = range(max(0, s - 1), min(self.original_img.shape[1], s + 2))
        points = [
            (u, v) for u in range_x for v in range_y if (u, v) not in self.tabu_lists[k]
        ]
        denom = sum(
            [
                self.pheromone_matrix[v, u] ** alpha
                * self.dist((u, v), initial_endpoint) ** beta
                for (u, v) in points
            ]
        )
        if denom == 0 or np.isnan(denom):
            return None
        probabilities = [
            (
                self.pheromone_matrix[j, i] ** alpha
                * self.dist((i, j), initial_endpoint) ** beta
            )
            / denom
            for (i, j) in points
        ]
        chosen_idx = np.random.choice(len(points), p=probabilities)
        return points[chosen_idx]

    def pheromone_update(self, rho, q):
        self.pheromone_matrix *= 1 - rho
        for k in range(len(self.ants)):
            i, j = self.ants[k]
            self.pheromone_matrix[j, i] += self.fitness(k) / q

    def fitness(self, k):
        j, i = self.ants[k]
        trail = self.tabu_lists[k]
        mean = np.mean([self.visibility_matrix[p] for p in trail])
        dev = np.std([self.visibility_matrix[p] for p in trail]) or 1
        return mean / (dev * len(trail))

    def calc_visibility(self, i, j, max_val):
        img = self.original_img
        h, w = img.shape
        img2 = np.array(img, dtype="int16")
        vals = (
            np.abs(img2[i - 1, j - 1] - img2[i + 1, j - 1])
            if (0 < i < w - 1 and j > 0)
            else 0,
            np.abs(img2[i - 1, j + 1] - img2[i + 1, j + 1])
            if (0 < i < w - 1 and j < w - 1)
            else 0,
            np.abs(img2[i, j - 1] - img2[i, j + 1]) if (0 < j < w - 1) else 0,
            np.abs(img2[i - 1, j] - img2[i + 1, j]) if (0 < i < h - 1) else 0,
        )
        return max(vals) / max_val

    def calc_visibility_matrix(self):
        img = self.original_img
        max_val = img.max()
        visibility_matrix = np.zeros(img.shape, dtype="double")
        h, w = img.shape
        for i in range(h):
            for j in range(w):
                visibility_matrix[i, j] = self.calc_visibility(i, j, max_val)
        return visibility_matrix

    @staticmethod
    @jit(float64(INT_PAIR, INT_PAIR), nopython=True, fastmath=True, cache=True)
    def _dist(p0, p1):
        return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

    # distance between a point and the closest endpoint
    def dist(self, p0, initial_endpoint):
        return min(
            [self._dist(p0, p1) for p1 in self.endpoints if p1 != initial_endpoint]
        )


# can iterate over and add elements
class LimitedQueue:
    def __init__(self, list=[], limit=10):
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
    # verify if image not already rgb
    if len(img.shape) == 3:
        img2 = img.copy()
    else:
        img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for pt in pts:
        # img2 = cv2.circle(img2, (pt[0], pt[1]), 2, (0,255,0), -1)
        # draw a small square
        img2 = cv2.rectangle(
            img2, (pt[0] - 1, pt[1] - 1), (pt[0] + 1, pt[1] + 1), (0, 255, 0), -1
        )
    return img2


def draw_comparison(sobel, visited):
    img = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
    img[sobel == 0] = [255, 0, 0]

    diff_img = np.where((visited == 0) & (sobel == 255), 0, 255)
    img[diff_img == 0] = [0, 0, 255]

    # set background to gray
    img[(visited == 255) & (sobel == 255)] = [128, 128, 128]

    return diff_img, img


def main(args):
    iterations = args.iterations
    alpha, beta, rho, q = args.alpha, args.beta, args.rho, args.q

    original_img = cv2.imread(args.grayscale, flags=cv2.IMREAD_GRAYSCALE)
    sobel_img = cv2.imread(args.sobel, flags=cv2.IMREAD_GRAYSCALE)

    segmentator = Segmentator(original_img, sobel_img, args.memory)
    print(f"{len(segmentator.endpoints)} endpoints found")
    segmentator.ant_colony_optimization(alpha, beta, rho, q, iterations)

    write_imgs(segmentator, sobel_img, args.output)


def write_imgs(segmentator, sobel_img, output_dir):
    outdir = pathlib.Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    visited_img = np.where(segmentator.visited_matrix == True, 0, 255)

    cv2.imwrite(
        outdir / "endpoints.bmp", draw_endpoints(sobel_img, segmentator.endpoints)
    )

    cv2.imwrite(outdir / "visibility_matrix.bmp", segmentator.visibility_matrix * 255)

    cv2.imwrite(outdir / "visited.bmp", visited_img)

    cv2.imwrite(
        outdir / "pheromone_matrix.bmp",
        (segmentator.pheromone_matrix / segmentator.pheromone_matrix.max()) * 255,
    )

    diff_img, cmp_img = draw_comparison(sobel_img, visited_img)

    cv2.imwrite(outdir / "diff.bmp", diff_img)
    cv2.imwrite(outdir / "cmp.bmp", cmp_img)
    cv2.imwrite(
        outdir / "cmp-endpoints.bmp", draw_endpoints(cmp_img, segmentator.endpoints)
    )

    result = np.minimum(visited_img, sobel_img)
    cv2.imwrite(outdir / "result.bmp", result)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ant Colony Optimization for image segmentation"
    )
    parser.add_argument("grayscale", type=str, help="Grayscale Image file")
    parser.add_argument("sobel", type=str, help="Sobel Image file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print debug information"
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=10, help="Number of iterations"
    )
    parser.add_argument("-o", "--output", default="outputs", help="Output directory")
    parser.add_argument(
        "-m", "--memory", type=int, default=10, help="Tabu list max size"
    )
    parser.add_argument("-a", "--alpha", type=float, default=10, help="Alpha parameter")
    parser.add_argument("-b", "--beta", type=float, default=1, help="Beta parameter")
    parser.add_argument("-r", "--rho", type=float, default=0.05, help="Rho parameter")
    parser.add_argument("-q", "--q", type=float, default=1, help="Q parameter")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    VERBOSE = args.verbose
    print(args)
    main(args)

# parse command line arguments
# -v verbose
# -i iterations
# -o output directory
# -h help
# python ants.py -i 10 -o res
# use argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.path import Path
from scipy.ndimage.filters import gaussian_filter


class RandomVoronoiMap(object):
    def __init__(self, img_size=400, min_num_cells = 10, max_num_cells=50, gaussian_sigma=2.0):
        self.img_size = img_size
        self.max_num_cells = max_num_cells
        self.min_num_cells = min_num_cells
        self.gaussian_sigma = gaussian_sigma

    def voronoi_finite_polygons_2d(self, vor, radius=None):
        """
        from: https://gist.github.com/pv/8036995
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.

        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def create_polygon(self, points, vertices):
        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)

        return grid

    def get_random_map(self):
        while True:
            # random points
            num_cells = np.random.randint(self.min_num_cells,self.max_num_cells)
            points = np.random.randint(0, high=self.img_size, size=(num_cells, 2))

            # create regions
            vor = Voronoi(points)
            try:
                regions, vertices = self.voronoi_finite_polygons_2d(vor)
            except:
                # sometimes this fails for unknown reasons.  try again.
                continue

            map_arr = np.zeros((self.img_size, self.img_size)).flatten()
            x, y = np.meshgrid(np.arange(self.img_size), np.arange(self.img_size))
            x, y = x.flatten(), y.flatten()
            query_points = np.vstack((x, y)).T
            for region in regions:
                polygon = np.array(vertices[region])
                map_arr += Path(polygon).contains_points(query_points) * np.random.rand()

            map_arr = map_arr.reshape((self.img_size, self.img_size))

            # gaussian blur
            # map_arr = gaussian_filter(map_arr, sigma=self.gaussian_sigma)

            # clip image
            map_arr = np.clip(map_arr, 0, 1)

            break

        return map_arr

    def get_random_gaussian_map(self):
        alpha = np.random.rand(self.img_size, self.img_size)
        alpha = (gaussian_filter(alpha, sigma=self.gaussian_sigma) - 0.5) * self.gaussian_sigma + 0.5
        # alpha = np.clip(alpha, 0, 1)
        alpha = 1.0 / (1.0 + np.exp(-(alpha - 0.5) * self.gaussian_sigma * 1.2)) #scale to make distribution more uniform
        scale_factor = 0.04
        alpha = (1 + scale_factor*2) * alpha - scale_factor
        alpha = np.clip(alpha, 0, 1)

        return alpha

if __name__ == '__main__':
    import time
    start_time = time.time()

    randommap = RandomVoronoiMap(num_cells=50, gaussian_sigma=15.0)
    n_samples = 100
    n_bins = 50
    hist_cumulative = np.zeros(n_bins)
    for i in range(n_samples):
        # map_arr = randommap.get_random_map()
        map_arr = randommap.get_random_gaussian_map()
        hist, bins = np.histogram(map_arr, bins=n_bins, range=(0, 1))
        hist_cumulative += hist

    print("--- avg time: %s seconds ---" % ((time.time() - start_time) / n_samples))

    plt.plot(bins[:-1], hist_cumulative / np.sum(hist_cumulative))

    plt.figure()
    plt.imshow(map_arr)

    plt.show()

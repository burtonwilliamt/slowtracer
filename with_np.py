from dataclasses import dataclass, field
import time

import numpy as np
from PIL import Image


# Utility functions
def rowwise_dot(a, b) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def unit_vectors(directions) -> np.ndarray:
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


@dataclass
class HitRecords:
    hit_ts: np.ndarray
    points: np.ndarray
    normals: np.ndarray


@dataclass
class Sphere:
    center: np.ndarray
    radius: float

    def hits(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        t_min: np.ndarray,
        t_max: np.ndarray,
    ) -> HitRecords:
        oc = origins - self.center
        a = rowwise_dot(directions, directions)
        half_b = rowwise_dot(oc, directions)
        c = rowwise_dot(oc, oc) - self.radius**2
        discriminant = half_b * half_b - a * c
        good_discriminants = discriminant >= 0
        # Zero out the negative descriminats so we can do sqrt next.
        discriminant *= (good_discriminants)
        sqrtd = np.sqrt(discriminant)

        roota = (-half_b - sqrtd) / a
        use_root_a = good_discriminants & (roota >= t_min) & (roota <= t_max)
        rootb = (-half_b + sqrtd) / a
        use_root_b = good_discriminants & np.bitwise_not(use_root_a) & (
            rootb >= t_min) & (rootb <= t_max)

        # These are the correct roots to use, or -1
        hit_ts = use_root_a * roota + use_root_b * rootb + (
            np.bitwise_not(use_root_a) & np.bitwise_not(use_root_b)) * -1

        points = origins + directions * hit_ts[:, np.newaxis]

        normals = (points - self.center) / self.radius

        return HitRecords(hit_ts, points, normals)


@dataclass
class Scene:
    _spheres: list[Sphere] = field(default_factory=list)

    def trace(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        t_min: np.ndarray,
        t_max: np.ndarray,
    ) -> HitRecords:
        result = None
        for s in self._spheres:
            records = s.hits(origins, directions, t_min, t_max)
            if result is None:
                result = records
                continue
            # The current value hit (records.hit_ts > 0)
            # And either we haven't seen a hit (result.hit_ts == -1) OR
            # this hit is better ()
            better_hits = (records.hit_ts > 0) & (
                (result.hit_ts == -1) | (records.hit_ts < result.hit_ts))

            not_better_hits = np.bitwise_not(better_hits)

            result.hit_ts *= not_better_hits
            result.hit_ts += records.hit_ts * better_hits

            result.points *= not_better_hits[:, np.newaxis]
            result.points += records.points * better_hits[:, np.newaxis]

            result.normals *= not_better_hits[:, np.newaxis]
            result.normals += records.normals * better_hits[:, np.newaxis]

        return result

    def add(self, sphere: Sphere):
        self._spheres.append(sphere)

    def ray_colors(self, origins, directions):
        records = self.trace(
            origins,
            directions,
            t_min=np.zeros(directions.shape[0]),
            t_max=np.full(directions.shape[0], np.inf),
        )
        rays_that_hit_sphere = records.hit_ts > 0
        color_at_t = 0.5 * (records.normals + 1)

        unit_directions = unit_vectors(directions)
        t = 0.5 * (unit_directions[..., 1] + 1)
        background = (1 - t)[:, np.newaxis] * np.array(
            [1, 1, 1]) + t[:, np.newaxis] * np.array([0.5, 0.7, 1.0])

        # NOTE: inlined IF function here
        return rays_that_hit_sphere[:, np.newaxis] * color_at_t + (
            1 - rays_that_hit_sphere)[:, np.newaxis] * background


@dataclass
class Camera:

    aspect_ratio: float
    image_width: int
    scene: Scene
    viewport_height: float = 2.0
    focal_length: float = 1.0
    num_samples: int = 100
    max_bounces: int = 50
    rng: np.random.Generator = np.random.default_rng(seed=42)

    @property
    def image_height(self) -> int:
        return int(self.image_width / self.aspect_ratio)

    @property
    def viewport_width(self) -> float:
        return self.aspect_ratio * self.viewport_height

    def perform_one_pass(self, current_pixels: np.ndarray) -> None:
        origin = np.array([0, 0, 0])
        horizontal = np.array([self.viewport_width, 0, 0])
        vertical = np.array([0, self.viewport_height, 0])
        lower_left_corner = (origin - horizontal / 2 - vertical / 2 -
                             np.array([0, 0, self.focal_length]))

        j, i, _ = np.indices(current_pixels.shape)
        u = (i + self.rng.random(i.shape)) / (self.image_width - 1)
        v = (j + self.rng.random(j.shape)) / (self.image_height - 1)

        directions = (lower_left_corner + u * horizontal + v * vertical -
                      origin)
        directions.shape = (self.image_height * self.image_width, 3)
        origins = np.full_like(directions, 0)

        results = self.scene.ray_colors(origins, directions)

        results.shape = (self.image_height, self.image_width, 3)

        current_pixels += results
        pass

    def render(self):
        pixels = np.zeros((self.image_height, self.image_width, 3))
        for _ in range(self.num_samples):
            self.perform_one_pass(pixels)
        pixels /= self.num_samples
        return pixels

    def to_file(self, file_name: str):
        beginning = time.time()
        pixels = self.render()
        print(f'Rendering took {time.time() - beginning}')
        with open('temp.ppm', 'wt', encoding='utf-8') as ppm:
            ppm.write(f'P3\n{self.image_width} {self.image_height}\n255\n')
            corrected = 255.999 * pixels
            for j in range(self.image_height - 1, -1, -1):
                for i in range(self.image_width):
                    pixel = corrected[j][i]
                    ppm.write(f'{int(pixel[0])} '
                              f'{int(pixel[1])} '
                              f'{int(pixel[2])}\n')
        img = Image.open('temp.ppm')
        img.save(file_name)


def main():
    scene = Scene()
    scene.add(Sphere(center=np.array([0, 0, -1]), radius=0.5))
    scene.add(Sphere(center=np.array([0, -100.5, -1]), radius=100))
    camera = Camera(aspect_ratio=16 / 9, image_width=400, scene=scene)
    camera.to_file('out.png')


if __name__ == '__main__':
    main()
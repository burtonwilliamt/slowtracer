from dataclasses import dataclass, field
import time

import numpy as np
from PIL import Image

INF = 10000.0


@dataclass
class HitRecords:
    hit_ts: np.ndarray
    points: np.ndarray
    normals: np.ndarray


@dataclass
class Scene:
    _spheres: list = field(default_factory=list)

    def trace(self, origins, directions) -> HitRecords:
        pass


def rowwise_dot(a, b) -> np.ndarray:
    return np.sum(a * b, axis=-1)


@dataclass
class Sphere:
    center: np.ndarray
    radius: float

    def hits(self, origins, directions) -> np.ndarray:
        oc = origins - self.center
        a = rowwise_dot(directions, directions)
        half_b = rowwise_dot(oc, directions)
        c = rowwise_dot(oc, oc) - self.radius**2
        discriminant = half_b * half_b - a * c
        neg_disc = discriminant < 0
        return (1 - neg_disc) * (-half_b - np.sqrt(
            (1 - neg_disc) * discriminant)) / a + (neg_disc) * -1.0


def unit_vectors(directions) -> np.ndarray:
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def ray_colors(origins, directions):
    sphere = Sphere(center=np.array([0, 0, -1]), radius=0.5)
    hit_ts = sphere.hits(origins, directions)
    rays_that_hit_sphere = hit_ts > 0

    normals = unit_vectors(origins + directions * hit_ts[:, np.newaxis] -
                           np.array([0, 0, -1]))
    #print(normals.shape)
    color_at_t = 0.5 * (normals + 1)

    unit_directions = unit_vectors(directions)
    t = 0.5 * (unit_directions[..., 1] + 1)
    background = (1 - t)[:, np.newaxis] * np.array(
        [1, 1, 1]) + t[:, np.newaxis] * np.array([0.5, 0.7, 1.0])

    return rays_that_hit_sphere[:, np.newaxis] * color_at_t + (
        1 - rays_that_hit_sphere)[:, np.newaxis] * background


@dataclass
class Camera:

    aspect_ratio: float
    image_width: int
    scene: Scene
    viewport_height: float = 2.0
    focal_length: float = 1.0
    num_samples: int = 1
    max_bounces: int = 50

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
        u = i / (self.image_width - 1)
        v = j / (self.image_height - 1)

        directions = (lower_left_corner + u * horizontal + v * vertical -
                      origin)
        directions.shape = (self.image_height * self.image_width, 3)
        origins = np.full_like(directions, 0)

        results = ray_colors(origins, directions)

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
    camera = Camera(aspect_ratio=16 / 9, image_width=400, scene=scene)
    camera.to_file('out.png')


if __name__ == '__main__':
    main()
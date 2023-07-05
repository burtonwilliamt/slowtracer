from dataclasses import dataclass
import math
from typing import Self

from PIL import Image


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __neg__(self) -> Self:
        return Vec3(-self.x, -self.y, -self.z)

    def __add__(self, other) -> Self:
        if isinstance(other, Vec3):
            return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        return Vec3(self.x + other, self.y + other, self.z + other)

    #def __radd__(self, other) -> Self:
    #return self + other

    def __sub__(self, other) -> Self:
        return self + (-other)

    #def __rsub__(self, other) -> Self:
    #return -self + other

    def __mul__(self, other) -> Self:
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other) -> Self:
        return self * other

    def __truediv__(self, other) -> Self:
        return Vec3(self.x / other, self.y / other, self.z / other)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def __imul__(self, t: float):
        self.x *= t
        self.y *= t
        self.z *= t

    def __idiv__(self, t: float):
        self *= 1 / t

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def length_squared(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2)

    def as_color_str(self) -> str:
        return f'{int(255.999 * self.x)} {int(255.999 * self.y)} {int(255.999 * self.z)}\n'


class Point3(Vec3):
    pass


class Color(Vec3):
    pass


@dataclass
class Ray:
    origin: Point3
    direction: Point3

    def at(self, t: int) -> Point3:
        return self.origin + t * self.direction


def ray_color(r: Ray) -> Color:
    unit_direction = r.direction / r.direction.length()
    t = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0)


@dataclass
class Camera:

    aspect_ratio: float
    image_width: int
    viewport_height: float = 2.0
    focal_length: float = 1.0

    @property
    def image_height(self) -> int:
        return int(self.image_width / self.aspect_ratio)

    @property
    def viewport_width(self) -> float:
        return self.aspect_ratio * self.viewport_height

    def render(self) -> list[list[Color]]:
        origin = Point3(0.0, 0.0, 0.0)
        horizontal = Vec3(self.viewport_width, 0.0, 0.0)
        vertical = Vec3(0.0, self.viewport_height, 0.0)
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - Vec3(
            0.0, 0.0, self.focal_length)

        pixels = [[None for _ in range(self.image_width)] for _ in range(self.image_height)]
        for j in range(self.image_height):
            for i in range(self.image_width):
                u = i / (self.image_width - 1)
                v = j / (self.image_height - 1)
                r = Ray(
                    origin, lower_left_corner + u * horizontal +
                    v * vertical - origin)
                pixels[j][i] = ray_color(r)
        return pixels

    def to_file(self, file_name: str):
        pixels = self.render()
        with open('temp.ppm', 'wt', encoding='utf-8') as ppm:
            ppm.write(f'P3\n{self.image_width} {self.image_height}\n255\n')
            for j in range(self.image_height - 1, -1, -1):
                for i in range(self.image_width):
                    ppm.write(pixels[j][i].as_color_str())
        img = Image.open('temp.ppm')
        img.save(file_name)


def main():
    camera = Camera(aspect_ratio=16 / 9, image_width=400)
    camera.to_file('out.jpg')


if __name__ == '__main__':
    main()
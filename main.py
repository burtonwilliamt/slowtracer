from dataclasses import dataclass, field
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

    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

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


@dataclass
class HitRecord:
    point: Point3
    t: float
    front_face: bool
    normal: Vec3

    def __init__(self, r: Ray, t: float, outward_normal: Vec3):
        self.point = r.at(t)
        self.t = t
        self.front_face = r.direction.dot(outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal


class Hittable:

    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord | None:
        raise NotImplementedError()


@dataclass
class Sphere(Hittable):
    center: Point3
    radius: float
    color: Color

    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord | None:
        # Sphere equation:
        #  (x - C_x)^2 + (y - C_y)^2 + (z - C_z)^2 = r^2
        # Intersection when:
        #  (R(t) - C) dot (R(t) - C) = r^2
        # Simplified:
        # t^2b dot b  + 2tb dot (A - C) + (A - C) dot (A - C) - r^2
        oc = r.origin - self.center
        a = r.direction.length_squared()
        half_b = oc.dot(r.direction)
        c = oc.length_squared() - self.radius * self.radius
        discriminant = half_b * half_b - a * c
        if discriminant < 0:
            return None
        sqrtd = math.sqrt(discriminant)

        # Find the nearest root that lies in the acceptable range
        root = (-half_b - sqrtd) / a
        if root < t_min or t_max < root:
            root = (-half_b + sqrtd) / a
            if root < t_min or t_max < root:
                return None
        return HitRecord(r=r,
                         t=root,
                         outward_normal=(r.at(root) - self.center) /
                         self.radius)


class HittableList(Hittable):

    def __init__(self):
        self._objects = []

    def hit(self, r: Ray, t_min: float, t_max: float) -> HitRecord | None:
        best_hit = None
        closest_so_far = t_max

        for o in self._objects:
            temp_rec = o.hit(r, t_min, closest_so_far)
            if temp_rec is None:
                continue
            best_hit = temp_rec
            closest_so_far = best_hit.t

        return best_hit

    def add(self, hittable: Hittable) -> None:
        self._objects.append(hittable)


@dataclass
class Scene:
    _hittables: HittableList = field(default_factory=HittableList)

    def trace(self, r: Ray) -> Color:
        record = self._hittables.hit(r, t_min=0.0, t_max=10000.0)
        if record is not None:
            return 0.5 * Color(record.normal.x + 1, record.normal.y + 1,
                               record.normal.z + 1)

        # Background gradient
        unit_direction = r.direction / r.direction.length()
        gradient = 0.5 * (unit_direction.y + 1.0)
        return (1.0 - gradient) * Color(1.0, 1.0, 1.0) + gradient * Color(
            0.5, 0.7, 1.0)

    def add(self, hittable: Hittable) -> None:
        self._hittables.add(hittable)


@dataclass
class Camera:

    aspect_ratio: float
    image_width: int
    scene: Scene
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

        pixels = [[None
                   for _ in range(self.image_width)]
                  for _ in range(self.image_height)]
        for j in range(self.image_height):
            for i in range(self.image_width):
                u = i / (self.image_width - 1)
                v = j / (self.image_height - 1)
                r = Ray(
                    origin,
                    lower_left_corner + u * horizontal + v * vertical - origin)
                pixels[j][i] = self.scene.trace(r)
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
    scene = Scene()
    scene.add(Sphere(Point3(0.0, 0.0, -1.0), 0.5, Color(1.0, 0.0, 0.0)))
    scene.add(Sphere(Point3(0.0, -100.5, -1.0), 100, Color(1.0, 0.0, 0.0)))
    camera = Camera(aspect_ratio=16 / 9, image_width=400, scene=scene)
    camera.to_file('out.png')


if __name__ == '__main__':
    main()
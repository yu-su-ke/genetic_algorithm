from math import sin, cos, acos, radians, sqrt, atan2


class DistCalculation:
    def __init__(self, earth_rad=6378.137):
        self.earth_rad = earth_rad

    def latlng_to_xyz(self, lat, lng):
        rlat, rlng = radians(lat), radians(lng)
        coslat = cos(rlat)

        return coslat*cos(rlng), coslat*sin(rlng), sin(rlat)

    def dist_on_sphere(self, pos0, pos1):
        xyz0, xyz1 = self.latlng_to_xyz(*pos0), self.latlng_to_xyz(*pos1)

        return acos(sum(x * y for x, y in zip(xyz0, xyz1)))*self.earth_rad

    def dist_test(self, pos0, pos1):
        latitude1 = pos0[0]
        longitude1 = pos0[1]
        latitude2 = pos1[0]
        longitude2 = pos1[1]

        different_longitude = longitude2 - longitude1
        different_latitude = latitude2 - latitude1

        a = sin(different_latitude / 2) ** 2 + cos(latitude1) * cos(latitude2) * sin(different_longitude / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = self.earth_rad * c

        return distance

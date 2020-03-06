import math

def calcIntersections(centroid1, centroid2, r1, r2):

    dist = math.sqrt((centroid2[1] - centroid1[1]) ** 2 + (centroid2[0] - centroid1[0]) ** 2)
    if dist > r1 + r2:
        return None
    elif dist < abs(r1 - r2):
        return None
    elif dist == 0 and r1 == r2:
        return None
    else:
        a = (r1 ** 2 - r2 ** 2 + dist ** 2) / (2 * dist)  # a is midpoint of intersection
        h = math.sqrt(r1 ** 2 - a ** 2)  # h is line section to upper intersect
        x2 = centroid1[1] + a * (centroid2[1] - centroid1[1]) / dist
        y2 = centroid1[0] + a * (centroid2[0] - centroid1[0]) / dist
        x3 = x2 + h * (centroid2[0] - centroid1[0]) / dist
        y3 = y2 - h * (centroid2[1] - centroid1[1]) / dist
        x4 = x2 - h * (centroid2[0] - centroid1[0]) / dist
        y4 = y2 + h * (centroid2[1] - centroid1[1]) / dist

        return(x3, y3, x4, y4)
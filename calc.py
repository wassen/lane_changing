def calc_timeToClosestPoint(self, car):

    x = car[0]
    y = car[1]
    vx = car[2]
    vy = car[3]

    if vx ==0 and vy == 0:
        return float('inf'), float('inf')

    import math

    timeToClosestPoint = -(x * vx + y * vy) / (vx ** 2 + vy ** 2)
    distanceToClosestPoint = abs(x * vy - y * vx) / math.sqrt(vx ** 2 + vy ** 2)
    
    return timeToClosestPoint, distanceToClosestPoint

l = [0, 1, 0.1, 1]
print(calc_timeToClosestPoint(l))

def get_area(pos, weight, height):
    assert height > 0 and weight > 0
    area = []
    for i in range(height):
        for j in range(weight):
            area.append([pos[0] + j, pos[1] + i])
    return area

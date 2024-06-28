import random
x_limit_min, x_limit_max = 0.6, 0.8
y_limit_min, y_limit_max = -0.05, 0.25
A = []
for i in range(50):
    random_pos_x = random.random() * (x_limit_max - x_limit_min) + x_limit_min
    random_pos_y = random.random() * (y_limit_max - y_limit_min) + y_limit_min
    print('[%4.2f, %4.2f],' % (random_pos_x, random_pos_y), end='')
    if i % 5 == 4:
        print()
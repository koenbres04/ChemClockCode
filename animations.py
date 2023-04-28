from io import StringIO


def temp_print(array, scale=1.):
    builder = StringIO()
    for y in range(array.shape[1]):
        for x in range(array.shape[0]):
            value = array[x, y]
            if value <= scale:
                builder.write("  ")
            elif value <= 2*scale:
                builder.write(". ")
            elif value <= 3*scale:
                builder.write("- ")
            elif value <= 4*scale:
                builder.write("* ")
            else:
                builder.write("# ")
        builder.write("\n")
    print(builder.getvalue())


def grid_animation(values, dt):
    # temporary  lasse doe dit
    for i, grid in enumerate(values):
        print(" ")
        print(f"t = {i*dt:2f}")
        temp_print(grid, scale=0.25)

def print_mat(mat, width=4):
    for row in mat:
        for n in row:
            if n:
                print("{:{width2}.{width}f}".format(n, width=width, width2=width+4), end=' ')
            else:
                print("{:{width}.0f}".format(n, width=(width + 4)), end=' ')

        print()

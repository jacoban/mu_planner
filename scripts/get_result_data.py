def get_data_from_single_file(filepath):
    not_solved = []
    times = []
    nodes = []
    checks_total = []
    checks_real = []
    path_costs = []
    p_coll = []

    f = open(filepath, 'r')
    lines = f.readlines()
    tot = 0
    for line in lines:
        s = line.split()
        if not len(s):
            continue

        tot += 1

        not_solved.append(eval(s[0]))

        if not_solved[-1] == False:
            path_costs.append(float(s[2]))
            p_coll.append(float(s[6]))

        nodes.append(float(s[1]))
        times.append(float(s[3]))
        checks_total.append(float(s[4]))
        checks_real.append(float(s[5]))

    return tot, not_solved, times, nodes, checks_total, checks_real, path_costs, p_coll
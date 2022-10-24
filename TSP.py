import matplotlib.pyplot as plt

def plot_route(route, coords, orders, max_capacity=1000):
    fig, ax = plt.subplots(ncols=1, figsize=(4,4))

    ax.set(xlabel="X", ylabel="Y", xlim=(0,100), ylim=(0,100))
    ax.scatter(coords[:,0], coords[:,1], marker="o", s=15, c=orders, cmap="inferno_r")
    ax.plot(coords[0][0], coords[0][1], ls="", marker="o", ms=10, c="red")
    ax.grid()

    capacity = max_capacity - orders[route[0]]
    ax.annotate("", xytext=coords[0], xy=coords[route[0]], arrowprops=dict(arrowstyle="->"))
    
    for i, f in zip(route, route[1:]):
        if capacity < orders[f]:
            capacity = max_capacity
            ax.annotate("", xytext=coords[i], xy=coords[0], arrowprops=dict(arrowstyle="->"))
            ax.annotate("", xytext=coords[0], xy=coords[f], arrowprops=dict(arrowstyle="->"))
            capacity -= orders[f]
            # print("Ups, go back")
        else:
            ax.annotate("", xytext=coords[i], xy=coords[f], arrowprops=dict(arrowstyle="->"))
            capacity -= orders[f]

    ax.annotate("", xytext=coords[route[-1]], xy=coords[0], arrowprops=dict(arrowstyle="->"))
    
    
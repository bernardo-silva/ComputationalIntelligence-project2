import matplotlib.pyplot as plt
from deap import tools
import random


def PMX(ind1, ind2):
    """Partially Mapped Crossover
    Takes into account that the cities are numbered from 1 to N

    Parameters
    ----------
    ind1, ind2 : Individual
        Parents

    Returns
    -------
    ind1, ind2 : Individual
        Offsprings.
    """
    ind1 -= 1
    ind2 -= 1
    tools.cxPartialyMatched(ind1, ind2)
    ind1 += 1
    ind2 += 1

    return (ind1, ind2)


def inversion(ind):
    """Inversion mutation

    Parameters
    ----------
    ind : Individual

    Returns
    -------
    ind : Individual
    """
    r1 = random.randrange(len(ind)+1)
    r2 = random.randrange(len(ind)+1)

    invpoint1, invpoint2 = min(r1, r2), max(r1, r2)

    ind[invpoint1:invpoint2] = ind[invpoint1:invpoint2][::-1]
    return ind



def plot_route(route, coords, orders, max_capacity=1000):
    """

    Parameters
    ----------
    route : TYPE
        DESCRIPTION.
    coords : TYPE
        DESCRIPTION.
    orders : TYPE
        DESCRIPTION.
    max_capacity : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(ncols=1, figsize=(4, 4))

    ax.set(xlabel="X", ylabel="Y", xlim=(0, 100), ylim=(0, 100))
    ax.scatter(coords[:, 0], coords[:, 1], marker="o",
               s=15, c=orders, cmap="inferno_r")
    ax.plot(coords[0][0], coords[0][1], ls="", marker="o", ms=10, c="red")
    ax.grid()

    capacity = max_capacity - orders[route[0]]
    ax.annotate("", xytext=coords[0], xy=coords[route[0]],
                arrowprops=dict(arrowstyle="->"))

    for i, f in zip(route, route[1:]):
        if capacity < orders[f]:
            capacity = max_capacity
            ax.annotate(
                "", xytext=coords[i], xy=coords[0], arrowprops=dict(arrowstyle="->"))
            ax.annotate(
                "", xytext=coords[0], xy=coords[f], arrowprops=dict(arrowstyle="->"))
            capacity -= orders[f]
            # print("Ups, go back")
        else:
            ax.annotate(
                "", xytext=coords[i], xy=coords[f], arrowprops=dict(arrowstyle="->"))
            capacity -= orders[f]

    ax.annotate("", xytext=coords[route[-1]],
                xy=coords[0], arrowprops=dict(arrowstyle="->"))


def plot_route_with_labels(route, coords, orders, max_capacity=1000, figsize=(4, 4)):
    """

    Parameters
    ----------
    route : TYPE
        DESCRIPTION.
    coords : TYPE
        DESCRIPTION.
    orders : TYPE
        DESCRIPTION.
    max_capacity : TYPE, optional
        DESCRIPTION. The default is 1000.
    figsize : TYPE, optional
        DESCRIPTION. The default is (4, 4).

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    arrowprops = dict(arrowstyle="->", shrinkA=2, shrinkB=7.5)
    textoptions = dict(ha="center", va="center", c="white", fontsize=10)

    ax.set(xlabel="X", ylabel="Y", xlim=(0, 100), ylim=(0, 100))
    ax.grid()
    ax.set_axisbelow(True)
    ax.scatter(coords[:, 0], coords[:, 1], marker="o",
               s=15**2, c=orders, cmap="Blues")
    ax.scatter(coords[0, 0], coords[0, 1], marker="o", s=15**2, c="red")
    # ax.plot(coords[0][0], coords[0][1], ls="", marker="o", ms=10, c="red",ec="red")

    capacity = max_capacity - orders[route[0]]
    ax.annotate("wh", xytext=coords[0], xy=coords[route[0]],
                arrowprops=arrowprops, **textoptions)

    for i, f in zip(route, route[1:]):
        if capacity < orders[f]:
            capacity = max_capacity
            ax.annotate(
                f"{i}", xytext=coords[i], xy=coords[0], arrowprops=arrowprops, **textoptions)
            ax.annotate(
                "wh", xytext=coords[0], xy=coords[f], arrowprops=arrowprops, **textoptions)
            capacity -= orders[f]
            # print("Ups, go back")
        else:
            ax.annotate(
                f"{i}", xytext=coords[i], xy=coords[f], arrowprops=arrowprops, **textoptions)
            capacity -= orders[f]

    ax.annotate(f"{f}", xytext=coords[route[-1]],
                xy=coords[0], arrowprops=arrowprops, **textoptions)

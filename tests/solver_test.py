"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Luan Marc Suquet Camargo      (nUSP: 11809090)
         Francisco Barbosa             (nUSP: 6672901)

Computacao IV (CCM): Chemistry project
"""
from typing import Optional
from numerical.chem.solver import ssa, elmaru
import matplotlib.pyplot as plt
import random as rand
import numpy as np


# Font configuration
font = { "family": "monospace", "size": 15 }
_font = { "family": "monospace", "size": 10 }
sub_font = { "family": "monospace", "size": 10, "weight": "bold" }
title_font = { "family" : "monospace", "size": 20, "weight": "bold" }

# Colors for CLI text
RUN = "\033[1;32m::\033[0m"
QST = "\033[1;35m=>\033[0m"
PLOT = "\033[1;34m->\033[0m"
TITL = "\033[1;35m"
RES = "\033[0m"
ERR = "\033[1;31m"


def dimkin():
    """Dimerisation kinetics: example 1"""
    print(f"\n{TITL}{' Dimerisation kinetics ':-^79}{RES}\n")
    ls = ["$P$", "$P_2$"]
    x0 = [301, 0]
    vol = 1e-15
    r = [
        [-2,  1], # Dimerisation
        [ 2, -1] # Dissociation
    ]

    c1 = 1.66e-3
    def dim(x):
        """Dimerisation propensity"""
        return c1 * x[0] * (x[0] - 1)/2

    c2 = 0.2
    def diss(x):
        """Dissociation propensity"""
        return c2 * x[1]

    print(f"{RUN} Calculating and plotting the evolution over 10s (SSA)\n")
    ssa(x0, r, [dim, diss], vol, title="Dimerisation kinetics: SSA", tspan=10,
        labels=ls, conc=True)

    print(f"{RUN} Calculating and plotting the evolution"
          " over 10s (Euler-Maruyama)\n")
    elmaru(x0, r, [dim, diss], vol, tspan=10, L=500, labels=ls,
           title="Dimerisation kinetics: Euler-Maruyama")


    print(f"{RUN} Running 20 simulation with SSA...")
    for _ in range(20):
        ts, xs, _ = ssa(x0, r, [dim, diss], vol, tspan=10, _plot=False)
        plt.plot(ts, [x[0] for x in xs], color="black")
    print(f"{PLOT} Plotting the overlay of 20 runs for the protein...\n")
    plt.title("Protein evolution: 20 runs", title_font)
    plt.ylim(0, x0[0])
    plt.xlabel("Time", font)
    plt.ylabel("Nr. molecules", font)
    plt.legend(["$P$"], prop=_font)
    plt.show()

    runs = 10000
    print(f"{RUN} Running {runs} stochastic simulations...")
    ps = []
    for _ in range(runs):
        _, xs, _ = ssa(x0, r, [dim, diss], vol, tspan=10, _plot=False)
        ps.append(xs[-1][0])

    print(f"{PLOT} Plotting the density histogram of the protein"
          " at time t = 10...\n")
    plt.hist(ps, bins=int(185/5), density=True,
             edgecolor="black", color="orange")
    plt.title("Density histogram of $P$: SSA", title_font)
    plt.xlabel("$P(10)$", font)
    plt.ylabel("Density", font)
    plt.show()

    runs = 1000
    print(f"{RUN} Running {runs} Euler-Maruyama simulations...")
    _L = 500
    _tspan = 10
    tau = _tspan / _L
    ts = np.array([n * tau for n in range(_L)])
    pem = np.zeros(_L)
    for _ in range(runs):
        _, xs, _ = elmaru(x0, r, [dim, diss], vol,
                          tspan=_tspan, L=_L, _plot=False)
        xs = np.array(xs)
        for t in range(_L):
            pem[t] += xs[t, 0]
    pem *= 1/runs # Take mean

    # Calculate standard deviation
    std_run = 3
    rec = np.zeros((std_run, _L))
    for run in range(std_run):
        _, xs, _ = elmaru(x0, r, [dim, diss], vol,
                          tspan=_tspan, L=_L, _plot=False)
        xs = np.array(xs)
        for t in range(_L):
            rec[run, t] = xs[t, 0]
    std = np.zeros(_L)
    for t in range(_L):
        std[t] = np.std(rec[:, t])

    print(f"{PLOT} Plotting sample mean for the protein evolution...")
    plt.plot(ts, pem, color="orange")
    plt.plot(ts, pem + std, color="magenta")
    plt.plot(ts, pem - std, color="green")
    plt.ylim(0, x0[0])
    plt.ylabel("Nr. molecules", font)
    plt.xlabel("Time", font)
    plt.legend(
        ["Mean of P",
         f"Mean of P + {std_run} samples std. deviation",
         f"Mean of P - {std_run} samples std. deviation"],
        prop=_font
    )
    plt.show()


def michmen():
    """Michael-Menten: example 2"""
    print(f"\n{TITL}{' Michael-Menten ':-^79}{RES}\n")
    ls = ["Substrate", "Enzyme", "Complex", "Product"]
    x0 = [301, 120, 0, 0] # Initial number of molecules
    r = [
        [-1, -1,  1, 0], # Binding
        [ 1,  1, -1, 0], # Dissociation
        [ 0,  1, -1, 1]  # Conversion
    ]
    vol = 1e-15 # Volume

    c1 = 1.66e-3
    def b(x):
        """Binding propensity"""
        return c1 * x[0] * x[1]

    c2 = 1e-4
    def d(x):
        """Dissociation propensity"""
        return c2 * x[2]

    c3 = 0.1
    def c(x):
        """Conversion propensity"""
        return c3 * x[2]

    print(f"{RUN} Calculating SSA simulation and plotting...\n")
    ssa(x0, r, [b, d, c], vol, labels=ls, conc=True,
        title="Michael-Menten: SSA")

    print(f"{RUN} Calculating Euler-Maruyama simulation and plotting...")
    elmaru(x0, r, [b, d, c], vol, labels=ls,
           title="Michael-Menten: Euler-Maruyama")


def argn():
    """Auto-regulatory genetic network"""
    print(f"\n{TITL}{' Auto-regulatory network ':-^79}{RES}\n")
    ls = ["Gene", "$P_2 \\cdot$Gene", "RNA", "$P$", "$P_2$"]
    x0 = [10, 0, 0, 0, 0] # Initial state
    r = [
        [-1,  1,  0,  0, -1], # Repression binding
        [ 1, -1,  0,  0,  1], # Reverse repression binding
        [ 0,  0,  1,  0,  0], # Transcription
        [ 0,  0,  0,  1,  0], # Translation
        [ 0,  0,  0, -2,  1], # Dimerisation
        [ 0,  0,  0,  2, -1], # Dissociation
        [ 0,  0, -1,  0,  0], # RNA degeneration
        [ 0,  0,  0, -1,  0]  # Protein degeneration
    ]

    k1 = 1
    def rb(x):
        """Repression binding:
        Gene + P2 -> P2.Gene
        """
        return k1 * x[0] * x[-1]

    k1r = 10
    def rrb(x):
        """Reverse repression binding:
        P2.Gene -> Gene + P2"""
        return k1r * x[1]

    k2 = 0.01
    def trsc(x):
        """Transcription:
        Gene -> Gene + RNA
        """
        return k2 * x[0]

    k3 = 10
    def trans(x):
        """Translation:
        RNA -> RNA + P
        """
        return k3 * x[2]

    k4 = 1
    def dim(x):
        """Dimerisation:
        P + P -> P2
        """
        return k4 * 0.5 * x[-2] * (x[-2] - 1)

    k4r = 1
    def diss(x):
        """Dissociation:
        P2 -> P + P
        """
        return k4r * x[-1]

    k5 = 0.1
    def rnadeg(x):
        """RNA degeneration:
        RNA -> nothing
        """
        return k5 * x[2]

    k6 = 0.01
    def pdeg(x):
        """Protein degeneration:
        P -> nothing
        """
        return k6 * x[-2]

    def constraint_time(ts: list[float], tfinal: float):
        """Auxiliary function for contraining time arrays:
        Returns the array containing all elments of `ts` less than or equal to
        `tfinal`. Assumes that `ts` is sorted.
        """
        _ts = []
        i = 0
        t = ts[i]
        while t <= tfinal:
            _ts.append(t)
            i += 1
            t = ts[i]
        return _ts

    a = [rb, rrb, trsc, trans, dim, diss, rnadeg, pdeg]

    print(f"{RUN} Calculating evolution of the network through"
          " a time span of 5000s (SSA)...")
    ts, xs, _ = ssa(x0, r, a, tspan=5000, _plot=False)
    xs = np.array(xs)
    plt.figure()

    print(f"{PLOT} Plotting evolution for time in [0, 5000]...")
    plt.subplot(321)
    plt.plot(ts, xs[:, 2], color="orange")
    plt.xlabel("Time", _font)
    plt.ylabel("RNA", _font)

    plt.subplot(323)
    plt.plot(ts, xs[:, 3], color="purple")
    plt.xlabel("Time", _font)
    plt.ylabel("$P$", _font)

    plt.subplot(325)
    plt.plot(ts, xs[:, 4], color="blue")
    plt.xlabel("Time", _font)
    plt.ylabel("$P_2$", _font)

    print(f"{PLOT} Plotting evolution for time in [0, 250]...\n")
    _ts = constraint_time(ts, 250)

    plt.subplot(322)
    plt.plot(_ts, xs[:len(_ts), 2], color="orange")
    plt.xlabel("Time", _font)
    plt.ylabel("RNA", _font)

    plt.subplot(324)
    plt.plot(_ts, xs[:len(_ts), 3], color="purple")
    plt.xlabel("Time", _font)
    plt.ylabel("$P$", _font)

    plt.subplot(326)
    plt.plot(_ts, xs[:len(_ts), 4], color="blue")
    plt.xlabel("Time", _font)
    plt.ylabel("$P_2$", _font)

    plt.suptitle("Auto-regulatory genetic network over 5000s (SSA)",
                 fontproperties=title_font)
    plt.show()


    print(f"{RUN} Calculating evolution of the network through"
          " a time span of 10s (SSA)...")
    ts, xs, _ = ssa(x0, r, a, tspan=10, _plot=False)
    xs = np.array(xs)
    print(f"{PLOT} Plotting evolution of P over 10s...\n")
    plt.plot(ts, xs[:, 3], color="purple")
    plt.title("Evolution of $P$ over 10s", title_font)
    plt.ylabel("Nr. molecules", font)
    plt.xlabel("Time", font)
    plt.show()

    def density_P10(runs: int, _c2: Optional[float], index: int):
        str_c2 = f"k2 = {_c2}"
        if _c2 == None:
            str_c2 = "k2 uniformly in [0.005, 0.03)"
        print(f"{RUN} Running {runs} simulations over 10s (SSA)"
              f" for {str_c2} ...")

        unif = False
        if _c2 == None:
            unif = True

        ps = []
        for _ in range(runs):
            if unif:
                _c2 = rand.uniform(0.005, 0.03)
            def _trsc(x):
                """Transcription with altered constant `c2`
                Gene -> Gene + RNA
                """
                return _c2 * x[0]

            _a = [rb, rrb, _trsc, trans, dim, diss, rnadeg, pdeg]

            _, xs, _ = ssa(x0, r, _a, tspan=10, labels=ls, _plot=False)
            ps.append(xs[-1][-2])

        print(f"{PLOT} Plotting density histogram of P over 10s...\n")
        plt.subplot(3, 1, index)
        plt.hist(ps, bins=int(185/5), density=True,
                edgecolor="black", color="orange")
        plt.xlabel("$P(10)$", _font)
        plt.ylabel("Density", _font)
        if not unif:
            plt.title(f"Density for $k_2 = {_c2}$", sub_font)
        else:
            plt.title(f"$k_2$ uniformly choosen in [0.005, 0.03)", sub_font)

    plt.subplots(constrained_layout=True)
    density_P10(1000, 0.01, 1)
    density_P10(1000, 0.02, 2)
    density_P10(1000, _c2=None, index=3)
    plt.suptitle("Density histogram of $P$ over 10s", fontproperties=title_font)
    plt.show()


def lac():
    """Lac-operon model"""
    print(f"\n{TITL}{' Lac-operon model ':-^79}{RES}\n")
    x0 = [1, 0, 50, 1, 100, 0, 0, 20, 0, 0, 0]
    r = [
        # Inhibitor transcription: IDNA -> IDNA + IRNA
        [0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        # Inhibitor translation: IRNA -> IRNA + I
        [0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
        # Lactose inhibitor binding: I + Lactose -> ILactose
        [0,  0, -1,  0,  0,  0,  0, -1,  1,  0,  0],
        # Lactose inhibitor dissociation: ILactose -> I + Lactose
        [0,  0,  1,  0,  0,  0,  0,  1, -1,  0,  0],
        # Inhibitor binding: I + Op -> IOp
        [0,  0, -1, -1,  0,  0,  0,  0,  0,  1,  0],
        # Inhibitor dissociation: IOp -> I + Op
        [0,  0,  1,  1,  0,  0,  0,  0,  0, -1,  0],
        # RNAp binding: Op + RNAp -> RNApOp
        [0,  0,  0, -1, -1,  0,  0,  0,  0,  0,  1],
        # RNAp dissociation: RNApOp -> Op + RNAp
        [0,  0,  0,  1,  1,  0,  0,  0,  0,  0, -1],
        # Transcription: RNApOp -> Op + RNAp + RNA
        [0,  0,  0,  1,  1,  1,  0,  0,  0,  0, -1],
        # Translation: RNA -> RNA + Z
        [0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
        # Conversion: Lactose + Z -> Z
        [0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
        # Inhibitor RNA degradation: IRNA -> nothing
        [0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        # Inhibitor degradation: I -> nothing
        [0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
        # Lactose inhibitor degradation: ILactose -> Lactose
        [0,  0,  0,  0,  0,  0,  0,  1, -1,  0,  0],
        # RNA degradation: RNA -> nothing
        [0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0],
        # Z degradation: Z -> nothing
        [0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],
    ]

    c1 = 0.02
    def in_trsc(x):
        """Inhibitor transcription:
        IDNA -> IDNA + IRNA
        """
        return c1 * x[0]

    c2 = 0.1
    def in_trans(x):
        """Inhibitor translation:
        IRNA -> IRNA + I
        """
        return c2 * x[1]

    c3 = 0.005
    def lac_in_bin(x):
        """Lactose inhibitor binding:
        I + Lactose -> ILactose
        """
        return c3 * x[2] * x[7]

    c4 = 0.1
    def lac_in_diss(x):
        """Lactose inhibitor dissociation:
        ILactose -> I + Lactose
        """
        return c4 * x[8]

    c5 = 1
    def in_bin(x):
        """Inhibitor binding:
        I + Op -> IOp
        """
        return c5 * x[2] * x[3]

    c6 = 0.01
    def in_diss(x):
        """Inhibitor dissociation:
        IOp -> I + Op
        """
        return c6 * x[9]

    c7 = 0.1
    def rnap_bin(x):
        """RNAp binding:
        Op + RNAp -> RNApOp
        """
        return c7 * x[3] * x[4]

    c8 = 0.01
    def rnap_diss(x):
        """RNAp dissociation:
        RNApOp -> Op + RNAp
        """
        return c8 * x[10]

    c9 = 0.03
    def trans(x):
        """Transcription:
        RNApOp -> Op + RNAp + RNA
        """
        return c9 * x[10]

    c10 = 0.1
    def transl(x):
        """Translation:
        RNA -> RNA + Z
        """
        return c10 * x[5]

    c11 = 1e-5
    def conv(x):
        """Conversion:
        Lactose + Z -> Z
        """
        return c11 * x[6] * x[7]

    c12 = 0.01
    def in_rna_deg(x):
        """Inhibitor RNA degradation:
        IRNA -> nothing
        """
        return c12 * x[1]

    c13 = 0.002
    def in_deg(x):
        """Inhibitor degradation:
        I -> nothing
        """
        return c13 * x[2]

    def lac_in_deg(x):
        """Lactose inhibitor degradation:
        ILactose -> Lactose
        """
        return c13 * x[8]

    c14 = 0.01
    def rna_deg(x):
        """RNA degradation:
        RNA -> nothing
        """
        return c14 * x[5]

    c15 = 0.001
    def z_deg(x):
        """Z degradation:
        Z -> nothing
        """
        return c15 * x[6]

    a = [
        in_trsc,    in_trans,  lac_in_bin, lac_in_diss,
        in_bin,     in_diss,   rnap_bin,   rnap_diss,
        trans,      transl,    conv,       in_rna_deg,
        in_deg,     lac_in_deg, rna_deg,   z_deg,
    ]

    t_event = 20000
    def intervention(t: float, x: list[int]) -> bool:
        """Event intervention at t = 20000:
        Adds 10000 lactose molecules to the current state
        """
        if t >= t_event:
            x[7] += 10000
            return False
        return True

    tspan = 50000
    print(f"{RUN} Running (SSA) simulation of the Lac-operon model over 50,000s...")
    ts, xs, _ = ssa(x0, r, a, tspan=tspan, _plot=False, event=intervention)
    xs = np.array(xs)

    print(f"{PLOT} Plotting results...")
    plt.subplot(311)
    plt.plot(ts, xs[:, 7], color="orange")
    plt.xlabel("Time", _font)
    plt.ylabel("Lactose", _font)

    plt.subplot(312)
    plt.plot(ts, xs[:, 5], color="purple")
    plt.xlabel("Time", _font)
    plt.ylabel("RNA", _font)

    plt.subplot(313)
    plt.plot(ts, xs[:, 6], color="tab:blue")
    plt.xlabel("Time", _font)
    plt.ylabel("Z", _font)

    plt.suptitle("Lac-Operon model for 50,000s (SSA)", fontproperties=title_font)
    plt.show()


def main():
    print(f"{TITL}{' Simulating Chemical Evironments ':-^79}{RES}")
    stop = False
    while not stop:
        ans = input(
            f"\n{QST} Availiable examples:\n"
            "   [1] Dimerisation kinetics;\n"
            "   [2] Michael-Menten;\n"
            "   [3] Auto-regulating genetic network;\n"
            "   [4] Lac-operon model.\n"
            f"{QST} Which one would you like to run? [1, 2, 3, 4] "
        )
        exs = {"1": dimkin, "2": michmen, "3": argn, "4": lac}

        if ans in exs:
            exs[ans]()
        else:
            ans_err = input(
                f"\n{ERR}[*] Unfortunately such example {ans} "
                f"is not available.{RES}\n"
                f"{QST} Would you like to quit? [Y/n] "
            )
            if not ans_err or ans_err in ["y", "Y"]:
                stop = True
            else:
                continue

        ans_ex = input(
            f"\n{QST} Would you like to test another example? [Y/n] "
        )
        if ans_ex in ["n", "N"]:
            stop = True


if __name__ == "__main__":
    main()

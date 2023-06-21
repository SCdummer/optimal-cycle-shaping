import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import os

def plot_trajectories(xT, target, V, angles, u, l1=1, l2=2, c_eff_penalty=0.0, T=1.0, q1=None, q2=None, u2=None, plotting_dir="Figures",
                      m1=1.0, m2=1.0, g=9.81, k2=0.5):

    fig = plt.figure()
    ax = plt.axes()

    # change of coordinates
    xT[:, 1] = xT[:, 1] + xT[:, 0]
    target[:, 1] = target[:, 1] + target[:, 0]

    # compute double pendulum position
    x1 = l1 * np.sin(xT[:, 0])
    y1 = -l1 * np.cos(xT[:, 0])
    x2 = x1 + l2 * np.sin(xT[:, 1])
    y2 = y1 - l2 * np.cos(xT[:, 1])
    o = np.zeros_like(x1)

    #gravitational potential
    y1m = -l1 * np.cos(q1)
    y2m = y1m - l2 * np.cos(q2 + q1)
    V_grav = m1*g*y1m + m2*g*y2m

    # compute targets x,y positions
    xt1 = l1 * np.sin(target[:, 0])
    yt1 = -l1 * np.cos(target[:, 0])
    xt2 = xt1 + l2 * np.sin(target[:, 1])
    yt2 = yt1 - l2 * np.cos(target[:, 1])

    # spring potential
    V_spring = k2 * (1 / 2 * np.pi - q2) ** 2


    p1 = ax.scatter(x1, y1, c=np.linspace(0, T, xT.shape[0], endpoint=True), cmap='twilight', s=10)
    p2 = ax.scatter(x2, y2, c=np.linspace(0, T, xT.shape[0], endpoint=True), cmap='twilight', s=10)
    p3 = ax.scatter(x1[0], y1[0], s=30, color="none", edgecolor="blue")
    p4 = ax.scatter(x2[0], y2[0], s=30, color="none", edgecolor="blue")
    p5 = ax.scatter(xt2, yt2, c='r', s=30, marker='x')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = fig.colorbar(p2)
    cbar.set_label('time', rotation=90)
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumTrajectory' + str(c_eff_penalty) + '.png'))
    ax.clear()

    p1 = ax.scatter(xT[:, 0], xT[:, 1] - xT[:, 0], c=np.linspace(0, T, xT.shape[0], endpoint=True), cmap='twilight', s=10)
    p2 = ax.scatter(xT[0, 0], xT[0, 1] - xT[0, 0], s=30, color="none", edgecolor="blue")
    xlimits = (min(np.min(xT[:, 0]) * 1.05, -np.pi), max(np.max(xT[:,0]) * 1.05, np.pi))
    ylimits = (min(np.min(xT[:, 1]-xT[:, 0]) * 1.05, -np.pi), max(np.max(xT[:, 1]-xT[:, 0]) * 1.05, np.pi))
    ax.set_xlim(xlimits[0], xlimits[1])
    ax.set_ylim(ylimits[0], ylimits[1])
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$q_2$')
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumTrajectory_angles_over_time_' + str(c_eff_penalty) + '.png'))
    ax.clear()

    c1 = plt.cm.twilight(np.linspace(0, 1, xT.shape[0], endpoint=True))
    c2 = plt.cm.twilight(np.linspace(0, 1, xT.shape[0], endpoint=True))

    # plot snapshots
    for i in range(0, x2.shape[0], 10):
        ax.clear()
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        l1 = ax.plot((o[i], x1[i]), (o[i], y1[i]), color=c1[i])
        l2 = ax.plot((x2[i], x1[i]), (y2[i], y1[i]), color=c2[i])
        p1 = ax.scatter(x1[0:i], y1[0:i], color=c1[0:i], s=10)
        p2 = ax.scatter(x2[0:i], y2[0:i], color=c2[0:i], s=10)
        p3 = ax.scatter(x1[0], y1[0], s=30, color="none", edgecolor="blue")
        p4 = ax.scatter(x2[0], y2[0], s=30, color="none", edgecolor="blue")
        p5 = ax.scatter(xt2, yt2, c='r', s=30, marker='x')
        p0 = ax.scatter(o[0], o[0], c='k', s=30, zorder=10)
        plt.xlabel('x')
        plt.ylabel('y')

        results_dir = os.path.join(plotting_dir, 'Snapshots')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        plt.savefig(os.path.join(results_dir, 'DoublePendulumTrajectory_t='+str(i) + '_' + str(c_eff_penalty) + '.png'))
    i = 999
    ax.clear()
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    l1 = ax.plot((o[i], x1[i]), (o[i], y1[i]), color=c1[i])
    l2 = ax.plot((x2[i], x1[i]), (y2[i], y1[i]), color=c2[i])
    p1 = ax.scatter(x1[0:i], y1[0:i], color=c1[0:i], s=10)
    p2 = ax.scatter(x2[0:i], y2[0:i], color=c2[0:i], s=10)

    p3 = ax.scatter(x1[0], y1[0], s=30, color="none", edgecolor="blue")
    p4 = ax.scatter(x2[0], y2[0], s=30, color="none", edgecolor="blue")
    p5 = ax.scatter(xt2, yt2, c='r', s=30, marker='x')
    p0 = ax.scatter(o[0], o[0], c='k', s=30, zorder=10)
    plt.xlabel('x')
    plt.ylabel('y')
    results_dir = os.path.join(plotting_dir, 'Snapshots')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, 'DoublePendulumTrajectory_t=' + str(i) + '_' + str(c_eff_penalty) + '.png'))

    ### trajectory animation (gif)
    def animate(i, x1, x2, y1, y2, o, c1, c2, xt2, yt2):
        ax.clear()
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        l1 = ax.plot((o[i], x1[i]), (o[i], y1[i]), color=c1[i])
        l2 = ax.plot((x2[i], x1[i]), (y2[i], y1[i]), color=c2[i])
        p1 = ax.scatter(x1[0:i], y1[0:i], color=c1[0:i], s=10)
        p2 = ax.scatter(x2[0:i], y2[0:i], color=c2[0:i], s=10)
        p3 = ax.scatter(x1[0], y1[0], s=30, color="none", edgecolor="blue")
        p4 = ax.scatter(x2[0], y2[0], s=30, color="none", edgecolor="blue")
        p5 = ax.scatter(xt2, yt2, c='r', s=30, marker='x')
        p0 = ax.scatter(o[0], o[0], c='k', s=30, zorder=10)
        plt.xlabel('x')
        plt.ylabel('y')
        return p1, p2

    gif = FuncAnimation(fig, animate, fargs=(x1, x2, y1, y2, o, c1, c2, xt2, yt2), blit=True, repeat=True,
                        frames=xT.shape[0], interval=1)
    gif.save(os.path.join(plotting_dir, "DoublePendulumTrajectory_" + str(c_eff_penalty) + "_.gif"), dpi=150, writer=PillowWriter(fps=30))
    ax.clear()

    ### Learned potential
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    p = ax4.scatter(q1, q2, c=V, cmap='magma', label=r'$V_{\theta}$')
    ax4.legend()
    fig4.colorbar(p)
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$q_2$')
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_Potential' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    ### Spring Potential
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    p = ax4.scatter(q1, q2, c=V_spring, cmap='magma', label=r'$V_{spring}$')
    ax4.legend()
    cbar = fig4.colorbar(p)
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$q_2$')
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_SpringPotential' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    ### Gravitational Potential
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    p = ax4.scatter(q1, q2, c=V_grav, cmap='magma', label=r'$V_{gravity}$')
    ax4.legend()
    cbar = fig4.colorbar(p)
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$q_2$')
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_GravityPotential' + str(c_eff_penalty) + '.png'))
    ax4.clear()


    ### Gravitational + Spring Potential
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    p = ax4.scatter(q1, q2, c=V_grav+V_spring, cmap='magma', label=r'$V_{gravity}+V_{spring}$')
    ax4.legend()
    cbar = fig4.colorbar(p)
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$q_2$')
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_GravitySpringPotential' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    ### Total Potential (Learned + Gravity + Spring)
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    p = ax4.scatter(q1, q2, c=V+V_grav, cmap='magma', label=r'$V_{\theta} + V_{gravity} + V_{spring}$')
    ax4.legend()
    fig4.colorbar(p)
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$q_2$')
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_OverallPotential' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    # grad potential wrt time
    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    p = ax4.scatter(t, u2[:, 0], c=t, cmap='twilight', s=5, label=r'$u_1(t)=(\nabla_{q_1}V_{\theta})(q(t))$')
    ax4.legend()
    cbar = fig4.colorbar(p)
    cbar.set_label('time', rotation=90)
    plt.xlabel('t')
    plt.ylabel(r'$u_1$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_GradPotential_q1_' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    # grad potential wrt time
    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    plt.plot(t, u2[:, 0], c='tab:blue', label=r'$u_1(t)=(\nabla_{q_1}V_{\theta})(q(t))$', linewidth=3)
    ax4.legend()
    plt.xlabel('t')
    plt.ylabel(r'$u_1$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_GradPotential_q1_plot' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    # grad potential wrt time
    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    p = ax4.scatter(t, u2[:, 1], c=t, cmap='twilight', s=5, label=r'$u_2(t)=(\nabla_{q_2}V_{\theta})(q(t))$')
    ax4.legend()
    cbar = fig4.colorbar(p)
    cbar.set_label('time', rotation=90)
    plt.xlabel('t')
    plt.ylabel(r'$u_2$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_GradPotential_q2_' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    # grad potential wrt time
    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    plt.plot(t, u2[:, 1], c='tab:orange', label=r'$u_2(t)=(\nabla_{q_2}V_{\theta})(q(t))$', linewidth=3)
    ax4.legend()
    plt.xlabel('t')
    plt.ylabel(r'$u_2$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_GradPotential_q2_plot' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    # grad potential wrt trajectory
    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    plt.plot(xT[:, 0], u2[:, 0], c='tab:blue', label=r'$u_1(t)=(\nabla_{q_1}V_{\theta})(q(t))$', linewidth=3)
    ax4.legend()
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$u_1$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_GradPotential_q1_traj_plot' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    # grad potential wrt trajectory
    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    plt.plot(xT[:, 1], u2[:, 1], c='tab:orange', label=r'$u_2(t)=(\nabla_{q_2}V_{\theta})(q(t))$', linewidth=3)
    ax4.legend()
    plt.xlabel(r'$q_2$')
    plt.ylabel(r'$u_2$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_GradPotential_q2_traj_plot' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    # q1 against time
    t = np.linspace(0, 1 * T, xT.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    plt.plot(t, xT[:, 0], c='tab:blue', linewidth=3)
    ax4.legend()
    plt.xlabel('t')
    plt.ylabel(r'$q_1$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_q1_against_time_plot' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    # q2 against time
    t = np.linspace(0, 1 * T, xT.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    plt.plot(t, xT[:, 1] - xT[:, 0], c='tab:orange', linewidth=3)
    ax4.legend()
    plt.xlabel('t')
    plt.ylabel(r'$q_2$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_q2_against_time_plot' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    p = ax4.scatter(t, np.square(u2[:, 0]), c=t, cmap='twilight', s=5, label="$||u_1||^2$")
    ax4.legend()
    cbar = fig4.colorbar(p)
    cbar.set_label('time', rotation=90)
    plt.xlabel('t')
    plt.ylabel(r'$||u_1||^2$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_SquaredControlEffort_u1_' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    plt.plot(t, np.square(u2[:, 0]), c='tab:blue', label="$||u_1||^2$", linewidth=3)
    ax4.legend()
    plt.xlabel('t')
    plt.ylabel(r'$||u_1||^2$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_SquaredControlEffort_u1_plot' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    p = ax4.scatter(t, np.square(u2[:, 1]), c=t, cmap='twilight', s=5, label="$||u_2||^2$")
    ax4.legend()
    cbar = fig4.colorbar(p)
    cbar.set_label('time', rotation=90)
    plt.xlabel('t')
    plt.ylabel(r'$||u_2||^2$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_SquaredControlEffort_u2_' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    plt.plot(t, np.square(u2[:, 1]), c='tab:orange', label="$||u_2||^2$")
    ax4.legend()
    plt.xlabel('t')
    plt.ylabel(r'$||u_2||^2$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_SquaredControlEffort_u2_plot' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    p1 = ax4.scatter(t, np.square(u2[:, 0]), s=1, label="$||u_1||^2$", linewidth=3)
    p2 = ax4.scatter(t, np.square(u2[:, 1]), s=1, label="$||u_2||^2$", linewidth=3)
    p3 = ax4.scatter(t, np.square(u2[:, 0]) + np.square(u2[:, 1]), c=t, cmap='twilight', s=5, label="$||u_1||^2 + ||u_2||^2$", linewidth=3)
    ax4.legend()
    cbar = fig4.colorbar(p3)
    cbar.set_label('time', rotation=90)
    plt.xlabel('t')
    plt.ylabel(r'$||u||^2$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_SquaredControlEffort_u1u2_' + str(c_eff_penalty) + '.png'))
    ax4.clear()

    t = np.linspace(0, 1 * T, u2.shape[0])
    fig4 = plt.figure()
    ax4 = plt.axes()
    c = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
    plt.plot(t, np.square(u2[:, 0]), c='tab:blue', label="$||u_1||^2$", linestyle='dashed', linewidth=2)
    plt.plot(t, np.square(u2[:, 1]), c='tab:orange', label="$||u_2||^2$", linestyle='dashed', linewidth=2)
    plt.plot(t, np.square(u2[:, 0]) + np.square(u2[:, 1]), c='tab:green', label="$||u_1||^2 + ||u_2||^2$", linewidth=3)
    ax4.legend()
    plt.xlabel('t')
    plt.ylabel(r'$||u||^2$')
    fig4.tight_layout()
    plt.savefig(os.path.join(plotting_dir, 'DoublePendulumLearned_SquaredControlEffort_u1u2_plot' + str(c_eff_penalty) + '.png'))
    ax4.clear()


def animate_single_dp_trajectory(xT_raw, l1=1, l2=1, plotting_dir="Figures"):
    l1, l2 = l1, l2
    fig = plt.figure()
    ax = plt.axes()

    xT = xT_raw

    # change coordinates
    xT[:, 1] = xT[:, 1] + xT[:, 0]

    # compute double pendulum position
    x1 = l1 * np.sin(xT[:, 0])
    y1 = -l1 * np.cos(xT[:, 0])
    x2 = x1 + l2 * np.sin(xT[:, 1])
    y2 = y1 - l2 * np.cos(xT[:, 1])
    o = np.zeros_like(x1)

    c1 = plt.cm.magma(np.linspace(0, 1, xT.shape[0], endpoint=False))
    c2 = plt.cm.rainbow(np.linspace(0, 1, xT.shape[0], endpoint=False))

    def animate(i, x1, x2, y1, y2, o, c1, c2):
        ax.clear()
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        p1 = ax.scatter(x1[0:i], y1[0:i], color=c1[0:i], s=10)
        p2 = ax.scatter(x2[0:i], y2[0:i], color=c2[0:i], s=10)
        l1 = ax.plot((o[i], x1[i]), (o[i], y1[i]), color=c1[i])
        l2 = ax.plot((x2[i], x1[i]), (y2[i], y1[i]), color=c2[i])

        p0 = ax.scatter(o[0], o[0], c='k', s=30, zorder=10)
        return p1, p2

    gif = FuncAnimation(fig, animate, fargs=(x1, x2, y1, y2, o, c1, c2),
                        blit=True, repeat=True, frames=xT.shape[0], interval=1)
    gif.save(os.path.join(plotting_dir, "DoublePendulumEigenModeTrajectory.gif"), dpi=150, writer=PillowWriter(fps=30))
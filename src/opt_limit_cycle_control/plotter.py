import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, ImageMagickWriter, HTMLWriter, FFMpegWriter, ImageMagickFileWriter
import seaborn as sns



def plot_trajectories(xT, target, l1=1, l2=2, pendulum=True, plot3D=False):

    if pendulum:
        if plot3D==False:
            l1 = l1
            fig = plt.figure()
            # ax = plt.axes(projection='3d')
            ax = plt.axes()

            # compute pendulum position
            x1 = l1 * np.sin(xT[:, 0])
            y1 = -l1 * np.cos(xT[:, 0])
            o = np.zeros_like(x1)

            # compute targets x,y positions
            xt1 = l1 * np.sin(target[0])
            yt1 = -l1 * np.cos(target[0])
            xt2 = l1 * np.sin(target[1])
            yt2 = -l1 * np.cos(target[1])

            c1 = plt.cm.magma(np.linspace(0, 1, xT.shape[0], endpoint=False))

            def animate(i, x1, y1, o, c1, xt1, yt1, xt2, yt2):
                ax.clear()
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                l1 = ax.plot((o[i], x1[i]), (o[i], y1[i]), color=c1[i])
                p1 = ax.scatter(x1[0:i], y1[0:i], color=c1[0:i], s=10)

                pt1 = ax.scatter(xt1, yt1, c='g', s=30, marker='x')
                pt2 = ax.scatter(xt2, yt2, c='b', s=30, marker='x')
                p0 = ax.scatter(o[0], o[0], c='k', s=30, zorder=10)
                return p1, l1

            gif = FuncAnimation(fig, animate, fargs=(x1, y1, o, c1, xt1, yt1, xt2, yt2),
                                blit=False, repeat=True, frames=xT.shape[0], interval=1)
            gif.save("Figures/PendulumTrajectory.gif", dpi=150, writer=PillowWriter(fps=30))
            ax.clear()

            p1 = ax.scatter(x1, y1, c=np.linspace(0, 1, xT.shape[0], endpoint=False),
                            cmap='magma', s=10)
            fig.colorbar(p1)
            plt.savefig('Figures/PendulumTrajectory')

        else:
            pass
    else:
        if plot3D==False:
            l1, l2 = l1, l2
            fig = plt.figure()
            # ax = plt.axes(projection='3d')
            ax = plt.axes()

            # compute double pendulum position
            x1 = l1 * np.sin(xT[:, 0])
            y1 = -l1 * np.cos(xT[:, 0])
            x2 = x1 + l2 * np.sin(xT[:, 1])
            y2 = y1 - l2 * np.cos(xT[:, 1])
            o = np.zeros_like(x1)

            # compute targets x,y positions
            xt1 = l1 * np.sin(target[0])
            yt1 = -l1 * np.cos(target[0])
            xt2 = xt1 + l2 * np.sin(target[2])
            yt2 = yt1 - l2 * np.cos(target[2])
            xt3 = l1 * np.sin(target[1])
            yt3 = -l1 * np.cos(target[1])
            xt4 = xt1 + l2 * np.sin(target[3])
            yt4 = yt1 - l2 * np.cos(target[3])

            c1 = plt.cm.magma(np.linspace(0, 1, xT.shape[0], endpoint=False))
            c2 = plt.cm.rainbow(np.linspace(0, 1, xT.shape[0], endpoint=False))

            def animate(i, x1, x2, y1, y2, o, c1, c2, xt1, yt1, xt2, yt2, xt3, yt3, xt4, yt4):
                ax.clear()
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-2.5, 2.5)
                l1 = ax.plot((o[i], x1[i]), (o[i], y1[i]), color=c1[i])
                l2 = ax.plot((x2[i], x1[i]), (y2[i], y1[i]), color=c2[i])
                p1 = ax.scatter(x1[0:i], y1[0:i], color=c1[0:i], s=10)
                p2 = ax.scatter(x2[0:i], y2[0:i], color=c2[0:i], s=10)

                pt1 = ax.scatter(xt1, yt1, c='g', s=30, marker='x')
                pt2 = ax.scatter(xt2, yt2, c='b', s=30, marker='x')
                pt3 = ax.scatter(xt3, yt3, c='g', s=30, marker='x')
                pt4 = ax.scatter(xt4, yt4, c='b', s=30, marker='x')
                p0 = ax.scatter(o[0], o[0], c='k', s=30, zorder=10)
                return p1, p2

            gif = FuncAnimation(fig, animate, fargs=(x1, x2, y1, y2, o, c1, c2, xt1, yt1, xt2, yt2, xt3, yt3, xt4, yt4),
                                blit=True, repeat=True, frames=xT.shape[0], interval=1)
            gif.save("Figures/DoublePendulumTrajectory.gif", dpi=150, writer=PillowWriter(fps=30))
            ax.clear()

            p1 = ax.scatter(x1, y1, c=np.linspace(0, 1, xT.shape[0], endpoint=False),
                            cmap='magma', s=10)
            p2 = ax.scatter(x2, y2, c=np.linspace(0, 1, xT.shape[0], endpoint=False),
                            cmap='rainbow', s=10)
            fig.colorbar(p1)
            fig.colorbar(p2)
            plt.savefig('Figures/DoublePendulumTrajectory')

        else:
            pass

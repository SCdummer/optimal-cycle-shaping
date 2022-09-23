import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, ImageMagickWriter, HTMLWriter, FFMpegWriter, ImageMagickFileWriter
import os

m, l, g = 1., 1, 9.81

m1, m2 = 1., 1.

def plot_trajectories(xT, target, V, angles, u, l1=1, l2=2, pendulum=True, plot3D=False, c_eff_penalty=0.0, T=1.0, q1=[], q2=[], u2=[], plotting_dir="Figures"):

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
            xt = l1 * np.sin(target)
            yt = -l1 * np.cos(target)
            # xt2 = l1 * np.sin(target[1])
            # yt2 = -l1 * np.cos(target[1])

            xall = l1 * np.sin(angles)
            yall = -l1 * np.cos(angles)

            # compute gravitational potential energy
            Vg = m * l1 * np.sin(angles)

            c1 = plt.cm.magma(np.linspace(0, 1, xT.shape[0], endpoint=False))

            def animate(i, x1, y1, o, c1, xt, yt):
                ax.clear()
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                l1 = ax.plot((o[i], x1[i]), (o[i], y1[i]), color=c1[i])
                p1 = ax.scatter(x1[0:i], y1[0:i], color=c1[0:i], s=10)

                pt = ax.scatter(xt, yt, c='g', s=30, marker='x')
                p0 = ax.scatter(o[0], o[0], c='k', s=30, zorder=10)
                return p1, l1

            gif = FuncAnimation(fig, animate, fargs=(x1, y1, o, c1, xt, yt),
                                blit=False, repeat=True, frames=xT.shape[0], interval=1)
            gif.save(os.path.join(plotting_dir, "PendulumTrajectory_") + str(c_eff_penalty) + "_.gif", dpi=150, writer=PillowWriter(fps=30))
            ax.clear()

            p1 = ax.scatter(x1, y1, c=np.linspace(0, 1, xT.shape[0], endpoint=False),
                            cmap='magma', s=10)
            ax.scatter(xt, yt, c='g', s=30, marker='x')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            fig.colorbar(p1)
            plt.savefig(os.path.join(plotting_dir, 'PendulumTrajectory' + str(c_eff_penalty)+ '.png'))

            fig2 = plt.figure()
            ax2 = plt.axes()
            p2 = ax2.scatter(xall, yall, c=V, cmap='magma', s=15)
            pt = ax2.scatter(xt, yt, c='g', s=30, marker='x')
            fig2.colorbar(p2)
            plt.savefig(os.path.join(plotting_dir, 'Potential' + str(c_eff_penalty)+ '.png'))

            fig3 = plt.figure()
            ax3 = plt.axes()
            p3 = ax3.scatter(xall, yall, c=u, cmap='magma', s=15)
            ax3.scatter(xt, yt, c='g', s=30, marker='x')
            fig3.colorbar(p3)
            plt.savefig(os.path.join(plotting_dir,'ControlInput' + str(c_eff_penalty)+ '.png'))

            t = np.linspace(0, 1, angles.shape[0])
            fig4 = plt.figure()
            ax4 = plt.axes()
            ax4.scatter(target[0], 0, c='g', s=30, marker='x')
            ax4.scatter(target[1], 0, c='b', s=30, marker='x')
            p7 = ax4.scatter(angles, V + Vg, c=t, cmap='RdYlGn', s=3, label='V total')
            p5 = ax4.scatter(angles, Vg, c=t, cmap='magma', s=7, label='V gravity')
            p6 = ax4.scatter(angles, V, c=t, cmap='rainbow', s=7, label='V learned')
            ax4.legend()
            fig4.colorbar(p5)
            fig4.colorbar(p6)
            fig4.colorbar(p7)
            fig4.tight_layout()
            plt.savefig(os.path.join(plotting_dir,'Learned_Gravitational_Potential' + str(c_eff_penalty)+ '.png'))

            t = np.linspace(0, 1*T, 1000)
            q = np.linspace(np.min(xT[:,0]), np.max(xT[:,0]), 1000)
            fig4 = plt.figure()
            ax4 = plt.axes()
            ax4.scatter(target[0], 0, c='g', s=30, marker='x')
            ax4.scatter(target[1], 0, c='b', s=30, marker='x')

            c3 = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
            for i in range(target.size(0)):
                ax4.scatter(target[i, 0], 0, c=c3[i], s=30, marker='x')
            p7 = ax4.scatter(q, V + Vg, c=t, cmap='RdYlGn', s=3, label='V total')
            p5 = ax4.scatter(q, Vg, c=t, cmap='magma', s=7, label='V gravity')
            p6 = ax4.scatter(q, V, c=t, cmap='rainbow', s=7, label='V learned')
            ax4.legend()
            fig4.colorbar(p5)
            fig4.colorbar(p6)
            fig4.colorbar(p7)
            fig4.tight_layout()
            plt.savefig(os.path.join(plotting_dir, 'Learned_Gravitational_Potential_Interval' + str(c_eff_penalty)+ '.png'))

        else:
            pass
    else:
        if plot3D==False:
            l1, l2 = l1, l2
            fig = plt.figure()
            # ax = plt.axes(projection='3d')
            ax = plt.axes()

            # change coordinates
            xT[:, 1] = xT[:, 1] + xT[:, 0]
            target[:, 1] = target[:, 1] + target[:, 0]

            # compute double pendulum position
            x1 = l1 * np.sin(xT[:, 0])
            y1 = -l1 * np.cos(xT[:, 0])
            x2 = x1 + l2 * np.sin(xT[:, 1])
            y2 = y1 - l2 * np.cos(xT[:, 1])
            o = np.zeros_like(x1)

            #gravitational potential

            # V gravity = m1gy1 + m2gy2

            y1m = -l1 * np.cos(q1)
            y2m = y1m - l2 * np.cos(q2)
            V_grav = m1*g*y1m + m2*g*y2m

            # compute targets x,y positions
            xt1 = l1 * np.sin(target[:, 0])
            yt1 = -l1 * np.cos(target[:, 0])
            xt2 = xt1 + l2 * np.sin(target[:, 1])
            yt2 = yt1 - l2 * np.cos(target[:, 1])


            p1 = ax.scatter(x1, y1, c=np.linspace(0, 1, xT.shape[0], endpoint=False),
                            cmap='twilight', s=10)
            p2 = ax.scatter(x2, y2, c=np.linspace(0, 1, xT.shape[0], endpoint=False),
                            cmap='twilight', s=10)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            fig.colorbar(p1)
            fig.colorbar(p2)
            plt.savefig(os.path.join(plotting_dir,'DoublePendulumTrajectory' + str(c_eff_penalty) + '.png'))
            ax.clear()

            c1 = plt.cm.twilight(np.linspace(0, 1, xT.shape[0], endpoint=False))
            c2 = plt.cm.twilight(np.linspace(0, 1, xT.shape[0], endpoint=False))

            def animate(i, x1, x2, y1, y2, o, c1, c2, xt1, yt1, xt2, yt2):
                ax.clear()
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-2.5, 2.5)
                l1 = ax.plot((o[i], x1[i]), (o[i], y1[i]), color=c1[i])
                l2 = ax.plot((x2[i], x1[i]), (y2[i], y1[i]), color=c2[i])
                p1 = ax.scatter(x1[0:i], y1[0:i], color=c1[0:i], s=10)
                p2 = ax.scatter(x2[0:i], y2[0:i], color=c2[0:i], s=10)

                # pt1 = ax.scatter(xt1, yt1, c='g', s=30, marker='x')
                pt2 = ax.scatter(xt2, yt2, c='b', s=30, marker='x')
                p0 = ax.scatter(o[0], o[0], c='k', s=30, zorder=10)
                return p1, p2

            gif = FuncAnimation(fig, animate, fargs=(x1, x2, y1, y2, o, c1, c2, xt1, yt1, xt2, yt2),
                                blit=True, repeat=True, frames=xT.shape[0], interval=1)
            gif.save(os.path.join(plotting_dir,"DoublePendulumTrajectory_" + str(c_eff_penalty) + "_.gif"), dpi=150,
                     writer=PillowWriter(fps=30))
            ax.clear()

            #ang = np.linspace(-np.pi, np.pi, angles.shape[0])
            #ang1 = np.linspace(np.min(xT[:, 0]), np.max(xT[:, 0]), 1000)
            #ang2 = np.linspace(np.min(xT[:, 1]), np.max(xT[:, 1]), 1000)
            fig4 = plt.figure()
            ax4 = plt.axes()
            c3 = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
            #for i in range(target.size(0)):
            #    ax4.scatter(target[i, 0], 0, c=c3[i], s=30, marker='x')
            #    ax4.scatter(target[i, 1], -1, c=c3[i], s=30, marker='x')
            #p7 = ax4.scatter(angles[:, 0], V[:, 0], c=ang, cmap='twilight', s=5, label='V1', marker=',')
            #p5 = ax4.scatter(angles[:, 1], V[:, 1], c=ang, cmap='twilight_shifted', s=5, label='V2')
            #p6 = ax4.pcolormesh(angles[:, 0], angles[:, 1], V.reshape(1000,), shading='gouraud', cmap='twilight')
            p6 = ax4.scatter(q1, q2, c=V, cmap='magma', label='V')
            #p6 = ax4.scatter(angles, V, c=t, cmap='rainbow', s=7, label='V learned')
            ax4.legend()
            #fig4.colorbar(p5)
            fig4.colorbar(p6)
            #fig4.colorbar(p7)
            ax4.set_title('Learned Potential (T={:.3f})'.format(T))
            #ax.set_xlabel('Angles', fontsize=10)
            #ax.set_ylabel('Value', fontsize=10)
            #fig4.tight_layout()
            plt.savefig(os.path.join(plotting_dir,'DoublePendulumLearned_Potential' + str(c_eff_penalty) + '.png'))
            ax4.clear()

            fig4 = plt.figure()
            ax4 = plt.axes()
            c3 = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
            # for i in range(target.size(0)):
            #    ax4.scatter(target[i, 0], 0, c=c3[i], s=30, marker='x')
            #    ax4.scatter(target[i, 1], -1, c=c3[i], s=30, marker='x')
            # p7 = ax4.scatter(angles[:, 0], V[:, 0], c=ang, cmap='twilight', s=5, label='V1', marker=',')
            # p5 = ax4.scatter(angles[:, 1], V[:, 1], c=ang, cmap='twilight_shifted', s=5, label='V2')
            # p6 = ax4.pcolormesh(angles[:, 0], angles[:, 1], V.reshape(1000,), shading='gouraud', cmap='twilight')
            p6 = ax4.scatter(q1, q2, c=V_grav, cmap='magma', label='V gravity')
            # p6 = ax4.scatter(angles, V, c=t, cmap='rainbow', s=7, label='V learned')
            ax4.legend()
            # fig4.colorbar(p5)
            fig4.colorbar(p6)
            # fig4.colorbar(p7)
            ax4.set_title('Learned Potential (T={:.3f})'.format(T))
            # ax.set_xlabel('Angles', fontsize=10)
            # ax.set_ylabel('Value', fontsize=10)
            # fig4.tight_layout()
            plt.savefig(os.path.join(plotting_dir,'DoublePendulumLearned_GravityPotential' + str(c_eff_penalty) + '.png'))
            ax4.clear()

            fig4 = plt.figure()
            ax4 = plt.axes()
            c3 = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
            # for i in range(target.size(0)):
            #    ax4.scatter(target[i, 0], 0, c=c3[i], s=30, marker='x')
            #    ax4.scatter(target[i, 1], -1, c=c3[i], s=30, marker='x')
            # p7 = ax4.scatter(angles[:, 0], V[:, 0], c=ang, cmap='twilight', s=5, label='V1', marker=',')
            # p5 = ax4.scatter(angles[:, 1], V[:, 1], c=ang, cmap='twilight_shifted', s=5, label='V2')
            # p6 = ax4.pcolormesh(angles[:, 0], angles[:, 1], V.reshape(1000,), shading='gouraud', cmap='twilight')
            p6 = ax4.scatter(q1, q2, c=V+V_grav, cmap='magma', label='V + V gravity')
            # p6 = ax4.scatter(angles, V, c=t, cmap='rainbow', s=7, label='V learned')
            ax4.legend()
            # fig4.colorbar(p5)
            fig4.colorbar(p6)
            # fig4.colorbar(p7)
            ax4.set_title('Learned Potential (T={:.3f})'.format(T))
            # ax.set_xlabel('Angles', fontsize=10)
            # ax.set_ylabel('Value', fontsize=10)
            # fig4.tight_layout()
            plt.savefig(os.path.join(plotting_dir,'DoublePendulumLearned_OverallPotential' + str(c_eff_penalty) + '.png'))
            ax4.clear()

            # grad potential
            ang = np.linspace(-np.pi, np.pi, u2.shape[0])
            fig4 = plt.figure()
            ax4 = plt.axes()
            c3 = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
            #for i in range(target.size(0)):
            #    ax4.scatter(target[i, 0], 0, c=c3[i], s=30, marker='x')
            #    ax4.scatter(target[i, 1], -1, c=c3[i], s=30, marker='x')
            p7 = ax4.scatter(ang, u2[:, 0], c=u2[:, 0], cmap='twilight', s=5, label='grad V wrt q1')
            #p8 = ax4.plot(ang, u2[:, 0], cmap='viridis', s=5, label='grad V wrt q1')
            #p5 = ax4.scatter(ang, u2[:, 1], c=u2[:, 1], cmap='magma', s=5, label='grad V wrt q2')
            #p5 = ax4.scatter(q2, u[:, 1], c=ang, cmap='twilight_shifted', s=5, label='grad V wrt q2')

            #p6 = ax4.scatter(angles, V, c=t, cmap='rainbow', s=7, label='V learned')
            ax4.legend()
            #fig4.colorbar(p5)
            #fig4.colorbar(p6)
            fig4.colorbar(p7)
            ax4.set_title('Gradient of Learned Potential (T={:.3f})'.format(T))
            #ax.set_xlabel('Angles', fontsize=10)
            #ax.set_ylabel('Value', fontsize=10)
            fig4.tight_layout()
            plt.savefig(os.path.join(plotting_dir,'DoublePendulumLearned_GradPotential_q1_' + str(c_eff_penalty) + '.png'))
            ax4.clear()

            # grad potential
            ang = np.linspace(-np.pi, np.pi, u2.shape[0])
            fig4 = plt.figure()
            ax4 = plt.axes()
            c3 = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
            #for i in range(target.size(0)):
            #    ax4.scatter(target[i, 0], 0, c=c3[i], s=30, marker='x')
            #    ax4.scatter(target[i, 1], -1, c=c3[i], s=30, marker='x')
            #p7 = ax4.scatter(ang, u2[:, 0], c=u2[:, 0], cmap='viridis', s=5, label='grad V wrt q1')
            #p8 = ax4.plot(ang, u2[:, 0], cmap='viridis', s=5, label='grad V wrt q1')
            p5 = ax4.scatter(ang, u2[:, 1], c=u2[:, 1], cmap='twilight', s=5, label='grad V wrt q2')
            #p5 = ax4.scatter(q2, u[:, 1], c=ang, cmap='twilight_shifted', s=5, label='grad V wrt q2')

            #p6 = ax4.scatter(angles, V, c=t, cmap='rainbow', s=7, label='V learned')
            ax4.legend()
            fig4.colorbar(p5)
            #fig4.colorbar(p6)
            #fig4.colorbar(p7)
            ax4.set_title('Gradient of Learned Potential (T={:.3f})'.format(T))
            #ax.set_xlabel('Angles', fontsize=10)
            #ax.set_ylabel('Value', fontsize=10)
            fig4.tight_layout()
            plt.savefig(os.path.join(plotting_dir,'DoublePendulumLearned_GradPotential_q2_' + str(c_eff_penalty) + '.png'))
            ax4.clear()

            # grad potential
            #t = np.linspace(0, 1*T, u2.shape[0])
            #fig4 = plt.figure()
            #ax4 = plt.axes()
            #c3 = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
            #for i in range(target.size(0)):
            #    ax4.scatter(target[i, 0], 0, c=c3[i], s=30, marker='x')
            #    ax4.scatter(target[i, 1], -1, c=c3[i], s=30, marker='x')
            #p7 = ax4.scatter(t, u2[:, 0], c=u2[:, 0], cmap='twilight', s=5, label='grad V wrt q1')
            #p5 = ax4.scatter(t, u2[:, 1], c=u2[:, 1], cmap='magma', s=5, label='grad V wrt q2')
            #p5 = ax4.scatter(q2, u[:, 1], c=ang, cmap='twilight_shifted', s=5, label='grad V wrt q2')

            #p6 = ax4.scatter(angles, V, c=t, cmap='rainbow', s=7, label='V learned')
            #ax4.legend()
            #fig4.colorbar(p5)
            #fig4.colorbar(p6)
            #fig4.colorbar(p7)
            #ax4.set_title('Gradient of Learned Potential (T={:.3f})'.format(T))
            #ax.set_xlabel('Angles', fontsize=10)
            #ax.set_ylabel('Value', fontsize=10)
            #fig4.tight_layout()
            #plt.savefig(os.path.join(plotting_dir,'DoublePendulumLearned_GradPotentialScaled_q1_' + str(c_eff_penalty) + '.png'))
            #ax4.clear()
            # grad potential
            #t = np.linspace(0, 1 * T, u2.shape[0])
            #fig4 = plt.figure()
            #ax4 = plt.axes()
            #c3 = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
            # for i in range(target.size(0)):
            #    ax4.scatter(target[i, 0], 0, c=c3[i], s=30, marker='x')
            #    ax4.scatter(target[i, 1], -1, c=c3[i], s=30, marker='x')
            #p7 = ax4.scatter(t, u2[:, 0], c=u2[:, 0], cmap='viridis', s=5, label='grad V wrt q1')
            #p5 = ax4.scatter(t, u2[:, 1], c=u2[:, 1], cmap='twilight', s=5, label='grad V wrt q2')
            # p5 = ax4.scatter(q2, u[:, 1], c=ang, cmap='twilight_shifted', s=5, label='grad V wrt q2')

            # p6 = ax4.scatter(angles, V, c=t, cmap='rainbow', s=7, label='V learned')
            #ax4.legend()
            #fig4.colorbar(p5)
            # fig4.colorbar(p6)
            #fig4.colorbar(p7)
            #ax4.set_title('Gradient of Learned Potential (T={:.3f})'.format(T))
            # ax.set_xlabel('Angles', fontsize=10)
            # ax.set_ylabel('Value', fontsize=10)
            #fig4.tight_layout()
            #plt.savefig(os.path.join(plotting_dir,'DoublePendulumLearned_GradPotentialScaled_q2_' + str(c_eff_penalty) + '.png'))
            #ax4.clear()

            # grad potential
            #ang = np.linspace(0, 1, u2.shape[0])
            #fig4 = plt.figure()
            #ax4 = plt.axes()
            #c3 = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
            #for i in range(target.size(0)):
            #    ax4.scatter(target[i, 0], 0, c=c3[i], s=30, marker='x')
            #    ax4.scatter(target[i, 1], -1, c=c3[i], s=30, marker='x')
            #p5 = ax4.scatter(q1, q2, c=u[:, 1], cmap='viridis', s=5, label='grad V wrt q2')

            #p5 = ax4.scatter(ang, u2[:, 1], c=u2[:, 1], cmap='viridis', s=5, label='grad V wrt q2')
            #p6 = ax4.scatter(angles, V, c=t, cmap='rainbow', s=7, label='V learned')
            #ax4.legend()
            #fig4.colorbar(p5)
            #fig4.colorbar(p6)
            #fig4.colorbar(p7)
            #ax4.set_title('Gradient of Learned Potential (T={:.3f})'.format(T))
            ##ax.set_xlabel('Angles', fontsize=10)
            #ax.set_ylabel('Value', fontsize=10)
            #fig4.tight_layout()
            #plt.savefig(os.path.join(plotting_dir,'DoublePendulumLearned_GradPotential_q12_' + str(c_eff_penalty) + '.png'))
            #ax4.clear()

            #t = np.linspace(0, 1*T, q1.shape[0])
            #fig4 = plt.figure()
            #ax4 = plt.axes()
            #c3 = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
            # for i in range(target.size(0)):
            #    ax4.scatter(target[i, 0], 0, c=c3[i], s=30, marker='x')
            #    ax4.scatter(target[i, 1], -1, c=c3[i], s=30, marker='x')
            #p7 = ax4.scatter(angles[:, 0], V[:, 0], c=t, cmap='twilight', s=5, label='V1', marker=',')
            #p5 = ax4.scatter(q1, q2, c=t, cmap='twilight_shifted', s=5, label='V2')
            # p6 = ax4.scatter(angles, V, c=t, cmap='rainbow', s=7, label='V learned')
            #ax4.legend()
            #fig4.colorbar(p5)
            # fig4.colorbar(p6)
            #fig4.colorbar(p7)
            #ax4.set_title('Learned Potential (T={:.3f})'.format(T))
            #ax.set_xlabel('Time', fontsize=10)
            #ax.set_ylabel('Value', fontsize=10)
            #fig4.tight_layout()
            #plt.savefig(os.path.join(plotting_dir,'DoublePendulumLearned_Potential_ScaledT' + str(c_eff_penalty) + '.png'))

            #t = np.linspace(0, 1 * T, angles.shape[0])
            #fig4 = plt.figure()
            #ax4 = plt.axes()
            #c3 = plt.cm.viridis(np.linspace(0, 1, target.size(0), endpoint=False))
            # for i in range(target.size(0)):
            #    ax4.scatter(target[i, 0], 0, c=c3[i], s=30, marker='x')
            #    ax4.scatter(target[i, 1], -1, c=c3[i], s=30, marker='x')
            #p7 = ax4.scatter(angles[:, 0], u[:, 0], c=t, cmap='twilight', s=5, label='grad V1', marker=',')
            #p5 = ax4.scatter(angles[:, 1], u[:, 1], c=t, cmap='twilight_shifted', s=5, label='grad V2')
            # p6 = ax4.scatter(angles, V, c=t, cmap='rainbow', s=7, label='V learned')
            #ax4.legend()
            #fig4.colorbar(p5)
            # fig4.colorbar(p6)
            #fig4.colorbar(p7)
            #ax4.set_title('Gradient of Learned Potential (T={:.3f})'.format(T))
            # ax.set_xlabel('Time', fontsize=10)
            # ax.set_ylabel('Value', fontsize=10)
            #fig4.tight_layout()
            #plt.savefig(os.path.join(plotting_dir,'DoublePendulumLearned_GradPotential_ScaledT' + str(c_eff_penalty) + '.png'))





        else:
            pass


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
    gif.save(os.path.join(plotting_dir,"DoublePendulumEigenModeTrajectory.gif"), dpi=150, writer=PillowWriter(fps=30))




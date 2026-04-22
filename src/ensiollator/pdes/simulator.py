import os

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from src.ensiollator.pdes.shallow_water_pdes import NaiveShallowWaterPDE

import pde
from pde import PDEBase
from pde import CartesianGrid, FileStorage, ScalarField, VectorField, FieldCollection
from pde.visualization.movies import movie


def print_msg(msg : str):
    print("=" * 30)
    print(msg)
    print("=" * 30)


class ShallowWaterSimulator:
    def non_dimensionalization(self,
                               g_reduced = 0.05,
                               beta = 2.28 * (10 ** (-11)), 
                               rayleigh_coefficient = 1e-7,
                               unperturbed_thermocline_mean_depth = 50, 
                               air_sea_coupling_strength = 0.0
                               ):
        """
        Computing scaling constants for non dimensionalization
        """
        # geostrophic wave speed 
        self.c = np.sqrt(g_reduced * unperturbed_thermocline_mean_depth)
        
         # spatial scale
        self.L = np.sqrt(self.c / beta)
        
        # temporal scale
        self.T = (1) / (np.sqrt(self.c * beta))
        
        # velocity scale 
        self.U = self.L /self.T
        
        # non-dimensionalized rayleigh coefficient
        self.eps = rayleigh_coefficient * self.T
        
        # coupling strength
        self.gamma = air_sea_coupling_strength
        
        print_msg(f"Rossby Deformation Radius is {self.L} \n Time Scale is {self.T} \n Gamma(Coupling) is {self.gamma} \n Damping strength is {self.eps}")
        
        
    def __init__(self,
                 initial_height_expression : str,
                 resolution : int,
                 trade_wind_stress_expressions : str,
                 out_dir : str,
                 pde : type[PDEBase] = NaiveShallowWaterPDE,
                 solver : str = 'scipy',
                 basin_width = 1.5 * 1e7,
                 height_to_width_ratio = 0.3,
                 unperturbed_thermocline_mean_depth = 50,
                 rayleigh_coefficient = 1e-7,
                 air_sea_coupling_strength = 0.0
                 ):
        
        self.non_dimensionalization(rayleigh_coefficient=rayleigh_coefficient, 
                                    unperturbed_thermocline_mean_depth=unperturbed_thermocline_mean_depth,
                                    air_sea_coupling_strength=air_sea_coupling_strength
                                    )
        
        self.X, self.Y = basin_width / self.L, basin_width * height_to_width_ratio / self.L 
        
        # 1. Setup Grid
        grid = CartesianGrid([(-self.X / 2,  self.X / 2), (-self.Y / 2,  self.Y / 2)], [resolution, int(resolution * height_to_width_ratio)])

        print_msg(f"Grid Configuration: \n [{-self.X/2}, {self.X/2}] x [{-self.Y/2}, {self.Y/2}] \n Resolution is {resolution}, {int(resolution * height_to_width_ratio)}")
              
        # 2. Set up initial condtion 
        u_initial = ScalarField(grid=grid, data=np.zeros(grid.shape), label="u")
        v_initial = ScalarField(grid=grid, data=np.zeros(grid.shape), label="v")
        h_initial = ScalarField.from_expression(grid=grid, expression=initial_height_expression)
        
        self.state = FieldCollection([u_initial, v_initial, h_initial])
        
        # 3. Set up boundary condition 
        
        def set_hbc(h, args=None):
            """
                reference: https://empslocal.ex.ac.uk/people/staff/gv219/codes/linearshallowwater.py
            """
            #free-slip west/east
            h[0, :] = h[1, :]
            h[-1, :] = h[-2, :]
            
            #free slip at north and south boundary
            h[:, 0] = h[:, 1]
            h[:, -1] = h[:, -2]
            
            
            # fix corners to be average of neighbors, this is requirbed because corners are
            # east/west, south/north boundary at the same time, difference bc leads to vastly different values 
            # can cause artifical large gradient -> numerical instability
            h[0, 0] =  0.5*(h[1, 0] + h[0, 1])
            h[-1, 0] = 0.5*(h[-2, 0] + h[-1, 1])
            h[0, -1] = 0.5*(h[1, -1] + h[0, -2])
            h[-1, -1] = 0.5*(h[-1, -2] + h[-2, -1])
            
        
        def set_ubc(u, args=None):
            """
                Callback for setter of boundary condition for the scalar field u(x,y)
                reference: https://empslocal.ex.ac.uk/people/staff/gv219/codes/linearshallowwater.py
            """
            # wall at west/east boundary for u
            u[0, :] = 0
            u[1, :] = 0
            u[-1, :] = 0
            u[-2, :] = 0
            
            #free slip at north and south boundary
            u[:, 0] = u[:, 1]
            u[:, -1] = u[:, -2]
            
            # fix corners to be average of neighbors, this is requirbed because corners are
            # east/west, south/north boundary at the same time, difference bc leads to vastly different values 
            # can cause artifical large gradient -> numerical instability
            u[0, 0] =  0.5*(u[1, 0] + u[0, 1])
            u[-1, 0] = 0.5*(u[-2, 0] + u[-1, 1])
            u[0, -1] = 0.5*(u[1, -1] + u[0, -2])
            u[-1, -1] = 0.5*(u[-1, -2] + u[-2, -1])
            
            
        def set_vbc(v, args=None):
            """
                reference: https://empslocal.ex.ac.uk/people/staff/gv219/codes/linearshallowwater.py
            """
            #free-slip west/east
            v[0, :] = v[1, :]
            v[-1, :] = v[-2, :]
            
            #free slip at north and south boundary
            v[:, 0] = v[:, 1]
            v[:, -1] = v[:, -2]
            
            
            # fix corners to be average of neighbors, this is requirbed because corners are
            # east/west, south/north boundary at the same time, difference bc leads to vastly different values 
            # can cause artifical large gradient -> numerical instability
            v[0, 0] =  0.5*(v[1, 0] + v[0, 1])
            v[-1, 0] = 0.5*(v[-2, 0] + v[-1, 1])
            v[0, -1] = 0.5*(v[1, -1] + v[0, -2])
            v[-1, -1] = 0.5*(v[-1, -2] + v[-2, -1])
        
        # self.boundary_conditions = {
        #     'h' : {"derivative" : 0.0},
        #     'u' : [{"value" : 0.0},{"derivative" : 0.0}],
        #     'v' : [{"derivative" : 0.0},{"derivative" : 0.0}]
        # } 
        self.boundary_conditions = {
            'h' : set_hbc,
            'u' : set_ubc,
            'v' : set_vbc
        } 
        
        # 4. Set up forcing
        # wind forcing by trade forcing
        self.tau_trade_forcing = (VectorField.from_expression(grid, expressions=trade_wind_stress_expressions).data)
        # 5. Set up storage
        self.out_dir = out_dir
        self.storage = FileStorage(os.path.join(out_dir, "simulation.hdf5"))
        
        params = {
            "bcs" : self.boundary_conditions,
            "eps" : self.eps, 
            "gamma" : self.gamma, 
            "trade_wind_forcing" : self.tau_trade_forcing
        }
        
        self.eq = pde(**params)
        
        self.solver : str = solver
    
    def _plot_wind_forcing_video(self, out_dir: str, step: int = 4, fps: int = 10, coarsen_factor: int = 100):
        coords = self.state.grid.cell_coords
        X = coords[:, :, 0]
        Y = coords[:, :, 1]
        tau_y = self.tau_trade_forcing[1]

        times = list(self.storage.times)
        forcing_fields = []
        for i, t in enumerate(tqdm(times, desc="Loading wind forcing frames")):
            state = self.storage[i]
            h = state[2].data
            Nx = h.shape[0]
            h_mean_west = h[: Nx // 2, :].mean()
            h_mean_east = h[Nx // 2 : -1, :].mean()
            wind_x = self.gamma * (h_mean_east - h_mean_west) - self.tau_trade_forcing[0]
            forcing_fields.append(wind_x)

        times = times[::coarsen_factor]
        forcing_fields = forcing_fields[::coarsen_factor]

        fig, ax = plt.subplots(figsize=(10, 4))
        quiv = ax.quiver(X[::step, ::step], Y[::step, ::step],
                         forcing_fields[0][::step, ::step], tau_y[::step, ::step])
        title = ax.set_title(f"Wind Forcing  t = {times[0]:.2f}")
        ax.set_xlabel("x (non-dim)")
        ax.set_ylabel("y (non-dim)")
        fig.tight_layout()

        def update(i):
            quiv.set_UVC(forcing_fields[i][::step, ::step], tau_y[::step, ::step])
            title.set_text(f"Wind Forcing  t = {times[i]:.2f}")
            return quiv, title

        anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
        anim.save(os.path.join(out_dir, "wind_forcing.mp4"), writer="ffmpeg", fps=fps, dpi=150)
        plt.close(fig)

    def _plot_thermocline_video(self, out_dir: str, fps: int = 10, coarsen_factor: int = 2):
        coords = self.state.grid.cell_coords
        X = coords[:, :, 0]
        Y = coords[:, :, 1]

        times = list(self.storage.times)
        h_frames = []
        for i in range(len(times)):
            h_frames.append(self.storage[i][2].data)

        times = times[::coarsen_factor]
        h_frames = h_frames[::coarsen_factor]

        h_all = np.stack(h_frames)
        zlim = (h_all.min(), h_all.max())
        z_pad = 0.05 * (zlim[1] - zlim[0]) or 0.1
        zlim = (zlim[0] - z_pad, zlim[1] + z_pad)

        x_range = X.max() - X.min()
        y_range = Y.max() - Y.min()

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([x_range, y_range, y_range * 0.4], zoom=1.4)
        ax.view_init(elev=10, azim=35)

        surf = ax.plot_surface(X, Y, h_frames[0], cmap='RdBu_r',
                               vmin=zlim[0], vmax=zlim[1], linewidth=0, antialiased=False)
        
        title = ax.set_title(f"Thermocline depth  t = {times[0]:.2f}")
        ax.set_xlabel("x (non-dim)",labelpad=20)
        ax.set_ylabel("y (non-dim)",labelpad=20)
        ax.set_zlabel("h (non-dim)",labelpad=1)
        
        ax.set_zlim(*zlim)
        # fig.colorbar(surf, ax=ax, shrink=0.15, pad=0.15, location='right')
        fig.tight_layout()

        pbar = tqdm(total=(len(times) + 1)//coarsen_factor, desc='Animating Thermocline Depth')
        
        def update(i):
            pbar.update(1)
            nonlocal surf
            surf.remove()
            surf = ax.plot_surface(X, Y, h_frames[i], cmap='RdBu_r',
                                   vmin=zlim[0], vmax=zlim[1], linewidth=0, antialiased=False)
            title.set_text(f"Thermocline depth  t = {times[i]:.2f}")
            return surf,

        anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
        anim.save(os.path.join(out_dir, "thermocline.mp4"), writer="ffmpeg", fps=fps, dpi=150)
        plt.close(fig)

    def run_one_simulation(self, t_range, dt):
        self.eq.solve(self.state, t_range=t_range, dt=dt, tracker=["progress", self.storage.tracker(1)], solver=self.solver, backend='numpy')
        # movie(self.storage, os.path.join(self.out_dir, "simulation.mp4"))
        # self._plot_wind_forcing_video(self.out_dir)
        # self._plot_thermocline_video(self.out_dir)
        
        

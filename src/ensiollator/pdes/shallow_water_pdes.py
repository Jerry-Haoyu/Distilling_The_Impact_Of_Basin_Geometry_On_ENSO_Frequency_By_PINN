from collections.abc import Callable
from typing import Callable

import numpy as np
import numpy.typing as npt

import pde
from pde import PDEBase
from pde import BoundariesBase
from pde import CartesianGrid, FileStorage, ScalarField, FieldCollection
from pde.visualization.movies import movie


class NaiveShallowWaterPDE(PDEBase):
    def __init__(self, 
               bcs : dict[str, BoundariesBase], 
               eps, 
               gamma,
               trade_wind_forcing : npt.NDArray
               ): 
        """
            Naive PDE  for linearized shallow water equation 
            (1) No compilation, using numpy for backend only 
            (2) Simplified air-sea coupling, use:
                forcing_x = gamma * (h_mean_east - h_mean_west) - trade_wind_forcing
                for basic Bejerkins feedback 
        """
        super().__init__()
        self.bcs = bcs
        self.eps = eps
        self.gamma = gamma
        self.trade_wind_forcing = trade_wind_forcing
        self.wind_forcing_history: list[tuple[float, npt.NDArray, npt.NDArray]] = []
      
    def compute_wind_forcing_x(self, state):
        h_field = state[2].data
        X, Y = h_field.shape
        h_mean_west = np.mean(h_field[0 : int(X * 0.1), :])
        h_mean_east = np.mean(h_field[int(X * 0.9) :, :])
        return self.gamma * (h_mean_east - h_mean_west) - self.trade_wind_forcing[0]

    def evolution_rate(self, state, t=0):
        wind_forcing_x_field = self.compute_wind_forcing_x(state)
        wind_forcing_y_field = self.trade_wind_forcing[1]

        self.wind_forcing_history.append((float(t), wind_forcing_x_field, wind_forcing_y_field))
        
        u, v, h = state
        
        grad_h = h.gradient(bc=self.bcs['h'])
        
        # grid value 
        y =  state.grid.cell_coords[:, :, 1]
        # gradient of h w.r.p to x
        h_x = grad_h[0]
        # gradient of h w.r.p to y
        h_y = grad_h[1]
        
        # the R.H.S of the PDE
        u_t =  y * v  - h_x - self.eps * u + wind_forcing_x_field
        v_t = -y * u  - h_y - self.eps * v + wind_forcing_y_field
        
        grad_u = u.gradient(bc=self.bcs['u'])
        grad_v = v.gradient(bc=self.bcs['v'])
        
        u_x = grad_u[0]
        v_y = grad_v[1]
        
        # divergence(\vec{u})
        h_t = - (u_x + v_y)
        
        return FieldCollection([u_t, v_t, h_t])
    
    #does not work, for some unknown reason
    def make_pde_rhs_numba(self, state):
        from pde.backends.numba.utils import jit

        numba_backend = pde.backends['numba']

        grad_u = state.grid.make_operator('gradient', bc=self.bcs['u'], backend=numba_backend, dtype=state.dtype)
        grad_v = state.grid.make_operator('gradient', bc=self.bcs['v'], backend=numba_backend, dtype=state.dtype)
        grad_h = state.grid.make_operator('gradient', bc=self.bcs['h'], backend=numba_backend, dtype=state.dtype)

        y_coords = state.grid.cell_coords[:, :, 1].copy()
        eps = float(self.eps)

        @jit
        def pde_rhs(state_data, t=0):
            u = state_data[0]
            v = state_data[1]
            h = state_data[2]

            dh = grad_h(h)
            h_x = dh[0]
            h_y = dh[1]

            u_t =  y_coords * v - h_x - eps * u
            v_t = -y_coords * u - h_y - eps * v

            du = grad_u(u)
            dv = grad_v(v)

            h_t = -(du[0] + dv[1])

            out = np.empty_like(state_data)
            out[0] = u_t
            out[1] = v_t
            out[2] = h_t
            return out

        return pde_rhs
            





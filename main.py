import numpy as np
import pde
from pde import CartesianGrid, DiffusionPDE, FileStorage, ScalarField
from pde.visualization.movies import movie

from src.ensiollator.pdes.simulator import ShallowWaterSimulator
from src.ensiollator.visualize.thermocline import plot_thermocline_video, plot_thermocline_timeseries

def simulate():
    params = {
        "initial_height_expression" : "exp(- (0.1 * y ** 2)) - 0.06 * x + 1.5",
        "resolution" : 64, 
        #["0.5 * cos(pi * y / 17)", "0.0"]
        "trade_wind_stress_expressions" : ["0.0","0.0"], #already none-dimensionalized
        "out_dir": "data/naiveSW",
        "rayleigh_coefficient":1e-7 ,
        "air_sea_coupling_strength": 0.0,
        "solver": 'crank-nicolson'
    }

    model = ShallowWaterSimulator(**params)

    model.run_one_simulation(t_range=3200, dt=1e-1)
   
def plot():
    plot_thermocline_video(
        data_dir="data/naiveSW",
        out_path="data/naiveSW/thermocline.mp4",
        coarsen_factor=30,
        steps=list(range(1,3200))
    )
    plot_thermocline_timeseries(data_dir="data/naiveSW",
        out_path="data/naiveSW/thermocline_timeseries"
    )
        
    
def main():
    simulate()
    plot()
   

if __name__ == "__main__":
    main()

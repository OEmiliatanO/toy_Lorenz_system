import numpy as np
import xarray as xr

def lorenz_system(state, sigma=10.0, r=28.0, b=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = r * x - y - x * z
    dz = x * y - b * z
    return np.array([dx, dy, dz])

def rk4_step(f, state, dt, **kwargs):
    k1 = f(state, **kwargs)
    k2 = f(state + 0.5*dt*k1, **kwargs)
    k3 = f(state + 0.5*dt*k2, **kwargs)
    k4 = f(state + dt*k3, **kwargs)
    return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def generate_lorenz_data(T=2000, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
    state = np.array([1.0, 1.0, 1.0])
    trajectory = np.zeros((T, 3))
    for t in range(T):
        trajectory[t] = state
        state = rk4_step(lorenz_system, state, dt, sigma=sigma, rho=rho, beta=beta)

    # 存成 xarray dataset
    ds = xr.Dataset(
        {"state": (("time", "dim"), trajectory)},
        coords={"time": np.arange(T)*dt, "dim": ["x", "y", "z"]}
    )
    
    return ds

R_values = [5, 10, 20, 28, 40, 50, 100]
for r in R_values:
    ds = generate_lorenz_data(T=2000, dt=0.01, r=r)
    save_path = f"lorenz_data_r={r}.nc"
    ds.to_netcdf(save_path)
    print(f"Saved dataset to {save_path}")
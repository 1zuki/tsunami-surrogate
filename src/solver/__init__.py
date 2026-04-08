from src.solver.shallow_water import ShallowWaterSolver
from src.solver.boussinesq import WeaklyDispersiveSolver


def build_solver(config: dict):
    sim_cfg = config.get("simulation", {})
    solver_name = str(sim_cfg.get("solver", "shallow_water")).lower()
    if solver_name in {"boussinesq", "weakly_dispersive", "dispersive"}:
        return WeaklyDispersiveSolver.from_config(config)
    return ShallowWaterSolver.from_config(config)

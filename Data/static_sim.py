import os
import sys
import argparse
import numpy as np
import pandas as pd
from diffeqpy import de
from mpi4py import MPI
import time

# Prevent MPI worker conflicts
os.environ["JAX_PLATFORM_NAME"] = "cpu"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# model argument (defaults to M3 if none provided)
parser = argparse.ArgumentParser(description="Distributed CRN Simulator")
parser.add_argument("--model", type=str, choices=["M1", "M2", "M3", "M5"], default="M3", 
                    help="Choose the mechanism to simulate (M1, M2, M3, or M5)")
args, unknown = parser.parse_known_args()
TARGET_MODEL = args.model

# Random seed based on job id
slurm_job_id = int(os.environ.get("SLURM_JOB_ID", 1000))
np.random.seed((slurm_job_id % 100000) * 1000 + rank)

TOTAL_SAMPLES = 234000

# Folder Creation based on the selected model
DATA_DIR = f"{TARGET_MODEL}_training_data" # change to testing for testing data
if rank == 0:
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Targeting Model: {TARGET_MODEL}. Saving chunks to: {DATA_DIR}/")
    
comm.Barrier()

# Species and prior bounds
species = ['S', 'P', 'cat', 'catS', 'cat_2', 'cat_2S', 'catP']
LOG_KMIN = np.log(1e-5)
LOG_KMAX = np.log(10e5)
N_OBS = 4

# Increased to 21 to accommodate the t=0 step
N_TIMES = 21

class BaseModel:
    def __init__(self, k, c):
        self.species_names = list(c.keys())
        self.species_index = {name: i for i, name in enumerate(self.species_names)}
        self.k = k
        self.x0 = np.array([c[name] for name in self.species_names], dtype=np.float64)

    def indexer(self, x):
        return {name: x[idx] for name, idx in self.species_index.items()}

    def dxdt(self, t, x):
        raise NotImplementedError

    def jacobian(self, t, x):
        raise NotImplementedError

    def validate_simulation(self, sim):
        """Universal checks applied to ALL models"""
        if sim['S'].values[-1] > 0.95: return False # Reject flatlines
        if sim['S'].values[3] < 0.2: return False   # Reject instant drops
        return True


class M1(BaseModel):
    def dxdt(self, t, x):
        c = self.indexer(x)
        k = self.k
        dx = np.zeros(len(x)) 
        dx[species.index("S")] = k['-1'] * c['catS'] - k['1'] * c['S'] * c['cat']
        dx[species.index("P")] = k['2'] * c['catS'] - k['-2'] * c['cat'] * c['P']
        dx[species.index("cat")] = (k['-1'] + k['2']) * c['catS'] - (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat']
        dx[species.index("catS")] = (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat'] - (k['-1'] + k['2']) * c['catS']
        return dx

    def jacobian(self, t, x):
        c = self.indexer(x)
        k = self.k
        n = len(x)
        J = np.zeros((n, n))
        idx = {sp: species.index(sp) for sp in species}
        J[idx["S"], idx["S"]]       = -k['1'] * c['cat']
        J[idx["S"], idx["cat"]]     = -k['1'] * c['S']
        J[idx["S"], idx["catS"]]    = k['-1']
        J[idx["P"], idx["P"]]       = -k['-2'] * c['cat']
        J[idx["P"], idx["cat"]]     = -k['-2'] * c['P']
        J[idx["P"], idx["catS"]]    = k['2']
        J[idx["cat"], idx["S"]]     = -k['1'] * c['cat'] 
        J[idx["cat"], idx["P"]]     = -k['-2'] * c['cat']
        J[idx["cat"], idx["cat"]]   = -k['1'] * c['S'] - k['-2'] * c['P']
        J[idx["cat"], idx["catS"]]  = k['-1'] + k['2']
        J[idx["catS"], idx["S"]]    = k['1'] * c['cat']
        J[idx["catS"], idx["P"]]    = k['-2'] * c['cat']
        J[idx["catS"], idx["cat"]]  = k['1'] * c['S'] + k['-2'] * c['P']
        J[idx["catS"], idx["catS"]] = -k['-1'] - k['2']
        return J

    def validate_simulation(self, sim):
        """M1 ONLY uses the universal flatline/instant checks"""
        return super().validate_simulation(sim)


class M2(BaseModel):
    def dxdt(self, t, x):
        c = self.indexer(x)
        k = self.k
        dx = np.zeros(len(x)) 
        
        dx[species.index("S")] = k['-1'] * c['catS'] - k['1'] * c['S'] * c['cat']
        dx[species.index("P")] = k['2'] * c['catS'] - k['-2'] * c['cat'] * c['P']
        dx[species.index("cat")] = (k['-1'] + k['2']) * c['catS'] + 2 * k['3'] * c['cat_2'] - (k['1'] * c['S'] + k['-2'] * c['P'] + 2 * k['-3'] * c['cat']) * c['cat']
        dx[species.index("catS")] = (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat'] - (k['-1'] + k['2']) * c['catS']
        dx[species.index("cat_2")] = k['-3'] * c['cat']**2 - k['3'] * c['cat_2']
        
        return dx

    def jacobian(self, t, x):
        c = self.indexer(x)
        k = self.k
        n = len(x)
        J = np.zeros((n, n))
        
        idx = {sp: species.index(sp) for sp in species}

        J[idx["S"], idx["S"]]       = -k['1'] * c['cat']
        J[idx["S"], idx["cat"]]     = -k['1'] * c['S']
        J[idx["S"], idx["catS"]]    = k['-1']

        J[idx["P"], idx["P"]]       = -k['-2'] * c['cat']
        J[idx["P"], idx["cat"]]     = -k['-2'] * c['P']
        J[idx["P"], idx["catS"]]    = k['2']

        J[idx["cat"], idx["S"]]     = -k['1'] * c['cat'] 
        J[idx["cat"], idx["P"]]     = -k['-2'] * c['cat']
        J[idx["cat"], idx["cat"]]   = -k['1'] * c['S'] - k['-2'] * c['P'] - 4 * k['-3'] * c['cat']
        J[idx["cat"], idx["catS"]]  = k['-1'] + k['2']
        J[idx["cat"], idx["cat_2"]] = 2 * k['3']

        J[idx["catS"], idx["S"]]    = k['1'] * c['cat']
        J[idx["catS"], idx["P"]]    = k['-2'] * c['cat']
        J[idx["catS"], idx["cat"]]  = k['1'] * c['S'] + k['-2'] * c['P']
        J[idx["catS"], idx["catS"]] = -k['-1'] - k['2']

        J[idx["cat_2"], idx["cat"]] = 2 * k['-3'] * c['cat']
        J[idx["cat_2"], idx["cat_2"]] = -k['3']

        return J

    def validate_simulation(self, sim):
        # Run the universal checks first
        if not super().validate_simulation(sim):
            return False
            
        # Run M2 specific checks
        S_vals = sim['S'].values
        # Isolate the window where conversion is between 20% and 50%
        window_mask = (S_vals <= 0.8) & (S_vals >= 0.5)
        
        # If it jumped over this window (e.g., too fast), reject it
        if not np.any(window_mask):
            return False
            
        # Check if cat_2 is SUSTAINED > 10% of total catalyst (0.005) for the whole window
        cat2_window = sim['cat_2'].values[window_mask]
        return np.all(cat2_window > 0.005)


class M3(BaseModel):
    def dxdt(self, t, x):
        c  = self.indexer(x)
        k  = self.k
        dx = np.zeros(len(x))
        dx[species.index("S")]      = k['-4'] * c['cat_2S'] - k['4'] * c['S'] * c['cat_2']
        dx[species.index("P")]      = k['5']  * c['cat_2S'] - k['-5'] * c['cat_2'] * c['P']
        dx[species.index("cat_2")]  = (k['-4'] + k['5']) * c['cat_2S'] + k['-3'] * c['cat']**2 - (k['4'] * c['S'] + k['-5'] * c['P'] + k['3']) * c['cat_2']
        dx[species.index("cat_2S")] = (k['4'] * c['S'] + k['-5'] * c['P']) * c['cat_2'] - (k['-4'] + k['5']) * c['cat_2S']
        dx[species.index("cat")]    = 2 * k['3'] * c['cat_2'] - 2 * k['-3'] * c['cat']**2
        return dx

    def jacobian(self, t, x):
        c = self.indexer(x)
        k = self.k
        n = len(x)
        J = np.zeros((n, n))
        idx = {sp: species.index(sp) for sp in species}
        J[idx["S"],      idx["S"]]      = -k['4'] * c['cat_2']
        J[idx["S"],      idx["cat_2"]]  = -k['4'] * c['S']
        J[idx["S"],      idx["cat_2S"]] =  k['-4']
        J[idx["P"],      idx["P"]]      = -k['-5'] * c['cat_2']
        J[idx["P"],      idx["cat_2"]]  = -k['-5'] * c['P']
        J[idx["P"],      idx["cat_2S"]] =  k['5']
        J[idx["cat"],    idx["cat"]]    = -4 * k['-3'] * c['cat']
        J[idx["cat"],    idx["cat_2"]]  =  2 * k['3']
        J[idx["cat_2S"], idx["S"]]      =  k['4']  * c['cat_2']
        J[idx["cat_2S"], idx["P"]]      =  k['-5'] * c['cat_2']
        J[idx["cat_2S"], idx["cat_2"]]  =  k['4']  * c['S'] + k['-5'] * c['P']
        J[idx["cat_2S"], idx["cat_2S"]] = -k['-4'] - k['5']
        J[idx["cat_2"],  idx["S"]]      = -k['4']  * c['cat_2']
        J[idx["cat_2"],  idx["P"]]      = -k['-5'] * c['cat_2']
        J[idx["cat_2"],  idx["cat"]]    =  2 * k['-3'] * c['cat']
        J[idx["cat_2"],  idx["cat_2"]]  = -k['4']  * c['S'] - k['-5'] * c['P'] - k['3']
        J[idx["cat_2"],  idx["cat_2S"]] =  k['-4'] + k['5']
        return J

    def validate_simulation(self, sim):
        # Run the universal checks first
        if not super().validate_simulation(sim):
            return False
            
        # Run M3 specific checks
        early_mask = sim['S'].values > 0.9
        if not np.any(early_mask):
            return False  
        cat_early = sim['cat'].values[early_mask]
        return np.all(cat_early > 0.0025)

class M5(BaseModel):
    def dxdt(self, t, x):
        c = self.indexer(x)
        k = self.k
        dx = np.zeros(len(x))
        
        dx[species.index("S")] = k['-1'] * c['catS'] - k['1'] * c['S'] * c['cat']
        dx[species.index("P")] = k['3'] * c['catP'] - k['-3'] * c['cat'] * c['P']
        dx[species.index("cat")] = k['-1'] * c['catS'] + k['3'] * c['catP'] - (k['1'] * c['S'] + k['-3'] * c['P']) * c['cat']
        dx[species.index("catS")] = (k['1'] * c['S'] + k['-2'] * c['catP']) * c['cat'] - (k['-1'] + k['2'] * c['cat']) * c['catS']
        dx[species.index("catP")] = k['2'] * c['catS'] * c['cat'] + k['-3'] * c['P'] * c['cat'] - (k['3'] + k['-2'] * c['cat']) * c['catP']
        
        return dx

    def jacobian(self, t, x):
        c = self.indexer(x)
        k = self.k
        n = len(x)
        J = np.zeros((n, n))
        
        idx = {sp: species.index(sp) for sp in species}

        J[idx["S"], idx["S"]]       = -k['1'] * c['cat']
        J[idx["S"], idx["cat"]]     = -k['1'] * c['S']
        J[idx["S"], idx["catS"]]    = k['-1']

        J[idx["P"], idx["P"]]       = -k['-3'] * c['cat']
        J[idx["P"], idx["cat"]]     = -k['-3'] * c['P']
        J[idx["P"], idx["catP"]]    = k['3']

        J[idx["cat"], idx["S"]]     = -k['1'] * c['cat']
        J[idx["cat"], idx["P"]]     = -k['-3'] * c['cat']
        J[idx["cat"], idx["cat"]]   = -k['1'] * c['S'] - k['-3'] * c['P']
        J[idx["cat"], idx["catS"]]  = k['-1']
        J[idx["cat"], idx["catP"]]  = k['3']

        J[idx["catS"], idx["S"]]    = k['1'] * c['cat']
        J[idx["catS"], idx["cat"]]  = k['1'] * c['S'] + k['-2'] * c['catP'] - k['2'] * c['catS']
        J[idx["catS"], idx["catS"]] = -k['-1'] - k['2'] * c['cat']
        J[idx["catS"], idx["catP"]] = k['-2'] * c['cat']

        J[idx["catP"], idx["P"]]    = k['-3'] * c['cat']
        J[idx["catP"], idx["cat"]]  = k['2'] * c['catS'] + k['-3'] * c['P'] - k['-2'] * c['catP']
        J[idx["catP"], idx["catS"]] = k['2'] * c['cat']
        J[idx["catP"], idx["catP"]] = -k['3'] - k['-2'] * c['cat']

        return J

    def validate_simulation(self, sim):
        """M5 uses the universal flatline/instant checks"""
        return super().validate_simulation(sim)


# Models and params 
MODELS_CONFIG = {
    "M1": {"class": M1, "keys": ['1', '-1', '2', '-2']},
    "M2": {"class": M2, "keys": ['1', '-1', '2', '-2', '3', '-3']},
    "M3": {"class": M3, "keys": ['3', '-3', '4', '-4', '5', '-5']}, #[-3, 3, 1, -1, 2, -2] in the paper
    "M5": {"class": M5, "keys": ['1', '-1', '2', '-2', '3', '-3']}, 
}

MODEL_CLASS = MODELS_CONFIG[TARGET_MODEL]["class"]
K_KEYS = MODELS_CONFIG[TARGET_MODEL]["keys"]
NUM_K = len(K_KEYS)

def make_diffeq_rhs(model):
    def f(du, u, p, t):
        du[:] = model.dxdt(t, np.asarray(u))
    return f

def make_diffeq_jac(model):
    def jac(J, u, p, t):
        J[:, :] = model.jacobian(t, np.asarray(u))
    return jac

def simulate_model_diffeq(k_dict, c_ini, t_obs, rtol=1e-6, atol=1e-9):
    # Dynamically inject the chosen model
    model  = MODEL_CLASS(k_dict, c_ini)
    tspan  = (0.0, 1.0)
    trep   = 0.25e-5
    t_eval = np.arange(0.0, 1.0, trep)

    f   = de.ODEFunction(make_diffeq_rhs(model), jac=make_diffeq_jac(model))
    try:
        prob = de.ODEProblem(f, model.x0, tspan)
        # maxiters=10000 to prevent from hanging
        sol  = de.solve(prob, de.Rodas5(),
                        saveat=t_eval, reltol=rtol, abstol=atol,
                        save_everystep=False, dense=False, maxiters=10000)
    except Exception:
        return None

    if sol.retcode != de.ReturnCode.Success:
        return None

    Y  = np.stack([np.asarray(ui) for ui in sol.u], axis=0)
    t  = np.asarray(sol.t)
    df = pd.DataFrame(Y, columns=model.species_names)
    
    df.insert(0, 'time', t.astype(np.float64))
    time_df = pd.DataFrame({'time': np.asarray(t_obs, dtype=np.float64)})
    merged  = pd.merge_asof(time_df, df, on='time', direction='backward')
    return merged

def generate_single_sim():
    max_attempts = 10000
    t_check = np.linspace(0.0, 1.0, 200)

    for _ in range(max_attempts):
        # Draw parameters
        log_k = np.random.uniform(LOG_KMIN, LOG_KMAX, size = NUM_K)
        k_dict = dict(zip(K_KEYS, np.exp(log_k)))

        # Run the standard reference check 
        c_ref = {'S': 1.0, 'P': 0.0, 'cat': 0.05, 'catS': 0.0, 'cat_2': 0.0, 'cat_2S': 0.0, 'catP': 0.0}
        model_validator = MODEL_CLASS(k_dict, c_ref)
        
        f = de.ODEFunction(make_diffeq_rhs(model_validator), jac=make_diffeq_jac(model_validator))
        try:
            prob = de.ODEProblem(f, model_validator.x0, (0.0, 1.0))
            sol = de.solve(prob, de.Rodas5(), saveat=t_check, reltol=1e-6, abstol=1e-9, save_everystep=False, dense=False, maxiters=10000)
            if sol.retcode != de.ReturnCode.Success: continue
            
            ref_sim_df = pd.DataFrame(np.stack([np.asarray(ui) for ui in sol.u], axis=0), columns = model_validator.species_names)
            ref_sim_df.insert(0, 'time', np.asarray(sol.t).astype(np.float64))
        except Exception:
            continue

        if not model_validator.validate_simulation(ref_sim_df):
            continue # Failed physical chemistry checks. Throw away and restart.

        # initial conditions context
        cat0 = np.random.uniform(0.01, 0.10, size=3)   
        cat0_4_idx = np.random.randint(0, 3)             
        S0_4 = np.random.uniform(0.4, 0.8)
        P0_4 = 1.0 - S0_4
        
        # Generate 20 random points, sort them, start with t=0.0
        random_t = np.random.uniform(0.001, 1.0, size=(N_OBS, N_TIMES - 1))
        random_t = np.sort(random_t, axis=1)
        zero_t = np.zeros((N_OBS, 1))
        t_obs = np.concatenate([zero_t, random_t], axis=1)

        c_ini_list = [
            {'S': 1.0,  'P': 0.0,  'cat': float(cat0[0]), 'catS': 0., 'cat_2': 0., 'cat_2S': 0., 'catP': 0.0},
            {'S': 1.0,  'P': 0.0,  'cat': float(cat0[1]), 'catS': 0., 'cat_2': 0., 'cat_2S': 0., 'catP': 0.0},
            {'S': 1.0,  'P': 0.0,  'cat': float(cat0[2]), 'catS': 0., 'cat_2': 0., 'cat_2S': 0., 'catP': 0.0},
            {'S': S0_4, 'P': P0_4, 'cat': float(cat0[cat0_4_idx]), 'catS': 0., 'cat_2': 0., 'cat_2S': 0., 'catP': 0.0},
        ]

        obs_data = np.zeros((N_OBS, N_TIMES, 2), dtype=np.float32)
        valid_training_data = True

        # Simulate all four experiments 
        for i in range(N_OBS):
            sim_df = simulate_model_diffeq(k_dict, c_ini_list[i], t_obs[i])
            if sim_df is None:
                valid_training_data = False
                break
                
            obs_data[i, :, 0] = sim_df['S'].values
            obs_data[i, :, 1] = sim_df['P'].values

        if not valid_training_data:
            continue

        # Measurment noise
        log_sigma_S = np.random.normal(np.log(0.00248), 1.0)
        log_sigma_P = np.random.normal(np.log(0.00248), 1.0)
        sigma_S = np.exp(log_sigma_S)
        sigma_P = np.exp(log_sigma_P)

        obs_data[:, :, 0] += np.random.normal(0, sigma_S, size=(N_OBS, N_TIMES))
        obs_data[:, :, 1] += np.random.normal(0, sigma_P, size=(N_OBS, N_TIMES))
        obs_data = np.clip(obs_data, 0.0, 1.0)

        theta = np.array([*log_k, log_sigma_S, log_sigma_P], dtype=np.float32)
        cat0_all = np.array([cat0[0], cat0[1], cat0[2], cat0[cat0_4_idx]], dtype=np.float32)
        
        context = {
            'cat0':        cat0.astype(np.float32),        
            'cat0_all':    cat0_all,                       
            'cat0_4_idx':  np.int32(cat0_4_idx),            
            'S0_4':        np.float32(S0_4),                
            'P0_4':        np.float32(P0_4),                
            't_obs':       t_obs.astype(np.float32),        
        }
        
        return {
            'prior_draws': {'theta': theta, 'context': context},
            'sim_data': {'data': obs_data}
        }
        
    raise RuntimeError("Failed to find valid parameters after max attempts")


# Distribute training data generation across multiple cores
if rank == 0:
    print(f"Initializing Distributed Data Generation for {TOTAL_SAMPLES} total samples across {size} CPU cores...")

base = TOTAL_SAMPLES // size
extra = TOTAL_SAMPLES % size
local_batch = base + (1 if rank < extra else 0)

print(f"Job {slurm_job_id} Rank {rank} starting generation of {local_batch} samples...")
start_time = time.time()

local_results = []
for i in range(local_batch):
    local_results.append(generate_single_sim())
    
    if rank == 0 and (i + 1) % 1000 == 0:
        print(f"Job {slurm_job_id} Rank 0 Progress: {i + 1} / {local_batch} completed...")

batch_data = [sim['sim_data']['data'] for sim in local_results]
batch_t_obs = [sim['prior_draws']['context']['t_obs'] for sim in local_results]
batch_cat0 = [sim['prior_draws']['context']['cat0_all'] for sim in local_results]
batch_theta = [sim['prior_draws']['theta'] for sim in local_results]

# Extract S0_4 so the initial conditions are not lost
batch_S0_4 = [sim['prior_draws']['context']['S0_4'] for sim in local_results]

chunk_filename = os.path.join(DATA_DIR, f"chunk_job{slurm_job_id}_rank{rank}.npz")
np.savez_compressed(
    chunk_filename,
    data=np.stack(batch_data, axis=0),
    t_obs=np.stack(batch_t_obs, axis=0),
    cat0=np.stack(batch_cat0, axis=0),
    theta=np.stack(batch_theta, axis=0),
    S0_4=np.stack(batch_S0_4, axis=0) # Saved to disk 
)

elapsed = time.time() - start_time
print(f"Job {slurm_job_id} Rank {rank} successfully saved {chunk_filename} in {elapsed:.2f} seconds!")

comm.Barrier()

if rank == 0:
    print(f"All {size} chunks have been safely written to disk for Job {slurm_job_id}. Generation complete!")
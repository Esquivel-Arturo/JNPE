""" This script is used to implement SMC-ABC on Mechanisms 1,2 and 3. It is built to run in parallel. The samples are simulated across multiple cores at each generation.   
    At the end of each generation all accepted samples are combined to be used in the proposals for the next generation. """

import numpy as np
import pandas as pd
from scipy.stats import uniform, multivariate_normal, norm
from scipy.special import logsumexp
import math
import pickle
import time 
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from diffeqpy import de
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

# Simulators - ODEs 
species = ['S', 'P', 'cat', 'catS', 'cat_2', 'cat_2S']

class BaseModel:
    def __init__(self, k, c):
        self.species_names = list(c.keys())
        self.species_index = {name: i for i, name in enumerate(self.species_names)}
        self.k = k
        self.x0 = np.array([c[name] for name in self.species_names])
    def indexer(self, x): return {name: x[idx] for name, idx in self.species_index.items()}
    def dxdt(self, t, x): raise NotImplementedError
    def jacobian(self, t, x): raise NotImplementedError

# Mechanism 1
class M1(BaseModel):
    def dxdt(self, t, x):
        c, k = self.indexer(x), self.k
        dx = np.zeros(len(x)) 
        dx[species.index("S")] = k['-1'] * c['catS'] - k['1'] * c['S'] * c['cat']
        dx[species.index("P")] = k['2'] * c['catS'] - k['-2'] * c['cat'] * c['P']
        dx[species.index("cat")] = (k['-1'] + k['2']) * c['catS'] - (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat']
        dx[species.index("catS")] = (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat'] - (k['-1'] + k['2']) * c['catS']
        return dx
    def jacobian(self, t, x):
        c, k, n = self.indexer(x), self.k, len(x)
        J, idx = np.zeros((n, n)), {sp: species.index(sp) for sp in species}
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

# Mechanism 2
class M2(BaseModel):
    def dxdt(self, t, x):
        c, k = self.indexer(x), self.k
        dx = np.zeros(len(x)) 
        dx[species.index("S")] = k['-1'] * c['catS'] - k['1'] * c['S'] * c['cat']
        dx[species.index("P")] = k['2'] * c['catS'] - k['-2'] * c['cat'] * c['P']
        dx[species.index("cat")] = (k['-1'] + k['2']) * c['catS'] + 2 * k['3'] * c['cat_2'] - (k['1'] * c['S'] + k['-2'] * c['P'] + 2 * k['-3'] * c['cat']) * c['cat']
        dx[species.index("catS")] = (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat'] - (k['-1'] + k['2']) * c['catS']
        dx[species.index("cat_2")] = k['-3'] * c['cat']**2 - k['3'] * c['cat_2']
        return dx
    def jacobian(self, t, x):
        c, k, n = self.indexer(x), self.k, len(x)
        J, idx = np.zeros((n, n)), {sp: species.index(sp) for sp in species}
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
    
class M3(BaseModel):
    def dxdt(self, t, x):
        c, k = self.indexer(x), self.k
        dx = np.zeros(len(x)) 
        dx[species.index("S")] = k['-4'] * c['cat_2S'] - k['4'] * c['S'] * c['cat_2']
        dx[species.index("P")] = k['5'] * c['cat_2S'] - k['-5'] * c['cat_2'] * c['P']
        dx[species.index("cat_2")] = (k['-4'] + k['5']) * c['cat_2S'] + k['-3'] * c['cat']**2 - (k['4'] * c['S'] + k['-5'] * c['P'] + k['3']) * c['cat_2'] 
        dx[species.index("cat_2S")] = (k['4'] * c['S'] + k['-5'] * c['P']) * c['cat_2'] - (k['-4'] + k['5']) * c['cat_2S'] 
        dx[species.index("cat")] = 2 * k['3'] * c['cat_2'] - 2 * k['-3'] * c['cat']**2 
        return dx 
    def jacobian(self, t, x):
        c, k, n = self.indexer(x), self.k, len(x)
        J, idx = np.zeros((n, n)), {sp: species.index(sp) for sp in species}
        J[idx["S"], idx["S"]]       = -k['4'] * c['cat_2'] 
        J[idx["S"], idx["cat_2"]]   = -k['4'] * c['S']
        J[idx["S"], idx["cat_2S"]]  = k['-4']
        J[idx["P"], idx["P"]]       = -k['-5'] * c['cat_2']
        J[idx["P"], idx["cat_2"]]   = -k['-5'] * c['P']
        J[idx["P"], idx["cat_2S"]]  = k['5']
        J[idx["cat"], idx["cat"]]   = -4 * k['-3'] * c['cat']
        J[idx["cat"], idx["cat_2"]] = 2 * k['3']
        J[idx["cat_2S"], idx["S"]]  = k['4'] * c['cat_2']
        J[idx["cat_2S"], idx["P"]]  = k['-5'] * c['cat_2']
        J[idx["cat_2S"], idx["cat_2"]]  = k['4'] * c['S'] + k['-5'] * c['P']
        J[idx["cat_2S"], idx["cat_2S"]] =  -k['-4'] - k['5']
        J[idx["cat_2"], idx["S"]]   = -k['4'] * c['cat_2']
        J[idx["cat_2"], idx["P"]]   = -k['-5'] * c['cat_2']
        J[idx["cat_2"], idx["cat"]] = 2 * k['-3'] * c['cat']
        J[idx["cat_2"], idx["cat_2"]] = -k['4'] * c['S'] - k['-5'] * c['P'] - k['3']
        J[idx["cat_2"], idx["cat_2S"]] =  k['-4'] + k['5']
        return J

model_classes = {0: M1, 1: M2, 2: M3}

# Explicit Keys with Measurement Noise Parameters for each model
M1_keys = ['1', '-1', '2', '-2', 'noise_S', 'noise_P']
M2_keys = ['1', '-1', '2', '-2', '3', '-3', 'noise_S', 'noise_P']
M3_keys = ['3', '-3', '4', '-4', '5', '-5', 'noise_S', 'noise_P'] # [-3, 3, 1, -1, 2, -2, 'noise_S', 'noise_P'] in the paper
model_keys = {0: M1_keys, 1: M2_keys, 2: M3_keys}

def make_diffeq_rhs(model):
    def f(du, u, p, t): du[:] = model.dxdt(t, np.asarray(u))
    return f

def make_diffeq_jac(model):
    def jac(J, u, p, t): J[:, :] = model.jacobian(t, np.asarray(u))
    return jac

# Integrate the ODE
def simulate_model_diffeq(model, t_obs, trep=0.25e-5, tfin=1.0, rtol=1e-6, atol=1e-9):
    tspan = (0.0, tfin)
    t_eval = np.arange(0.0, tfin, trep)
    f = de.ODEFunction(make_diffeq_rhs(model), jac=make_diffeq_jac(model))
    model.x0 = np.asarray(model.x0, dtype=np.float64)
    try:
        prob = de.ODEProblem(f, model.x0, tspan)
        sol = de.solve(prob, de.Rodas5(), saveat=t_eval, reltol=rtol, abstol=atol, save_everystep=False, dense=False)
    except Exception:
        return None
    if sol.retcode != de.ReturnCode.Success:
        return None

    Y = np.stack([np.asarray(ui) for ui in sol.u], axis=0)
    t = np.asarray(sol.t)
    df = pd.DataFrame(Y, columns=model.species_names)
    
    # Force float64 casting on both sides for Pandas merge
    df.insert(0, "time", t.astype(np.float64))
    time_df = pd.DataFrame({"time": np.asarray(t_obs, dtype=np.float64)})
    
    return pd.merge_asof(time_df, df, on="time", direction="backward")


# INDEPENDENT SMC-ABC (DIDELOT ET AL.)
class IndependentSMCABC:
    def __init__(self, model_name, model_class, param_keys, c_ini_list, t_obs_list, prior_bounds_log, x_df_list,
                 N=1800, max_retries=15000):
        self.model_name = model_name
        self.model_class = model_class
        self.param_keys = param_keys
        self.D = len(param_keys)
        self.c_ini_list = c_ini_list
        self.t_obs_list = t_obs_list
        self.n_obs = len(c_ini_list)
        self.kmin_log, self.kmax_log = prior_bounds_log
        self.N = N
        self.max_retries = max_retries        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_ranks = self.comm.Get_size()
        self.total_proposals = 0
        
        # Dynamically scale proposal covariance for the two different priors
        kinetic_width = self.kmax_log - self.kmin_log
        noise_width = 1.0 * 6  # Approx 99.7% of normal distribution mass
        widths = np.array([kinetic_width] * (self.D - 2) + [noise_width, noise_width])
        self.Sigma = np.diag((0.15 * widths)**2)
        
        self.summary_obs = self.summary_stat(x_df_list)

    # Parallelization
    def _distribute_work(self, N):
        base = N // self.n_ranks
        extra = N % self.n_ranks
        start = self.rank * base + min(self.rank, extra)
        end   = start + base + (1 if self.rank < extra else 0)
        return start, end   

    def sample_prior_log(self): 
        # Draw kinetics from Uniform and noise from Normal
        kinetics = np.random.uniform(self.kmin_log, self.kmax_log, size=self.D - 2)
        noises = np.random.normal(np.log(0.00001), 1.0, size=2)
        return np.concatenate([kinetics, noises])

    def prior_logpdf(self, logk): 
        # Evaluate log-probabilities independently and sum them
        lp_kinetics = np.sum(uniform.logpdf(logk[:-2], loc=self.kmin_log, scale=(self.kmax_log - self.kmin_log)))
        lp_noises = np.sum(norm.logpdf(logk[-2:], loc=np.log(0.00001), scale=1.0))
        return lp_kinetics + lp_noises

    # porposal kernel
    def K(self, logk_1): return np.random.multivariate_normal(mean=logk_1, cov=self.Sigma)

    # kernel pdf
    def K_logpdf(self, logk, logk_1):
        try: return multivariate_normal.logpdf(logk, mean=logk_1, cov=self.Sigma, allow_singular=True)
        except Exception: return -np.inf

    # Observations as summary statistics    
    def summary_stat(self, sim_data):
        raws = []
        for df in sim_data:
            raws.append(df['S'].values)
            raws.append(df['P'].values)
        return np.concatenate(raws)

    def compute_distance(self, summary_sim, summary_obs): 
        # Raw Euclidean distance
        return np.linalg.norm(summary_sim - summary_obs)

    # Run simulator
    def _simulate_all(self, log_params, rtol=1e-3, atol=1e-6):
        k_dict = dict(zip(self.param_keys[:-2], np.exp(np.clip(log_params[:-2], -100, 100))))
        
        # Unpack exact BayesFlow noise logic
        sigma_S = np.exp(log_params[-2])
        sigma_P = np.exp(log_params[-1])

        sims = []
        for c_ini, t_obs in zip(self.c_ini_list, self.t_obs_list):
            model = self.model_class(k_dict, c_ini)
            sim = simulate_model_diffeq(model, t_obs, rtol=rtol, atol=atol)
            if sim is None: return None
            
            # Inject micro-scale variance
            sim['S'] += np.random.normal(0, sigma_S, len(t_obs))
            sim['P'] += np.random.normal(0, sigma_P, len(t_obs))
            sims.append(sim)
        return sims

    # Basic rejection sampling for generation 0 
    def rejection_samp(self, initial_epsilon):
        start, end = self._distribute_work(self.N)
        n_local = end - start
        ks_local, rhos_local = [], []
        attempts = 0

        while len(ks_local) < n_local and attempts < self.max_retries:
            attempts += 1
            logk = self.sample_prior_log() 
            sims = self._simulate_all(logk, rtol=1e-3, atol=1e-6) 
            if sims is None: continue

            try: summary_sim = self.summary_stat(sims)
            except: continue

            rho = self.compute_distance(summary_sim, self.summary_obs)
            
            if rho <= initial_epsilon:
                ks_local.append(logk)
                rhos_local.append(rho)

            if self.rank == 0 and attempts % 100 == 0:
                print(f"\r{self.model_name} Rank 0 burn-in: {len(ks_local)}/{n_local} | Attempts: {attempts}", end="")
                sys.stdout.flush()

        all_ks = self.comm.gather(ks_local, root=0)
        all_rhos = self.comm.gather(rhos_local, root=0)
        total_attempts = self.comm.reduce(attempts, op=MPI.SUM, root=0)

        if self.rank == 0:
            ks_out = np.vstack(all_ks)
            rhos_out = np.concatenate(all_rhos)
        else:
            ks_out, rhos_out = None, None

        ks_out = self.comm.bcast(ks_out, root=0)
        rhos_out = self.comm.bcast(rhos_out, root=0)
        total_attempts = self.comm.bcast(total_attempts, root=0)
        return ks_out, rhos_out, total_attempts

    # Run SMC-ABC 
    def run(self, T=6, initial_epsilon=8.5):
        posterior_by_generation = []
        evidence_by_generation = []
        
        ks = np.zeros((self.N, self.D, T))
        W = np.zeros((self.N, T))

        if self.rank == 0: print(f"\n--- Starting {self.model_name} Independent SMC ---")
        ks[:, :, 0], acc_rhos, total_attempts = self.rejection_samp(initial_epsilon)
        W[:, 0] = 1.0 / self.N
        self.total_proposals += total_attempts
        
        Z_t = self.N / total_attempts
        evidence_by_generation.append(Z_t)
        
        # Save generation 0 uniformly weighted particles
        if self.rank == 0:
            print(
                f"\n{self.model_name} Gen 0 COMPLETED | Epsilon = {initial_epsilon} | Ev (Z_0) = {Z_t:.4e} Total Proposals: {total_attempts}")
            
            gen_dict = {k: ks[:, i, 0].tolist() for i, k in enumerate(self.param_keys)}
            gen_dict['weights'] = W[:, 0].tolist()
            posterior_by_generation.append(gen_dict)

        for t in range(1, T):
            if t < 2: rtol, atol = 1e-3, 1e-6
            elif t < 4: rtol, atol = 1e-4, 1e-7
            else: rtol, atol = 1e-6, 1e-9

            # Adaptive threshold
            epsilon = np.quantile(acc_rhos, 0.55)
            if self.rank == 0: print(f"\n{self.model_name} Gen {t} | Epsilon = {epsilon:.4f} | Current Ev = {Z_t:.4e}")

            start, end = self._distribute_work(self.N)
            ks_local = np.zeros((end - start, self.D))
            w_unnorm_local = np.zeros(end - start)
            rhos_local = []
            local_attempts = 0
            w_prev = W[:, t-1]

            # Main loop per generation
            for local_i in range(end - start):
                accepted = False
                while not accepted and local_attempts < self.max_retries:
                    local_attempts += 1
                    
                    j = np.random.choice(self.N, p=w_prev)
                    logk_1 = ks[j, :, t-1].copy()

                    logk_prop = self.K(logk_1)
                    if np.any(logk_prop[:-2] > self.kmax_log) or np.any(logk_prop[:-2] < self.kmin_log): continue

                    sims = self._simulate_all(logk_prop, rtol=rtol, atol=atol) 
                    if sims is None: continue

                    try: summary_sim = self.summary_stat(sims)
                    except: continue

                    rho = self.compute_distance(summary_sim, self.summary_obs)

                    if rho <= epsilon:
                        accepted = True
                        ks_local[local_i, :] = logk_prop
                        rhos_local.append(rho)
                        
                        if self.rank == 0:
                            print(f"\rGen {t} progress: {local_i + 1}/{end - start}", end="")
                            sys.stdout.flush()

                        log_prior = self.prior_logpdf(logk_prop)
                        log_terms = [math.log(w_prev[idx]) + self.K_logpdf(logk_prop, ks[idx, :, t-1]) for idx in
                                     range(self.N) if w_prev[idx] > 0]
                        log_back = logsumexp(np.array(log_terms))
                        
                        log_w_unnorm = log_prior - log_back 
                        w_unnorm_local[local_i] = math.exp(log_w_unnorm) if np.isfinite(log_w_unnorm) else 0.0

            all_ks = self.comm.gather(ks_local, root=0)
            all_w_unnorm = self.comm.gather(w_unnorm_local, root=0)
            all_rhos = self.comm.gather(rhos_local, root=0)
            total_attempts = self.comm.reduce(local_attempts, op=MPI.SUM, root=0)

            if self.rank == 0:
                ks[:, :, t] = np.vstack(all_ks)
                w_unnorm_full = np.concatenate(all_w_unnorm)
                acc_rhos = np.concatenate(all_rhos)
                
                mean_w = np.sum(w_unnorm_full) / total_attempts 
                Z_t = Z_t * mean_w
                evidence_by_generation.append(Z_t)
                
                wsum = np.sum(w_unnorm_full)
                w_norm = w_unnorm_full / wsum if wsum > 0 else np.ones(self.N) / self.N
                
                ESS = 1.0 / np.sum(w_norm**2)
                print(f"\n{self.model_name} Gen {t} ESS: {ESS:.2f} | Proposals: {total_attempts}")
                
                # Save the parameter dictionary BEFORE resampling so weights actually reflect particle density
                gen_dict = {k: ks[:, i, t].tolist() for i, k in enumerate(self.param_keys)}
                gen_dict['weights'] = w_norm.tolist()
                posterior_by_generation.append(gen_dict)

                # Resample within independent model
                J_resample = np.random.choice(self.N, size=self.N, replace=True, p=w_norm)
                ks[:, :, t] = ks[J_resample, :, t]
                W[:, t] = 1.0 / self.N
            
            ks = self.comm.bcast(ks, root=0)
            W = self.comm.bcast(W, root=0)
            acc_rhos = self.comm.bcast(acc_rhos, root=0)
            Z_t = self.comm.bcast(Z_t, root=0)
            self.total_proposals += self.comm.bcast(total_attempts, root=0)

        return posterior_by_generation, evidence_by_generation, self.total_proposals

# POSTERIOR PREDICTIVE Checks
def posterior_predictive_plot(final_results, x_df_list, t_obs_list, c_ini_list, n_samples=100):
    models = list(final_results.keys())
    evidences = np.array([final_results[m]['evidence'] for m in models])
    total_ev = np.sum(evidences)
    
    # Calculate analytical Model Posteriors P(M|x)
    p_M = evidences / total_ev if total_ev > 0 else np.ones(len(models)) / len(models)
    
    # Sample mechanisms according to their evidence
    sampled_models = np.random.choice(models, size=n_samples, p=p_M)

    model_colors = {'M1': 'steelblue', 'M2': 'darkorange', 'M3': 'seagreen'}
    species_cols = ['S', 'P']
    species_styles = {'S': '-', 'P': '--'}
    
    trajectories = [{sp: {m: [] for m in models} for sp in species_cols} for _ in range(len(t_obs_list))]

    print("\nRunning posterior predictive simulations (Pure ODE)...")
    for s_idx, m_name in enumerate(sampled_models):
        m_idx = int(m_name[-1]) - 1 
        last_gen = final_results[m_name]['posteriors'][-1]
        
        # Because we saved normalized weights, sample an index based on those weights
        p_idx = np.random.choice(len(last_gen['weights']), p=last_gen['weights'])
        
        # Extract kinetic parameters only (ignore noise_S, noise_P for deterministic ODE plotting)
        k_dict = {k: np.exp(last_gen[k][p_idx]) for k in model_keys[m_idx][:-2]}
        
        for obs_i, (c_ini, t_obs) in enumerate(zip(c_ini_list, t_obs_list)):
            model = model_classes[m_idx](k_dict, c_ini)
            sim = simulate_model_diffeq(model, t_obs, rtol=1e-6, atol=1e-9)
            if sim is not None:
                for sp in species_cols:
                    trajectories[obs_i][sp][m_name].append(sim[sp].values)
                    
        print(f"\r  Simulated {s_idx + 1}/{n_samples}", end="")
    print()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for obs_i in range(len(t_obs_list)):
        ax = axes[obs_i]
        t_obs = t_obs_list[obs_i]
        obs_df = x_df_list[obs_i]

        for sp in species_cols:
            for m in models:
                for traj in trajectories[obs_i][sp][m]:
                    ax.plot(t_obs, traj, color=model_colors[m], linestyle=species_styles[sp], alpha=0.15, linewidth=0.8)
            ax.plot(t_obs, obs_df[sp].values, color='black', linestyle=species_styles[sp], linewidth=2.0)

        ax.set_title(f'Observation {obs_i + 1}', fontsize=11)
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')

    handles = []
    for m in models:
        for sp in species_cols:
            handles.append(mlines.Line2D([], [], color=model_colors[m], linestyle=species_styles[sp], linewidth=2, label=f'{m} – {sp}'))
    for sp in species_cols:
        handles.append(mlines.Line2D([], [], color='black', linestyle=species_styles[sp], linewidth=2, label=f'Observed {sp}'))

    fig.legend(handles=handles, loc='lower center', ncol=len(models) + 1, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    plt.suptitle('Posterior Predictive — S and P across 4 Observations', fontsize=13)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig('posterior_predictive_smc.png', dpi=150, bbox_inches='tight')


# Execute Script
if __name__ == "__main__":
    import os
    import glob

    kmin_log = np.log(1e-5)
    kmax_log = np.log(1e5)

    # --- DYNAMICALLY FIND GLOBAL TEST CASE 
    """test_file_path = "path/to/test/case"
    local_idx = 32

    if rank == 0:
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"Could not find {test_file_path}")
            
        dataset = np.load(test_file_path)
        
        # Extract the specific local index directly
        raw_data = dataset['data'][local_idx]      
        raw_t_obs = dataset['t_obs'][local_idx]    
        raw_cat0 = dataset['cat0'][local_idx]      
        raw_S0_4 = dataset['S0_4'][local_idx]      
        true_theta = dataset['theta'][local_idx]   
        
        print(f"\nLoaded Specific Test Case")
        print(f"File: {test_file_path} (Local Index: {local_idx})")
        
        # 1. Reconstruct Initial Conditions (c_ini_list)
        c_ini_list = []
        for i in range(3):
            c_ini_list.append({'S': 1.0, 'P': 0.0, 'cat': float(raw_cat0[i]), 'catS': 0.0, 'cat_2': 0.0, 'cat_2S': 0.0})
        # Observation 4 uses the randomized initial split
        c_ini_list.append({'S': float(raw_S0_4), 'P': float(1.0 - raw_S0_4), 'cat': float(raw_cat0[3]), 'catS': 0.0, 'cat_2': 0.0, 'cat_2S': 0.0})
        
        # 2. Reconstruct Time Arrays (t_obs_list)
        t_obs_list = [raw_t_obs[i] for i in range(4)]
        
        # 3. Reconstruct Observations (x_df_list)
        x_df_list = []
        for i in range(4):
            df = pd.DataFrame({
                'time': t_obs_list[i],
                'S': raw_data[i, :, 0],
                'P': raw_data[i, :, 1]
            })
            x_df_list.append(df)
            
        print(f"True S0_4: {raw_S0_4:.4f}")
        print(f"True Theta (Kinetics + Noise): \n{true_theta}")
    else:
        c_ini_list, t_obs_list, x_df_list = None, None, None """

    # On the fly "observed data" simulation as example (the actual implementation reads data with the code docstrined above)
    if rank == 0:
        np.random.seed(29)
        
        # 1. Define specific observational inputs
        c_ini_list = [
            {'S': 1.0, 'P': 0.0, 'cat': 0.05933071, 'catS': 0, 'cat_2': 0, 'cat_2S': 0},
            {'S': 1.0, 'P': 0.0, 'cat': 0.04112486, 'catS': 0, 'cat_2': 0, 'cat_2S': 0},
            {'S': 1.0, 'P': 0.0, 'cat': 0.08603659, 'catS': 0, 'cat_2': 0, 'cat_2S': 0},
            {'S': 0.51543897, 'P': 0.48456102, 'cat': 0.04112486, 'catS': 0, 'cat_2': 0, 'cat_2S': 0}
        ]

        t_obs_list = [
            np.array([0., 0.0357, 0.0598, 0.0889, 0.1570, 0.1643, 0.1954, 0.2112, 0.2356, 0.3056, 0.3286, 0.3711, 0.4005, 0.4517, 0.4830, 0.4949, 0.5253, 0.5538, 0.6093, 0.6487, 0.6624, 0.7096]),
            np.array([0., 0.0209, 0.0567, 0.0721, 0.1009, 0.1559, 0.1920, 0.2447, 0.2976, 0.3387, 0.3839, 0.4117, 0.4549, 0.5094, 0.5163, 0.5819, 0.6076, 0.6174, 0.6279, 0.6517, 0.6566, 0.6844]),
            np.array([0., 0.0291, 0.0749, 0.1076, 0.1731, 0.2192, 0.2417, 0.3029, 0.3519, 0.4196, 0.4710, 0.5386, 0.5529, 0.5724, 0.6405, 0.6420, 0.7067, 0.7704, 0.7799, 0.7905, 0.8546, 0.9218]),
            np.array([0., 0.0609, 0.1116, 0.1648, 0.1863, 0.1868, 0.2308, 0.2926, 0.3590, 0.3950, 0.3956, 0.4078, 0.4392, 0.4554, 0.4573, 0.4708, 0.5058, 0.5147, 0.5226, 0.5342, 0.5365, 0.5914])
        ]
        
        # 2. Simulate Ground Truth Data to act as "Observed"
        print("Simulating synthetic 'observed' data using Model 1...")
        true_m_idx = 0  
        true_k_dict = {'1': 20.0, '-1': 5.0, '2': 3.0, '-2': 0.1} 
        true_sigma_S = 0.05 
        true_sigma_P = 0.05 
        
        x_df_list = []
        for c_ini, t_obs in zip(c_ini_list, t_obs_list):
            model = model_classes[true_m_idx](true_k_dict, c_ini)
            sim_df = simulate_model_diffeq(model, t_obs, rtol=1e-6, atol=1e-9)
            
            # Inject Gaussian measurement noise to replicate experimental variance
            sim_df['S'] += np.random.normal(0, true_sigma_S, len(t_obs))
            sim_df['P'] += np.random.normal(0, true_sigma_P, len(t_obs))
            x_df_list.append(sim_df)



    # Broadcast the data out to all CPU cores
    c_ini_list = comm.bcast(c_ini_list, root=0)
    t_obs_list = comm.bcast(t_obs_list, root=0)
    x_df_list = comm.bcast(x_df_list, root=0)

    if rank == 0: np.random.seed(29)
    else: np.random.seed(29 + rank)

    final_results = {}
    
    # Run the 3 Models independently
    for m_idx in [0, 1, 2]:
        smc = IndependentSMCABC(
            model_name=f"M{m_idx+1}",
            model_class=model_classes[m_idx],
            param_keys=model_keys[m_idx],
            c_ini_list=c_ini_list,      
            t_obs_list=t_obs_list, 
            prior_bounds_log=(kmin_log, kmax_log),
            x_df_list=x_df_list,
            N=3000,
            max_retries=15000
        )
        post_by_gen, ev_by_gen, total_sims = smc.run(T=6)
        final_results[f"M{m_idx+1}"] = {
            'posteriors': post_by_gen,
            'evidence': ev_by_gen[-1],
            'total_proposals': total_sims
        }

    # MODEL SELECTION & EXPORT
    if rank == 0:
        print("\n==========================================")
        print("   INDEPENDENT SMC-ABC MODEL SELECTION    ")
        print("==========================================")
        
        total_evidence = sum([data['evidence'] for m, data in final_results.items()])
        
        for m, data in final_results.items():
            prob = (data['evidence'] / total_evidence) * 100 if total_evidence > 0 else 0
            print(f"{m} Evidence (Z): {data['evidence']:.4e} | P(M|x): {prob:.2f}% | Total ODE Proposals: {data['total_proposals']}")
            
        print("\nBayes Factors (B_ij = Z_i / Z_j):")
        if final_results['M2']['evidence'] > 0:
            print(f"B_12: {final_results['M1']['evidence'] / final_results['M2']['evidence']:.2f}")
        if final_results['M3']['evidence'] > 0:
            print(f"B_13: {final_results['M1']['evidence'] / final_results['M3']['evidence']:.2f}")
            print(f"B_23: {final_results['M2']['evidence'] / final_results['M3']['evidence']:.2f}")

        with open('independent_smc_results_3M.pkl', 'wb') as f:
            pickle.dump(final_results, f)
        print("\nSaved all independent posteriors and evidences to independent_smc_results_3M.pkl")
        
        # Trigger Posterior Predictive Plot
        posterior_predictive_plot(final_results, x_df_list, t_obs_list, c_ini_list, n_samples=100)
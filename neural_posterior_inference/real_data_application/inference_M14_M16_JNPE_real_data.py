import os
# Use cpu for inference
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np
import keras
from keras import layers
from keras import ops
import bayesflow as bf
import pickle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from diffeqpy import de
import matplotlib as mpl
import pandas as pd

# Styling parameters for figures
mpl.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 15,
    'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': False        
})

N_OBS, MAX_TIMES = 3, 21
# Normalization constants to unscale neural network predictions 
MICRO_MEAN, MICRO_STD = 0.0, 12.0
NOISE_MEAN, NOISE_STD = -6.0, 1.0  
S0_molar = 0.01  # Real initial substrate concentration (10 mM)

MECH_COLORS = {"M14": "#0072B2", "M15": "#D55E00", "M16": "#009E73"}

# LaTeX formatted parameter names
PARAM_NAMES = {
    "M14": [r'$\log k_{1}$', r'$\log k_{-1}$', r'$\log k_{2}$', r'$\log k_{-2}$', r'$\log k_{-3}$', r'$\log \sigma_{P}$'],
    "M15": [r'$\log k_{1}$', r'$\log k_{-1}$', r'$\log k_{2}$', r'$\log k_{-2}$', r'$\log k_{-3}$', r'$\log \sigma_{P}$'],
    "M16": [r'$\log k_{1}$', r'$\log k_{-1}$', r'$\log k_{2}$', r'$\log k_{-2}$', r'$\log k_{-3}$', r'$\log \sigma_{P}$']
}

# ODEs
species = ['S', 'P', 'cat', 'catS', 'inactive_catS', 'inhibitor', 'inactive_catSI', 'inactive_catS2']

class BaseModel:
    def __init__(self, k, c):
        self.species_names = list(c.keys())
        self.species_index = {name: i for i, name in enumerate(self.species_names)}
        self.k = k
        self.x0 = np.array([c[name] for name in self.species_names], dtype=np.float64)

    def indexer(self, x): return {name: x[idx] for name, idx in self.species_index.items()}

class M14(BaseModel):

    def dxdt(self, t, x):
        c, k = self.indexer(x), self.k
        dx = np.zeros(len(x)) 
        dx[species.index("S")] = k['-1'] * c['catS'] - k['1'] * c['S'] * c['cat']
        dx[species.index("P")] = k['2'] * c['catS'] - k['-2'] * c['cat'] * c['P']
        dx[species.index("cat")] = (k['-1'] + k['2']) * c['catS'] - (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat']
        dx[species.index("catS")] = (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat'] - (k['-1'] + k['2'] + k['-3']) * c['catS']
        dx[species.index("inactive_catS")] = k['-3'] * c['catS']

        return dx
    
    def jacobian(self, t, x):
        c, k, n, idx = self.indexer(x), self.k, len(x), {sp: species.index(sp) for sp in species}
        J = np.zeros((n, n))

        J[idx["S"], idx["S"]] = -k['1'] * c['cat']
        J[idx["S"], idx["cat"]] = -k['1'] * c['S']
        J[idx["S"], idx["catS"]] = k['-1']
        J[idx["P"], idx["P"]] = -k['-2'] * c['cat']
        J[idx["P"], idx["cat"]] = -k['-2'] * c['P']
        J[idx["P"], idx["catS"]] = k['2']
        J[idx["cat"], idx["S"]] = -k['1'] * c['cat']
        J[idx["cat"], idx["P"]] = -k['-2'] * c['cat']
        J[idx["cat"], idx["cat"]] = -k['1'] * c['S'] - k['-2'] * c['P']
        J[idx["cat"], idx["catS"]] = k['-1'] + k['2']
        J[idx["catS"], idx["S"]] = k['1'] * c['cat']
        J[idx["catS"], idx["P"]] = k['-2'] * c['cat']
        J[idx["catS"], idx["cat"]] = k['1'] * c['S'] + k['-2'] * c['P']
        J[idx["catS"], idx["catS"]] = -(k['-1'] + k['2'] + k['-3'])
        J[idx["inactive_catS"], idx["catS"]] = k['-3']

        return J

class M15(BaseModel):
    def dxdt(self, t, x):
        c, k = self.indexer(x), self.k
        dx = np.zeros(len(x)) 

        dx[species.index("S")] = k['-1'] * c['catS'] - k['1'] * c['S'] * c['cat']
        dx[species.index("P")] = k['2'] * c['catS'] - k['-2'] * c['cat'] * c['P']
        dx[species.index("cat")] = (k['-1'] + k['2']) * c['catS'] - (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat']
        dx[species.index("catS")] = (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat'] - (k['-1'] + k['2'] + k['-3'] * c['inhibitor']) * c['catS']
        dx[species.index("inhibitor")] = -k['-3'] * c['inhibitor'] * c['catS']
        dx[species.index("inactive_catSI")] = k['-3'] * c['inhibitor'] * c['catS']

        return dx
    
    def jacobian(self, t, x):
        c, k, n, idx = self.indexer(x), self.k, len(x), {sp: species.index(sp) for sp in species}
        J = np.zeros((n, n))

        J[idx["S"], idx["S"]] = -k['1'] * c['cat']
        J[idx["S"], idx["cat"]] = -k['1'] * c['S']
        J[idx["S"], idx["catS"]] = k['-1']
        J[idx["P"], idx["P"]] = -k['-2'] * c['cat']
        J[idx["P"], idx["cat"]] = -k['-2'] * c['P']
        J[idx["P"], idx["catS"]] = k['2']
        J[idx["cat"], idx["S"]] = -k['1'] * c['cat']
        J[idx["cat"], idx["P"]] = -k['-2'] * c['cat']
        J[idx["cat"], idx["cat"]] = -k['1'] * c['S'] - k['-2'] * c['P']
        J[idx["cat"], idx["catS"]] = k['-1'] + k['2']
        J[idx["catS"], idx["S"]] = k['1'] * c['cat']
        J[idx["catS"], idx["P"]] = k['-2'] * c['cat']
        J[idx["catS"], idx["cat"]] = k['1'] * c['S'] + k['-2'] * c['P']
        J[idx["catS"], idx["catS"]] = -(k['-1'] + k['2'] + k['-3'] * c['inhibitor'])
        J[idx["catS"], idx["inhibitor"]] = -k['-3'] * c['catS']
        J[idx["inhibitor"], idx["catS"]] = -k['-3'] * c['inhibitor']
        J[idx["inhibitor"], idx["inhibitor"]] = -k['-3'] * c['catS']
        J[idx["inactive_catSI"], idx["catS"]] = k['-3'] * c['inhibitor']
        J[idx["inactive_catSI"], idx["inhibitor"]] = k['-3'] * c['catS']

        return J

class M16(BaseModel):

    def dxdt(self, t, x):
        c, k = self.indexer(x), self.k
        dx = np.zeros(len(x))

        dx[species.index("S")] = k['-1'] * c['catS'] - k['1'] * c['S'] * c['cat'] - k['-3'] * c['S'] * c['catS']
        dx[species.index("P")] = k['2'] * c['catS'] - k['-2'] * c['cat'] * c['P']
        dx[species.index("cat")] = (k['-1'] + k['2']) * c['catS'] - (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat']
        dx[species.index("catS")] = (k['1'] * c['S'] + k['-2'] * c['P']) * c['cat'] - (k['-1'] + k['2'] + k['-3'] * c['S']) * c['catS']
        dx[species.index("inactive_catS2")] = k['-3'] * c['S'] * c['catS']

        return dx
    
    def jacobian(self, t, x):
        c, k, n, idx = self.indexer(x), self.k, len(x), {sp: species.index(sp) for sp in species}
        J = np.zeros((n, n))

        J[idx["S"], idx["S"]] = -k['1'] * c['cat'] - k['-3'] * c['catS']
        J[idx["S"], idx["cat"]] = -k['1'] * c['S']
        J[idx["S"], idx["catS"]] = k['-1'] - k['-3'] * c['S']
        J[idx["P"], idx["P"]] = -k['-2'] * c['cat']
        J[idx["P"], idx["cat"]] = -k['-2'] * c['P']
        J[idx["P"], idx["catS"]] = k['2']
        J[idx["cat"], idx["S"]] = -k['1'] * c['cat']
        J[idx["cat"], idx["P"]] = -k['-2'] * c['cat']
        J[idx["cat"], idx["cat"]] = -k['1'] * c['S'] - k['-2'] * c['P']
        J[idx["cat"], idx["catS"]] = k['-1'] + k['2']
        J[idx["catS"], idx["S"]] = k['1'] * c['cat'] - k['-3'] * c['catS']
        J[idx["catS"], idx["P"]] = k['-2'] * c['cat']
        J[idx["catS"], idx["cat"]] = k['1'] * c['S'] + k['-2'] * c['P']
        J[idx["catS"], idx["catS"]] = -(k['-1'] + k['2'] + k['-3'] * c['S'])
        J[idx["inactive_catS2"], idx["S"]] = k['-3'] * c['catS']
        J[idx["inactive_catS2"], idx["catS"]] = k['-3'] * c['S']

        return J

MODELS_CONFIG = {
    "M14": {"class": M14, "keys": ['1', '-1', '2', '-2', '-3']},
    "M15": {"class": M15, "keys": ['1', '-1', '2', '-2', '-3']},
    "M16": {"class": M16, "keys": ['1', '-1', '2', '-2', '-3']},
}

def make_diffeq_rhs(model):
    def f(du, u, p, t): du[:] = model.dxdt(t, np.asarray(u))
    return f
    
def make_diffeq_jac(model):
    def jac(J, u, p, t): J[:, :] = model.jacobian(t, np.asarray(u))
    return jac

def get_smooth_simulation(model_class, k_dict, c_ini):
    safe_k = {key: float(np.clip(val, 1e-8, 1e6)) for key, val in k_dict.items()}
    model = model_class(safe_k, c_ini)
    t_eval = np.linspace(0.0, 1.0, 100)
    f = de.ODEFunction(make_diffeq_rhs(model), jac=make_diffeq_jac(model))
    prob = de.ODEProblem(f, model.x0, (0.0, 1.0))

    try:
        sol = de.solve(prob, de.Rodas5(), saveat=t_eval, reltol=1e-5, abstol=1e-8, maxiters=10000, verbose=False)
        if "Success" not in str(sol.retcode) and str(sol.retcode) != "Default": return None
        if not hasattr(sol, 'u') or len(sol.u) != len(t_eval): return None
        Y = np.stack([np.asarray(ui) for ui in sol.u], axis=0)

        return t_eval, Y[:, species.index('S')], Y[:, species.index('P')]
    
    except Exception:
        return None

# Permutation invariant CNN_Transformer trunk
class HybridSummaryTrunk(keras.Model):
    def __init__(self, filters=(64, 128, 256), d_model=256, n_heads=4, n_obs=3, dropout_rate=0.1):
        super().__init__()
        self.d_model, self.n_obs = d_model, n_obs

        # Temporal Feature Extractor (1D CNN)
        self.cnn = keras.Sequential()
        for f in filters:
            self.cnn.add(layers.Conv1D(f, kernel_size=3, padding='same', activation='relu'))
            self.cnn.add(layers.LayerNormalization())
        
        self.obs_proj = layers.Dense(d_model, activation='relu')

        # Set Attention Block (unordered set)
        self.cross_attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout_rate)
        self.cross_norm = layers.LayerNormalization()
        self.cross_ffn = keras.Sequential([layers.Dense(d_model * 2, activation='relu'), 
                                           layers.Dropout(dropout_rate), 
                                           layers.Dense(d_model)] )
        
        self.cross_ffn_norm = layers.LayerNormalization()

    def call(self, x, training=False):
        # x shape: (Batch, N_observations, Time, Channels)
        B, N, T, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2], ops.shape(x)[3]

        # Extract time channel to identify padded steps (padded values are -1.0)
        flat_mask = ops.reshape(ops.cast(x[..., 0] >= 0.0, dtype=x.dtype), (-1, T, 1))

        # Masked Global Average Pooling across the Time dimension            
        cnn_out = self.cnn(ops.reshape(x, (-1, T, C)), training=training) * flat_mask  

        obs_emb = self.obs_proj(ops.sum(cnn_out, axis=1) / ops.maximum(ops.sum(flat_mask, axis=1), 1.0)) 

        # Unflatten back to sets of observations: (Batch, N_obs, d_model)                           
        obs_emb = ops.reshape(obs_emb, (-1, self.n_obs, self.d_model)) 

        # Set Attention: Shared context across independent experiments
        obs_emb = self.cross_norm(obs_emb + self.cross_attn(obs_emb, obs_emb, training=training))

        # Final Mean Pooling across the N_obs dimension to yield a fixed-size representation
        return ops.mean(self.cross_ffn_norm(obs_emb + self.cross_ffn(obs_emb, training=training)), axis=1)

class JointSummaryHead(keras.Model):
    """
    Splits the summary embedding into two distinct tasks:
    1. logits_head: Categorical model selection (P(M|x)).
    2. embedding_head: conditional context for the Normalizing Flows (P(θ|x,M)).
    """

    def __init__(self, shared_trunk, n_models=3, embedding_dim=256):
        super().__init__()
        self.shared_trunk = shared_trunk

        # Shared MLP processing the representation
        self.shared_body = keras.Sequential(
            [layers.Dense(512, activation='relu'), 
             layers.Dense(256, activation='relu')]
             )
        # Dual Output Heads
        self.logits_head = layers.Dense(n_models) # For Model Selection
        self.embedding_head = layers.Dense(embedding_dim, activation='relu') # For Parameter Inference

    def call(self, x, training=False):
        body_out = self.shared_body(self.shared_trunk(x, training=training), training=training)
        return self.logits_head(body_out, training=training), self.embedding_head(body_out, training=training) 

# Real Data Inference
if __name__ == "__main__":
    OUT_DIR = 'real_data_inference_M14toM16'
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Building architectures and flows...")
    shared_trunk = HybridSummaryTrunk(filters=(64, 128, 256), d_model=256, n_heads=4, n_obs=N_OBS)
    summary_head = JointSummaryHead(shared_trunk, n_models=3, embedding_dim=256)

    flow_m14 = bf.networks.CouplingFlow(num_params=6, num_coupling_layers=8, coupling_design='affine')
    flow_m15 = bf.networks.CouplingFlow(num_params=6, num_coupling_layers=8, coupling_design='affine')
    flow_m16 = bf.networks.CouplingFlow(num_params=6, num_coupling_layers=8, coupling_design='affine')
    flow_map = {"M14": flow_m14, "M15": flow_m15, "M16": flow_m16}

    # Run dummy inputs to lock in shape dimensions before loading weights
    dummy_input = np.zeros((1, N_OBS, MAX_TIMES, 5), dtype=np.float32)
    dummy_logits, dummy_emb = summary_head(dummy_input, training=False)
    _ = flow_m14(np.zeros((1, 6), dtype=np.float32), conditions=dummy_emb, training=False)
    _ = flow_m15(np.zeros((1, 6), dtype=np.float32), conditions=dummy_emb, training=False)
    _ = flow_m16(np.zeros((1, 6), dtype=np.float32), conditions=dummy_emb, training=False)

    print("Loading trained weights...")
    try:
        with open("M14_M16_JNPE_summary_head.pkl", "rb") as f: summary_head.set_weights(pickle.load(f))
        with open("M14_M16_JNPE_flow_m14.pkl", "rb") as f: flow_m14.set_weights(pickle.load(f))
        with open("M14_M16_JNPE_flow_m15.pkl", "rb") as f: flow_m15.set_weights(pickle.load(f))
        with open("M14_M16_JNPE_flow_m16.pkl", "rb") as f: flow_m16.set_weights(pickle.load(f))

    except FileNotFoundError:
        print("WARNING: Weight files not found.")
        exit()

    # Inference on real data - protodeboronation
    print("\n--- Real Data Inference (protodeboronation) ---")
    t_max, S0_EXP = 84, 100.0
    # Values taken from tabel S22 of Ser et al., [2025]
    raw_time = np.array([0, 12, 24, 36, 48, 60, 72, 84], dtype=np.float32)
    raw_yields = np.array([
        [0.05, 0.31, 0.80, 1.24, 1.83, 2.25, 2.64, 3.13], 
        [0.12, 0.23, 0.53, 0.91, 1.48, 2.06, 2.82, 3.47], 
        [1.41, 2.38, 2.77, 3.84, 4.26, 6.29, 8.84, 9.13]  
    ], dtype=np.float32)

    cat0_exp = np.array([0.05, 0.10, 0.15], dtype=np.float32)
    inhib0_exp = np.array([0, 0, 0], dtype=np.float32)

    norm_time = raw_time / t_max
    norm_product = raw_yields / S0_EXP

    B = 1
    T_exp = len(raw_time)
    
    input_tensor = np.zeros((B, N_OBS, MAX_TIMES, 5), dtype=np.float32)
    input_tensor[..., 0] = -1.0 

    for i in range(N_OBS):
        for j in range(T_exp):
            input_tensor[0, i, j, 0] = norm_time[j]
            input_tensor[0, i, j, 1] = norm_product[i, j]
            input_tensor[0, i, j, 2] = cat0_exp[i]
            input_tensor[0, i, j, 3] = inhib0_exp[i]
            input_tensor[0, i, j, 4] = 1.0 if j == 0 else 0.0

    # Classify real data
    real_logits, real_emb = summary_head(input_tensor, training=False)
    real_probs = np.array(ops.softmax(real_logits))[0]
    
    print("\n--- Classification Probabilities ---")
    model_labels = ["M14", "M15", "M16"]
    for idx, m_name in enumerate(model_labels):
        print(f"{m_name}: {real_probs[idx]*100:.2f}%")

    # Compute stats for models 
    print("\n--- Posterior Parameter Estimates ---")
    log_t_max = np.log(t_max)
    log_S0 = np.log(S0_molar)

    # Compute summary statistics for all models 
    for m_name in model_labels:
        flow = flow_map[m_name]
        
        # Draw 2000 samples from the flow conditioned on the real experimental data
        raw_s = np.array(
            flow(np.random.normal(size=(2000, 6)).astype(np.float32), 
                 conditions=np.repeat(real_emb, 2000, axis=0), 
                 inverse=True, 
                 training=False)
                 )
        
        # Un normalize outputs
        log_k_infer = raw_s[:, :5] * MICRO_STD + MICRO_MEAN
        noise_P     = raw_s[:, -1:] * NOISE_STD + NOISE_MEAN

        # Revert dimensionless scaling to physical units (min^-1 or M^-1 min^-1).
        log_k_physical = np.zeros_like(log_k_infer)
        log_k_physical[:, 0] = log_k_infer[:, 0] - log_t_max - log_S0  
        log_k_physical[:, 1] = log_k_infer[:, 1] - log_t_max           
        log_k_physical[:, 2] = log_k_infer[:, 2] - log_t_max           
        log_k_physical[:, 3] = log_k_infer[:, 3] - log_t_max - log_S0  
        
        # M14's deactivation step is unimolecular, while M15/M16 are bimolecular
        if m_name == "M14": 
            log_k_physical[:, 4] = log_k_infer[:, 4] - log_t_max       
        else:                      
            log_k_physical[:, 4] = log_k_infer[:, 4] - log_t_max - log_S0 

        post_samples_physical = np.concatenate([log_k_physical, noise_P], axis=-1)
            
        post_means = np.mean(post_samples_physical, axis=0)
        post_medians = np.median(post_samples_physical, axis=0)
        
        print(f"\n{m_name} Parameters:")
        print(f"{'Parameter':<15} | {'Mean':<10} | {'Median':<10}")
        print("-" * 42)
        for idx, param in enumerate(PARAM_NAMES[m_name]):
            print(f"{param:<15} | {post_means[idx]:<10.4f} | {post_medians[idx]:<10.4f}")

    # Generate Posterior Predictive Checks
    print("\nGenerating PPC Plots...")
    fig_ppc, axes_ppc = plt.subplots(1, 3, figsize=(18, 5))
    fig_ppc.suptitle("Protodeboronation Real Data Posterior Predictive Distributions", fontsize=18)
    axes_ppc = axes_ppc.flatten()

    # Set up baseline initial conditions for the 3 real experimental runs
    c_ini_dicts = []
    for i in range(N_OBS):
        c_dict = {sp: 0.0 for sp in species}
        c_dict.update({'S': 1.0, 'P': 0.0, 'cat': float(cat0_exp[i]), 'inhibitor': 0.0})
        c_ini_dicts.append(c_dict)

    # Distribute 300 simulated trajectories across models sampling depending on their estimated probabilities
    N_PPC = 300
    model_counts = np.random.multinomial(N_PPC, real_probs)
    
    for m_idx, m_name in enumerate(model_labels):
        count = model_counts[m_idx]
        if count == 0: continue
        
        flow = flow_map[m_name]
        m_class = MODELS_CONFIG[m_name]["class"]
        m_keys = MODELS_CONFIG[m_name]["keys"]
        m_color = MECH_COLORS[m_name]
        
        raw_s = np.array(flow(np.random.normal(size=(count, 6)).astype(np.float32), conditions=np.repeat(real_emb, count, axis=0), inverse=True, training=False))
        post_samples = raw_s.copy()
        
        # The ODE simulator requires dimensionless parameters, so we only reverse the Z-score here
        post_samples[:, :5] = raw_s[:, :5] * MICRO_STD + MICRO_MEAN
        
        for sample_idx in range(count):
            log_k_raw = np.clip(post_samples[sample_idx, :5], -20.0, 15.0)
            k_values = [np.exp(val) for val in log_k_raw]
            
            for obs_idx in range(N_OBS):
                sim_res = get_smooth_simulation(m_class, dict(zip(m_keys, k_values)), c_ini_dicts[obs_idx])
                if sim_res is None: continue
                t_plot, S_plot, P_plot = sim_res
                
                # Plot simulated trajectories mapped back to the normalized [0, 1] scale
                axes_ppc[obs_idx].plot(t_plot * t_max, P_plot, color=m_color, alpha=0.15)

    for obs_idx in range(N_OBS):
        # Overlay the actual data points
        axes_ppc[obs_idx].scatter(raw_time, raw_yields[obs_idx] / S0_EXP, color='black', zorder=5)
        axes_ppc[obs_idx].set_title(f"Experiment {obs_idx+1} (cat={cat0_exp[obs_idx]*100:.0f}%)")
        axes_ppc[obs_idx].set_xlabel("Time (min)")
        axes_ppc[obs_idx].set_ylabel("Product Yield")
        axes_ppc[obs_idx].set_ylim(0, 0.25)

    handles = [mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Real Observations')]
    for m_idx, m_name in enumerate(model_labels):
        if model_counts[m_idx] > 0:
            handles.append(mlines.Line2D([], [], color=MECH_COLORS[m_name], lw=2, label=f'{m_name} Predictive (p={real_probs[m_idx]:.2f})'))

    fig_ppc.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(handles), fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.2) 
    plt.savefig(os.path.join(OUT_DIR, "REAL_DATA_PPC.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # Plot Parameter Posterior Distributions for the most likely mechanism
    print("Generating Parameter Posterior Plots...")
    cols = 3; rows = 2
    fig_post, axes_post = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    fig_post.suptitle(f"Parameter Posterior Distributions - Protodeboronation", fontsize=16)
    axes_post = axes_post.flatten()
    
    # Identify the winning model to visualize its parameter distributions
    winning_m_name = model_labels[np.argmax(real_probs)]
    winning_flow = flow_map[winning_m_name]
    
    # Draw a dense sample for the winning model 
    raw_s = np.array(
        winning_flow(np.random.normal(size=(2000, 6)).astype(np.float32),
                     conditions=np.repeat(real_emb, 2000, axis=0), 
                     inverse=True, 
                     training=False)
                     )
    
    log_k_infer = raw_s[:, :5] * MICRO_STD + MICRO_MEAN
    noise_P     = raw_s[:, -1:] * NOISE_STD + NOISE_MEAN

    # Apply the physical unit conversions (time and concentration adjustments)
    log_k_physical = np.zeros_like(log_k_infer)
    log_k_physical[:, 0] = log_k_infer[:, 0] - log_t_max - log_S0  
    log_k_physical[:, 1] = log_k_infer[:, 1] - log_t_max           
    log_k_physical[:, 2] = log_k_infer[:, 2] - log_t_max           
    log_k_physical[:, 3] = log_k_infer[:, 3] - log_t_max - log_S0  
    
    if winning_m_name == "M14": 
        log_k_physical[:, 4] = log_k_infer[:, 4] - log_t_max       
    else:                      
        log_k_physical[:, 4] = log_k_infer[:, 4] - log_t_max - log_S0 

    winning_post_samples_physical = np.concatenate([log_k_physical, noise_P], axis=-1)

    param_names = PARAM_NAMES[winning_m_name]
    for i in range(len(param_names)):
        ax = axes_post[i]
        
        # Plot Histogram + continuous KDE estimation curve
        sns.histplot(winning_post_samples_physical[:, i], ax=ax, color=MECH_COLORS[winning_m_name], stat='density', alpha=0.4, kde=True, line_kws={'linewidth': 2})
        ax.set_title(param_names[i])
        
        # Add a vertical marker indicating the Posterior Mean
        mean_val = np.mean(winning_post_samples_physical[:, i])
        ax.axvline(mean_val, color='black', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.2f}')
        ax.legend()
        
    for i in range(len(param_names), len(axes_post)): axes_post[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "REAL_DATA_posteriors.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # Bivariate posterior distributions
    print("Generating Bivariate Parameter Posterior Plots...")
    
    param_names = PARAM_NAMES[winning_m_name]
    df_post = pd.DataFrame(winning_post_samples_physical, columns=param_names)
    
    # Initialize a PairGrid that only plots the lower triangle
    g = sns.PairGrid(df_post, corner=True, diag_sharey=False)
    
    # Map 2D Kernel Density Estimates to the lower triangle
    g.map_lower(sns.kdeplot, fill=True, cmap="Blues", alpha=0.8, thresh=0.05)
    
    # Map 1D Histograms + KDEs to the diagonal
    g.map_diag(sns.histplot, color=MECH_COLORS[winning_m_name], stat='density', kde=True, alpha=0.4)
    
    # Add means to the diagonal plots for reference
    for i, param in enumerate(param_names):
        mean_val = np.mean(winning_post_samples_physical[:, i])
        g.axes[i, i].axvline(mean_val, color='black', linestyle='--', linewidth=2)
    
    plt.suptitle(f"Bivariate Posterior Distributions - {winning_m_name}", y=1.02, fontsize=18)
    plt.savefig(os.path.join(OUT_DIR, "REAL_DATA_bivariate_posteriors.pdf"), format='pdf', bbox_inches='tight')
    plt.close()
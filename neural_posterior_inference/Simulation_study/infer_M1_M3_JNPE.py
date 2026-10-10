import os
# Use cpu for inference
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import glob
import numpy as np
import pandas as pd
import keras
from keras import layers
from keras import ops
import bayesflow as bf
import pickle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve
from diffeqpy import de
import matplotlib as mpl

# Styling parameters for figures
mpl.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 15,
    'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': False        
})

N_OBS, MAX_TIMES = 4, 21
# Normalization constants to unscale neural network predictions 
MICRO_MEAN, MICRO_STD = 0.0, 12.0
NOISE_MEAN, NOISE_STD = -6.0, 1.0  

MECH_COLORS = {"M1": "#0072B2", "M2": "#D55E00", "M3": "#009E73"}

# LaTeX formatted parameter names 
PARAM_NAMES = {
    "M1": [r'$\log k_{1}$', r'$\log k_{-1}$', r'$\log k_{2}$', r'$\log k_{-2}$', r'$\log \sigma_{S}$', r'$\log \sigma_{P}$'],
    "M2": [r'$\log k_{1}$', r'$\log k_{-1}$', r'$\log k_{2}$', r'$\log k_{-2}$', r'$\log k_{3}$', r'$\log k_{-3}$', r'$\log \sigma_{S}$', r'$\log \sigma_{P}$'],
    "M3": [r'$\log k_{-3}$', r'$\log k_{3}$', r'$\log k_{1}$', r'$\log k_{-1}$', r'$\log k_{2}$', r'$\log k_{-2}$', r'$\log \sigma_{S}$', r'$\log \sigma_{P}$']
}

# ODEs
species = ['S', 'P', 'cat', 'catS', 'cat_2', 'cat_2S']
class BaseModel:
    def __init__(self, k, c):
        self.species_index = {name: i for i, name in enumerate(list(c.keys()))}
        self.k, self.x0 = k, np.array(list(c.values()), dtype=np.float64)
    def indexer(self, x): return {name: x[idx] for name, idx in self.species_index.items()}

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
        J[idx["catS"], idx["catS"]] = -k['-1'] - k['2']

        return J

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
        J[idx["cat"], idx["cat"]] = -k['1'] * c['S'] - k['-2'] * c['P'] - 4 * k['-3'] * c['cat']
        J[idx["cat"], idx["catS"]] = k['-1'] + k['2']
        J[idx["cat"], idx["cat_2"]] = 2 * k['3']
        J[idx["catS"], idx["S"]] = k['1'] * c['cat']
        J[idx["catS"], idx["P"]] = k['-2'] * c['cat']
        J[idx["catS"], idx["cat"]] = k['1'] * c['S'] + k['-2'] * c['P']
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
        c, k, n, idx = self.indexer(x), self.k, len(x), {sp: species.index(sp) for sp in species}
        J = np.zeros((n, n))

        J[idx["S"], idx["S"]] = -k['4'] * c['cat_2']
        J[idx["S"], idx["cat_2"]] = -k['4'] * c['S']
        J[idx["S"], idx["cat_2S"]] = k['-4']
        J[idx["P"], idx["P"]] = -k['-5'] * c['cat_2']
        J[idx["P"], idx["cat_2"]] = -k['-5'] * c['P']
        J[idx["P"], idx["cat_2S"]] = k['5']
        J[idx["cat"], idx["cat"]] = -4 * k['-3'] * c['cat']
        J[idx["cat"], idx["cat_2"]] = 2 * k['3']
        J[idx["cat_2S"], idx["S"]] = k['4'] * c['cat_2']
        J[idx["cat_2S"], idx["P"]] = k['-5'] * c['cat_2']
        J[idx["cat_2S"], idx["cat_2"]] = k['4'] * c['S'] + k['-5'] * c['P']
        J[idx["cat_2S"], idx["cat_2S"]] = -k['-4'] - k['5']
        J[idx["cat_2"], idx["S"]] = -k['4'] * c['cat_2']
        J[idx["cat_2"], idx["P"]] = -k['-5'] * c['cat_2']
        J[idx["cat_2"], idx["cat"]] = 2 * k['-3'] * c['cat']
        J[idx["cat_2"], idx["cat_2"]] = -k['4'] * c['S'] - k['-5'] * c['P'] - k['3']
        J[idx["cat_2"], idx["cat_2S"]] = k['-4'] + k['5']

        return J

MODELS_CONFIG = {
    "M1": {"class": M1, "keys": ['1', '-1', '2', '-2']},
    "M2": {"class": M2, "keys": ['1', '-1', '2', '-2', '3', '-3']},
    "M3": {"class": M3, "keys": ['3', '-3', '4', '-4', '5', '-5']},
}

def make_diffeq_rhs(model):
    def f(du, u, p, t): du[:] = model.dxdt(t, np.asarray(u))
    return f

def make_diffeq_jac(model):
    def jac(J, u, p, t): J[:, :] = model.jacobian(t, np.asarray(u))
    return jac

def get_smooth_simulation(model_class, k_dict, c_ini):
    """ Executes a high-resolution forward ODE simulation (100 points) to generate 
        smooth continuous lines for the Posterior Predictive Check plots. """
    
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
    """
    Extracts summary embeddings from experimental data.
    Uses a 1D-CNN for temporal processing, followed by a Set Attention Block to maintain permutation invariance across the 4 observations.
    """
    def __init__(self, filters=(64, 128, 256), d_model=256, n_heads=4, n_obs=4, dropout_rate=0.1):
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
        self.cross_ffn = keras.Sequential(
            [layers.Dense(d_model * 2, activation='relu'), 
             layers.Dropout(dropout_rate), 
             layers.Dense(d_model)]
             )
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
        body_out = self.shared_body(
            self.shared_trunk(x, training=training), training=training )
        return self.logits_head(body_out, training=training), self.embedding_head(body_out, training=training) 

# Load testig data
def validation_crn_bayesflow_adapter(raw_data, t_obs, cat0, S0_4, theta, model_name):
    """
    Prepares test sets for the network, and applies temporal dropout. T_keep is the number of observations.
    """
    B, _, T, _ = raw_data.shape

    # Channel Assembly (Time, S, P, Initial Catalyst, Time-Zero Indicator)
    SP_data = raw_data[..., :2].copy()
    SP_data[:, 3, 0, 0] = S0_4; SP_data[:, :3, 0, 0] = 1.0            
    SP_data[:, 3, 0, 1] = 1.0 - S0_4; SP_data[:, :3, 0, 1] = 0.0

    is_t0 = np.zeros((B, N_OBS, T, 1), dtype=np.float32)
    is_t0[:, :, 0, :] = 1.0
    cat0_channel = np.broadcast_to(cat0[:, :, np.newaxis, np.newaxis], (B, N_OBS, T, 1)).astype(np.float32)
    
    data_full = np.concatenate([t_obs[..., np.newaxis], SP_data, cat0_channel, is_t0], axis=-1).astype(np.float32)                                                         
    
    # Temporal dropout 
    T_keep = 20 # change to test other data counts (e.g., 10)
    data_dropped = np.zeros_like(data_full)
    data_dropped[..., 0] = -1.0  # Pad omitted time points with -1.0 so the mask drops them
    for b in range(B):
        for n in range(N_OBS):
            idx = np.sort(np.random.choice(T, T_keep, replace=False))
            data_dropped[b, n, :T_keep, :] = data_full[b, n, idx, :]
            
    return {'summary_variables': data_dropped}, theta.copy()

def load_validation_data(n_per_model=1000):
    """ Loads unseen datasets for final evaluation. """

    print("\n--- Loading Test Data ---")
    val_set = []
    for m_idx, m_name in enumerate(["M1", "M2", "M3"]):
        files = glob.glob(f"{m_name}_testing_data/*.npz")
        raw_list, t_list, cat0_list, S0_list, theta_list = [], [], [], [], []
        for f in files:
            data = np.load(f)
            raw_list.append(data['data']); t_list.append(data['t_obs'])
            cat0_list.append(data['cat0']); S0_list.append(data['S0_4']); theta_list.append(data['theta'])
            
        raw_data = np.concatenate(raw_list, axis=0)[:n_per_model]
        t_obs = np.concatenate(t_list, axis=0)[:n_per_model]
        cat0 = np.concatenate(cat0_list, axis=0)[:n_per_model]
        S0_4 = np.concatenate(S0_list, axis=0)[:n_per_model]
        theta = np.concatenate(theta_list, axis=0)[:n_per_model]
        
        sum_dict, true_micro = validation_crn_bayesflow_adapter(raw_data, t_obs, cat0, S0_4, theta, m_name)
        for i in range(len(raw_data)):
            val_set.append({'true_model': m_name, 'true_model_idx': m_idx, 'summary_variables': sum_dict['summary_variables'][i:i+1],
                            'true_micro': true_micro[i], 'cat0': cat0[i], 'S0_4': S0_4[i]})
        print(f"Loaded {len(raw_data)} test samples for {m_name}.")
    np.random.shuffle(val_set)
    return val_set

# Performance metrics
def plot_multiclass_calibration(y_true, y_probs, out_dir, n_bins=10, model_names=["M1", "M2", "M3"]):
    """ Evaluates the reliability of the model selection probabilities (Expected Calibration Error). """

    y_true, y_probs = np.array(y_true), np.array(y_probs)
    plt.figure(figsize=(8, 8))
    
    # The diagonal y=x line represents perfect calibration: 
    # e.g., when the model predicts 80% probability, it is correct exactly 80% of the time.
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    ece_scores = {}
    
    for i, m_name in enumerate(model_names):
        
        # current model's ground truth (1 if true model, 0 otherwise) 
        # and its estimated probability.
        y_true_binary = (y_true == i).astype(int)
        prob_m = y_probs[:, i]
        
        # Brier score (Mean Squared Error of predictions)
        brier = brier_score_loss(y_true_binary, prob_m)
        
        # prob_true: The actual empirical accuracy in each bin.
        # prob_pred: The average estimated probability in each bin.
        prob_true, prob_pred = calibration_curve(y_true_binary, prob_m, n_bins=n_bins, strategy='uniform')
        
        # Calculate Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for j in range(n_bins):
            # predictions falling inside the current probability bin.
            # The final bin uses <= to ensure probability sums to 1
            bin_mask = (prob_m >= bin_edges[j]) & (prob_m <= bin_edges[j+1] if j == n_bins - 1 else prob_m < bin_edges[j+1])
            bin_count = np.sum(bin_mask)
            
            if bin_count > 0:
                # ECE Component = (Bin Weight) * |True Bin Accuracy - Mean Bin Confidence|
                bin_weight = bin_count / len(prob_m)
                bin_accuracy = np.mean(y_true_binary[bin_mask])
                bin_confidence = np.mean(prob_m[bin_mask])
                
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
                
        ece_scores[m_name] = ece
        
        # Plot the reliability diagram for current model  
        plt.plot(prob_pred, prob_true, "s-", color=MECH_COLORS[m_name], label=f"{m_name} (ECE={ece:.3f}, Brier={brier:.3f})")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Empirical Accuracy")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multiclass_calibration_curve.pdf"), format='pdf', bbox_inches='tight')
    plt.close()
    return ece_scores

def evaluate_parameter_metrics(model_eval_dict, out_dir):
    """ Evaluates parameter point estimation accuracy (RMSE/R2) and posterior consistency via Simulation-Based Calibration (SBC). """

    print("\n--- PARAMETER INFERENCE METRICS ---")
    for m_name, data in model_eval_dict.items():
        if len(data['true']) == 0: continue
        true_vals = np.array(data['true'])            # Shape: (N, Parameters)
        post_samples = np.array(data['post'])         # Shape: (N, 2000_Samples, Parameters)
        post_means = np.mean(post_samples, axis=1)

        # R2 between posterior mean and true value
        rmse = np.sqrt(np.mean((post_means - true_vals)**2, axis=0))
        ss_res = np.sum((true_vals - post_means)**2, axis=0)
        ss_tot = np.sum((true_vals - np.mean(true_vals, axis=0))**2, axis=0)
        r2 = 1 - (ss_res / (ss_tot + 1e-8)) 

        # Variance shrinkage
        prior_var = np.var(true_vals, axis=0) + 1e-8
        post_var = np.mean(np.var(post_samples, axis=1), axis=0)
        shrinkage = 1.0 - (post_var / prior_var)
        
        # Simulation-Based Calibration (SBC) Ranks
        ranks = np.sum(post_samples < true_vals[:, None, :], axis=1) / post_samples.shape[1]
        
        print(f"\nModel {m_name}:")
        params = PARAM_NAMES[m_name]
        for i, p in enumerate(params):
            print(f"  {p:12s}: RMSE = {rmse[i]:.3f} | R2 = {r2[i]:.3f} | Shrinkage = {shrinkage[i]:.3f}")
            
        n_params = len(params)
        fig, axes = plt.subplots(n_params, 2, figsize=(10, 3 * n_params))
        fig.suptitle(f"{m_name} Parameter Recovery & SBC", fontsize=16)
        
        for i, p in enumerate(params):
            ax_scatter, ax_hist = axes[i, 0], axes[i, 1]
            
            ax_scatter.scatter(true_vals[:, i], post_means[:, i], alpha=0.5, color=MECH_COLORS[m_name], s=10)
            min_val, max_val = np.min(true_vals[:, i]), np.max(true_vals[:, i])
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
            ax_scatter.set_title(f"{p} (R2={r2[i]:.2f})"); ax_scatter.set_xlabel("True Value"); ax_scatter.set_ylabel("Posterior Mean")
            
            # A uniform SBC histogram indicates that the inferred posteriors match the true data-generating process
            ax_hist.hist(ranks[:, i], bins=20, range=(0, 1), color=MECH_COLORS[m_name], alpha=0.7, edgecolor='black', density=True)
            ax_hist.axhline(1.0, color='k', linestyle='dashed')
            ax_hist.set_title(f"{p} Rank Histogram"); ax_hist.set_xlabel("Rank")
            
        plt.tight_layout(); plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(out_dir, f"parameter_recovery_{m_name}.pdf"), format='pdf', bbox_inches='tight'); plt.close()

# Execution
if __name__ == "__main__":
    OUT_DIR = 'validation_plots'
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Building architecture and flows...")
    shared_trunk = HybridSummaryTrunk(filters=(64, 128, 256), d_model=256, n_heads=4, n_obs=N_OBS)
    summary_head = JointSummaryHead(shared_trunk, n_models=3, embedding_dim=256)

    flow_m1 = bf.networks.CouplingFlow(num_params=6, num_coupling_layers=8, coupling_design='affine')
    flow_m2 = bf.networks.CouplingFlow(num_params=8, num_coupling_layers=8, coupling_design='affine')
    flow_m3 = bf.networks.CouplingFlow(num_params=8, num_coupling_layers=8, coupling_design='affine')

    # Run dummy inputs to lock in shape dimensions before loading weights
    dummy_input = np.zeros((1, N_OBS, MAX_TIMES, 5), dtype=np.float32)
    dummy_logits, dummy_emb = summary_head(dummy_input, training=False)
    _ = flow_m1(np.zeros((1, 6), dtype=np.float32), conditions=dummy_emb, training=False)
    _ = flow_m2(np.zeros((1, 8), dtype=np.float32), conditions=dummy_emb, training=False)
    _ = flow_m3(np.zeros((1, 8), dtype=np.float32), conditions=dummy_emb, training=False)

    print("Loading learned weights...")
    try:
        with open("M1_M3_JNPE_summary_head.pkl", "rb") as f: summary_head.set_weights(pickle.load(f))
        with open("M1_M3_JNPE_flow_m1.pkl", "rb") as f: flow_m1.set_weights(pickle.load(f))
        with open("M1_M3_JNPE_flow_m2.pkl", "rb") as f: flow_m2.set_weights(pickle.load(f))
        with open("M1_M3_JNPE_flow_m3.pkl", "rb") as f: flow_m3.set_weights(pickle.load(f))
    except FileNotFoundError:
        print("WARNING: Weight files not found. ")
        exit()

    val_data = load_validation_data(n_per_model=1000) 
    total_samples = len(val_data)
    
    y_true, y_pred, all_probs = [], [], []
    coverage_counts = {"M1": np.zeros(6), "M2": np.zeros(8), "M3": np.zeros(8)}
    total_counts = {"M1": 0, "M2": 0, "M3": 0}
    stashed_ppc_cases = []
    
    model_eval_dict = {"M1": {"true": [], "post": []}, "M2": {"true": [], "post": []}, "M3": {"true": [], "post": []}}
    
    print(f"\n--- Starting Evaluation Loop ({total_samples} Iterations) ---")
    for i, sample in enumerate(val_data):
        if i % 100 == 0: print(f"Processing testing sample {i}/{total_samples}...")
        
        # Model Selection: P(M|x)
        logits, emb = summary_head(sample['summary_variables'], training=False)
        probs = np.array(ops.softmax(logits))[0]
        y_true.append(sample['true_model_idx'])
        y_pred.append(np.argmax(probs))
        all_probs.append(probs)
        
        # Parameter Inference: P(θ|x, M_true) - Inverse map conditions through the respective Normalizing Flow
        true_m = sample['true_model']
        flow = {"M1": flow_m1, "M2": flow_m2, "M3": flow_m3}[true_m]
        raw_samples = np.array(
            flow(
                np.random.normal(size=(2000, len(sample['true_micro']))).astype(np.float32), 
                conditions=np.repeat(emb, 2000, axis=0), 
                inverse=True, training=False)
                )
        
        # Inverse Scaling outputs
        post_samples = raw_samples.copy()
        post_samples[:, :-2] = raw_samples[:, :-2] * MICRO_STD + MICRO_MEAN
        post_samples[:, -2:] = raw_samples[:, -2:] * NOISE_STD + NOISE_MEAN
        
        model_eval_dict[true_m]['true'].append(sample['true_micro'])
        model_eval_dict[true_m]['post'].append(post_samples)
        
        # Record 95% Credible Interval Coverage
        low, high = np.percentile(post_samples, 2.5, axis=0), np.percentile(post_samples, 97.5, axis=0)
        coverage_counts[true_m] += (sample['true_micro'] >= low) & (sample['true_micro'] <= high)
        total_counts[true_m] += 1
        
        # Output individual results for 10 random test cases
        if len(stashed_ppc_cases) < 10 and np.random.rand() < 0.05:
            stashed_ppc_cases.append({'idx': i, 'true_m': true_m, 'probs': probs, 'emb': emb, 
                                      'summary_variables': sample['summary_variables'], 'cat0': sample['cat0'], 
                                      'S0_4': sample['S0_4'], 'true_micro': sample['true_micro']})

    print("\n==========================================")
    print("            VALIDATION REPORT     ")
    print("==========================================")
    print("\n--- CLASSIFICATION PERFORMANCE ---")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["M1", "M2", "M3"]))
    
    ece_results = plot_multiclass_calibration(y_true, all_probs, OUT_DIR)
    for m in ["M1", "M2", "M3"]: print(f"Calibration {m}: ECE = {ece_results[m]:.4f}")
    
    evaluate_parameter_metrics(model_eval_dict, OUT_DIR)
    
    print("\n--- POSTERIOR COVERAGE (95% CI) ---")
    for m in ["M1", "M2", "M3"]:
        print(f"\nModel {m} (N = {total_counts[m]}):")
        for name, pct in zip(PARAM_NAMES[m], (coverage_counts[m] / total_counts[m]) * 100):
            print(f"  {name:12s}: {pct:.1f}%")

    # PPCs
    print("\n--- Generating PPC and Posterior Plots ---")
    for case in stashed_ppc_cases:
        N_PPC = 100
        # Chose model based on posterior probabilities
        pvals = np.clip(np.asarray(case['probs'], dtype=np.float64), 0.0, 1.0)
        pvals /= np.sum(pvals)
        model_counts = np.random.multinomial(N_PPC, pvals)
        prob_str = {m: pvals[idx] for idx, m in enumerate(["M1", "M2", "M3"])}
        
        ppc_samples_by_model = {}
        
        for i, m_name in enumerate(["M1", "M2", "M3"]):
            # Skip models with zero prob
            if model_counts[i] == 0: continue
            
            # Generate posterior parameter samples P(theta | x, M) conditioned on the summary embedding.
            raw_s = np.array(
                [flow_m1, flow_m2, flow_m3][i](np.random.normal(size=(model_counts[i], len(PARAM_NAMES[m_name]))).astype(np.float32),
                                                conditions=np.repeat(case['emb'], model_counts[i], axis=0),
                                                  inverse=True,
                                                    training=False)
                                                    )
            
            # Revert normalization applied during training
            unscaled = raw_s.copy()
            unscaled[:, :-2] = raw_s[:, :-2] * MICRO_STD + MICRO_MEAN
            unscaled[:, -2:] = raw_s[:, -2:] * NOISE_STD + NOISE_MEAN
            ppc_samples_by_model[m_name] = unscaled

        # Plot titles     
        fig_ppc, axes_ppc = plt.subplots(N_OBS, 2, figsize=(14, 16))
        pred_m = ["M1", "M2", "M3"][np.argmax(pvals)]
        fig_ppc.suptitle(f"PPC Case {case['idx']} (True: {case['true_m']} | Pred: {pred_m})", fontsize=18, y=0.98)
        
        obs_titles = [f"Obs 1 (cat0={case['cat0'][0]:.3f})", 
                      f"Obs 2 (cat0={case['cat0'][1]:.3f})", 
                      f"Obs 3 (cat0={case['cat0'][2]:.3f})", 
                      f"Obs 4 (cat0={case['cat0'][3]:.3f}, S0={case['S0_4']:.2f})"]
        
        # Plot True data vs. estimated trajectories
        for obs_idx in range(N_OBS):
            ax_P, ax_S = axes_ppc[obs_idx, 0], axes_ppc[obs_idx, 1]
            ax_P.set_title(f"{obs_titles[obs_idx]} - Product"); ax_S.set_title(f"{obs_titles[obs_idx]} - Substrate")
            ax_P.set_ylim(-0.05, 1.05); ax_S.set_ylim(-0.05, 1.05)
            
            # Experimental data
            sum_vars = case['summary_variables'][0]  
            t_net = sum_vars[obs_idx, :, 0]
            
            # Filter temporal dropout mask
            valid_mask = t_net >= 0.0
            t_scatter = t_net[valid_mask]
            S_scatter = sum_vars[obs_idx, valid_mask, 1]
            P_scatter = sum_vars[obs_idx, valid_mask, 2]
            
            ax_P.scatter(t_scatter, P_scatter, color='black', zorder=5)
            ax_S.scatter(t_scatter, S_scatter, color='black', zorder=5)
            
            # Initial conditions dictionary for the ODE simulator
            c_ini_base = {sp: 0.0 for sp in species}
            if obs_idx < 3: 
                c_ini_base.update({'S': 1.0, 'P': 0.0, 'cat': float(case['cat0'][obs_idx])})
            else: 
                # Observation 4 uses the randomized initial split of S and P
                c_ini_base.update({'S': float(case['S0_4']), 'P': float(1.0 - case['S0_4']), 'cat': float(case['cat0'][obs_idx])})
            
            # Simulate and plot a smooth ODE trajectory for every posterior parameter sample
            for m_name, samples in ppc_samples_by_model.items():
                m_class, m_keys, m_color = MODELS_CONFIG[m_name]["class"], MODELS_CONFIG[m_name]["keys"], MECH_COLORS[m_name]
                
                for sample_idx in range(len(samples)):
                    log_k_raw = samples[sample_idx, :-2]
                    k_values = [np.exp(val) for val in log_k_raw]
                    
                    # Run high-resolution simulation
                    sim_res = get_smooth_simulation(m_class, dict(zip(m_keys, k_values)), c_ini_base)
                    if sim_res is None: continue
                    
                    t_plot, S_plot, P_plot = sim_res
                    
                    # Plot trajectories with transparency
                    ax_P.plot(t_plot, P_plot, color=m_color, alpha=0.15)
                    ax_S.plot(t_plot, S_plot, color=m_color, alpha=0.15)

        handles = [mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='True Product (20 pts)'), 
                   mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='True Substrate (20 pts)')]
        
        for m in sorted(ppc_samples_by_model.keys()): 
            handles.append(mlines.Line2D([], [], color=MECH_COLORS[m], lw=2, label=f'{m} Predictive (p={prob_str[m]:.2f})'))
            
        fig_ppc.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(handles), fontsize=13)

        # Posterior distributions for true model
        true_m_case = case['true_m']
        flow_true = {"M1": flow_m1, "M2": flow_m2, "M3": flow_m3}[true_m_case]
        
        # Draw 2000 samples specifically for the true model to produce a smooth KDE
        raw_s_post = np.array(
            flow_true(np.random.normal(size=(2000, len(PARAM_NAMES[true_m_case]))).astype(np.float32), 
                      conditions=np.repeat(case['emb'], 2000, axis=0), 
                      inverse=True, training=False)
                      )
        post_s_scaled = raw_s_post.copy()
        post_s_scaled[:, :-2] = raw_s_post[:, :-2] * MICRO_STD + MICRO_MEAN
        post_s_scaled[:, -2:] = raw_s_post[:, -2:] * NOISE_STD + NOISE_MEAN
        
        n_params = len(PARAM_NAMES[true_m_case])
        cols = 3
        rows = (n_params + cols - 1) // cols
        
        fig_post, axes_post = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        fig_post.suptitle(f"Posterior Distributions - PPC Case {case['idx']} (True Model: {true_m_case})", fontsize=18, y=0.98)
        axes_post = axes_post.flatten()
        
        for pid in range(n_params):
            ax = axes_post[pid]
            
            # Plot Histogram + KDE estimation curve
            sns.histplot(post_s_scaled[:, pid], ax=ax, color=MECH_COLORS[true_m_case], stat='density', alpha=0.4, kde=True, line_kws={'linewidth': 2})
            
            # Add mark for the True Value
            true_val = case['true_micro'][pid]
            ax.axvline(true_val, color='black', linestyle='--', linewidth=2.5, label=f'True: {true_val:.2f}')
            
            ax.set_title(PARAM_NAMES[true_m_case][pid])
            ax.legend()
            
        # Hide any unused subplots (e.g. for M1 which has 6 params in a 9-grid)
        for pid in range(n_params, len(axes_post)):
            axes_post[pid].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"case_{case['idx']}_true_{case['true_m']}_posteriors.pdf"), format='pdf', bbox_inches='tight')
        plt.close()
import os
import glob
import numpy as np
import keras
from keras import layers
from keras import ops
import bayesflow as bf
import pickle
import matplotlib.pyplot as plt
import time

# Randomly load and batch training data 
class SimulationDataLoader:
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found in {data_dir}")
        print(f"[{data_dir}] Found {len(self.files)} chunks.")
        np.random.shuffle(self.files)
        self.file_idx = 0
        self.load_chunk(self.files[self.file_idx])

    # Training data is stored in various files
    def load_chunk(self, filepath):
        data = np.load(filepath)
        self.obs_data  = data['data']           
        self.t_obs     = data['t_obs']          
        self.cat0      = data['cat0']           
        self.theta     = data['theta']          
        self.S0_4      = data['S0_4']
        
        self.inhib0 = data['inhib0'] if 'inhib0' in data else np.zeros_like(self.cat0)
        
        self.num_samples = self.obs_data.shape[0]
        self.indices   = np.random.permutation(self.num_samples)
        self.pointer   = 0

    def sample(self, batch_size):
        if self.pointer + batch_size > self.num_samples:
            self.file_idx = (self.file_idx + 1) % len(self.files)
            if self.file_idx == 0:
                np.random.shuffle(self.files)
            self.load_chunk(self.files[self.file_idx])
            
        idx = self.indices[self.pointer : self.pointer + batch_size]
        self.pointer += batch_size
        
        return {
            'sim_data':    {'data': self.obs_data[idx]},
            'prior_draws': {
                'theta': self.theta[idx],
                'context': {
                    't_obs':      self.t_obs[idx],
                    'cat0_all':   self.cat0[idx],
                    'inhib0_all': self.inhib0[idx],
                    'S0_4':       self.S0_4[idx]
                }
            }
        }

# number of experiments
N_OBS      = 3  
# Normalization parameters
K_MEAN     = 0.0
K_STD      = 12.0
NOISE_MEAN = -6.0  
NOISE_STD  = 1.0


# Data processing adapter
def crn_bayesflow_adapter(forward_dict, model_name, **kwargs):
    # scale parameters 
    if kwargs.get('inverse', False):
        theta_scaled = forward_dict['inference_variables']
        k_raw = theta_scaled[..., :5] * K_STD + K_MEAN
        noise_P = theta_scaled[..., -1:] * NOISE_STD + NOISE_MEAN
        forward_dict['inference_variables'] = np.concatenate([k_raw, noise_P], axis=-1)
        return forward_dict

    # For pre-processed data
    if 'summary_variables' in forward_dict:
        return forward_dict

    raw_data = forward_dict['sim_data']['data'][:, :3, :, :].copy()                    
    t_obs    = forward_dict['prior_draws']['context']['t_obs'][:, :3, :]    
    cat0     = forward_dict['prior_draws']['context']['cat0_all'][:, :3] 
    inhib0   = forward_dict['prior_draws']['context']['inhib0_all'][:, :3] 
    theta    = forward_dict['prior_draws']['theta'].copy()

    B, _, T, _ = raw_data.shape

    # Extract P channel and set initial conditions
    P_data = raw_data[..., 1:2].copy() 
    P0 = np.zeros((B, N_OBS), dtype=np.float32)
    P_data[:, :, 0, 0] = P0 # Only 3 experiments are used to match real data structure

    is_t0 = np.zeros((B, N_OBS, T, 1), dtype=np.float32)
    is_t0[:, :, 0, :] = 1.0

    cat0_channel = np.broadcast_to(cat0[:, :, np.newaxis, np.newaxis], (B, N_OBS, T, 1)).astype(np.float32)
    inhib0_channel = np.broadcast_to(inhib0[:, :, np.newaxis, np.newaxis], (B, N_OBS, T, 1)).astype(np.float32)
    t_exp = t_obs[..., np.newaxis]                             
    
    # 5 Channels total: t, P, cat0, inhib0, is_t0
    data_full = np.concatenate([t_exp, P_data, cat0_channel, inhib0_channel, is_t0], axis=-1).astype(np.float32)

    # Temporal dropout (5 to 21 Observations) 
    data_dropped = np.zeros_like(data_full)
    data_dropped[..., 0] = -1.0  # Pad time with -1.0 so the mask ignores it
    
    for b in range(B):
        for n in range(N_OBS):
            # Randomly select between 5 and T (21) observations for this experiment
            T_keep = np.random.randint(5, T + 1)
            # Choose distinct indices and keep them sorted chronologically
            idx = np.sort(np.random.choice(T, T_keep, replace=False))
            # Pack the chosen data to the front
            data_dropped[b, n, :T_keep, :] = data_full[b, n, idx, :]
            
    data_full = data_dropped

    log_k = theta[..., :5]       
    noise_P_raw = theta[..., -1:] 

    k_scaled = (log_k - K_MEAN) / K_STD
    noise_scaled = (noise_P_raw - NOISE_MEAN) / NOISE_STD
    theta_scaled = np.concatenate([k_scaled, noise_scaled], axis=-1)
    
    return {
        'summary_variables':    data_full,
        'inference_variables':  theta_scaled.astype(np.float32),
    }

# Permutation invariant CNN_Transformer trunk
class HybridSummaryTrunk(keras.Model):
    def __init__(self, filters=(64, 128, 256), d_model=256, n_heads=4, n_obs=3, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_obs   = n_obs
        
        # Temporal Feature Extractor (1D CNN)
        self.cnn = keras.Sequential()
        for f in filters:
            self.cnn.add(layers.Conv1D(f, kernel_size=3, padding='same', activation='relu'))
            self.cnn.add(layers.LayerNormalization())
            
        self.obs_proj = layers.Dense(d_model, activation='relu')
        
        # Set Attention Block (unordered set)
        self.cross_attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout_rate)
        self.cross_norm = layers.LayerNormalization()
        self.cross_ffn  = keras.Sequential([
            layers.Dense(d_model * 2, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)
        ])
        self.cross_ffn_norm = layers.LayerNormalization()

    def call(self, x, training=False):
        # x shape: (Batch, N_observations, Time, Channels)
        B, N, T, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2], ops.shape(x)[3]
        
        # Extract time channel to identify padded steps (padded values are -1.0)
        t_channel  = x[..., 0]                                      
        valid_mask = ops.cast(t_channel >= 0.0, dtype=x.dtype)      
        flat_mask  = ops.reshape(valid_mask, (-1, T, 1))            
        
        # Flatten Batch and N_obs to process each observation's time series independently
        x_flat  = ops.reshape(x, (-1, T, C))                        
        cnn_out = self.cnn(x_flat, training=training)               
        
        # Mask out the dynamically dropped time steps 
        masked_cnn = cnn_out * flat_mask
        sum_features = ops.sum(masked_cnn, axis=1)                  
        
        # Masked Global Average Pooling across the Time dimension
        valid_counts = ops.maximum(ops.sum(flat_mask, axis=1), 1.0) 
        obs_emb = sum_features / valid_counts                       
        
        obs_emb = self.obs_proj(obs_emb)                            
        
        # Unflatten back to sets of observations: (Batch, N_obs, d_model)
        obs_emb = ops.reshape(obs_emb, (-1, self.n_obs, self.d_model)) 
        
        # Set Attention: Observations attend to each other to share context (Permutation Invariant)
        h = self.cross_attn(obs_emb, obs_emb, training=training)
        obs_emb = self.cross_norm(obs_emb + h)
        h = self.cross_ffn(obs_emb, training=training)
        obs_emb = self.cross_ffn_norm(obs_emb + h)
        
        # Final Mean Pooling across the N_obs dimension to yield a fixed-size representation
        return ops.mean(obs_emb, axis=1)

class JointSummaryHead(keras.Model):
    def __init__(self, shared_trunk, n_models=3, embedding_dim=256):
        super().__init__()
        self.shared_trunk = shared_trunk
        
        # Shared MLP processing the representation
        self.shared_body = keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
        ])
        
        # Dual Output Heads
        self.logits_head = layers.Dense(n_models) # For Model Selection
        self.embedding_head = layers.Dense(embedding_dim, activation='relu')  # For Parameter Inference 

    def call(self, x, training=False):
        trunk_out  = self.shared_trunk(x, training=training)
        body_out   = self.shared_body(trunk_out, training=training)
        
        logits     = self.logits_head(body_out, training=training)
        embedding  = self.embedding_head(body_out, training=training)
        
        return logits, embedding 

class JointNeuralInference(keras.Model):
    """
    Amortized Neural Inference Model optimizing both model selection 
    and parameter inference simultaneously.
    """
    def __init__(self, summary_head, flow_m14, flow_m15, flow_m16):
        super().__init__()
        self.summary_head = summary_head
        self.flow_m14 = flow_m14
        self.flow_m15 = flow_m15
        self.flow_m16 = flow_m16

    def call(self, inputs, training=False):
        # Default forward pass for inference (yielding Model Posteriors)
        m14_d = inputs['m14']
        logits, _ = self.summary_head(m14_d['summary_variables'], training=training)
        return ops.softmax(logits)   

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        # x contains balanced batches of data generated from M14, M15, and M16
        m14_d, m15_d, m16_d = x['m14'], x['m15'], x['m16']
        total_loss = 0.0

        for m_idx, m_d, flow in [
            (0, m14_d, self.flow_m14),
            (1, m15_d, self.flow_m15),
            (2, m16_d, self.flow_m16),
        ]:
            # Forward pass through the shared CNN-Transformer
            logits, embedding = self.summary_head(m_d['summary_variables'], training=True)
            
            # Cross-entropy loss for model selection P(M|x)
            log_probs_model = ops.log_softmax(logits)
            log_p_model     = log_probs_model[:, m_idx]  
            
            # Negative log-posterior for Parameter Inference P(theta|x, M)
            log_p_theta = flow.log_prob(m_d['inference_variables'], conditions=embedding, training=True)                                                            
            
            # Joint objective: Maximize log p(M, theta | x) = log p(M | x) + log p(theta | x, M)
            total_loss += -ops.mean(log_p_model + log_p_theta)

        # Return the average loss across all three mechanistic models
        return total_loss / 3.0

def joint_data_generator(sim14, sim15, sim16, batch_size):
    """
    Yields synchronized batches containing data from all three models. 
    This guarantees balanced classes for the categorical cross-entropy loss.
    """
    while True:
        yield ({
            'm14': crn_bayesflow_adapter(sim14.sample(batch_size), model_name="M14"),
            'm15': crn_bayesflow_adapter(sim15.sample(batch_size), model_name="M15"),
            'm16': crn_bayesflow_adapter(sim16.sample(batch_size), model_name="M16")
        },)

# Training
if __name__ == "__main__":
    start_time = time.time()
    
    print("Loading training data...")
    sim_m14 = SimulationDataLoader("M14_training_data")
    sim_m15 = SimulationDataLoader("M15_training_data")
    sim_m16 = SimulationDataLoader("M16_training_data")

    # 180 samples per model 
    BATCH_SIZE_PER_MODEL    = 180
    TOTAL_SAMPLES_PER_MODEL = 936000
    BATCHES_PER_EPOCH       = TOTAL_SAMPLES_PER_MODEL // BATCH_SIZE_PER_MODEL

    gen = joint_data_generator(sim_m14, sim_m15, sim_m16, batch_size=BATCH_SIZE_PER_MODEL)

    print("Building Hybrid Architecture...")
    shared_trunk = HybridSummaryTrunk(filters=(64, 128, 256), d_model=256, n_heads=4, n_obs=N_OBS)
    summary_head = JointSummaryHead(shared_trunk, n_models=3, embedding_dim=256)

    # Initialize BayesFlow cINNs
    # 6 targets for all models (5 kinetic parameters + 1 noise parameter)
    flow_m14 = bf.networks.CouplingFlow(num_params=6, num_coupling_layers=8, coupling_design='affine')
    flow_m15 = bf.networks.CouplingFlow(num_params=6, num_coupling_layers=8, coupling_design='affine')
    flow_m16 = bf.networks.CouplingFlow(num_params=6, num_coupling_layers=8, coupling_design='affine')

    print("Pre-building weights...")
    # A dummy batch is run to build the weights
    dummy_data = next(gen)[0]
    m14_d, m15_d, m16_d = dummy_data['m14'], dummy_data['m15'], dummy_data['m16']

    logits_d, emb_d = summary_head(m14_d['summary_variables'])
    _ = flow_m14(m14_d['inference_variables'], conditions=emb_d)
    
    logits_d, emb_d = summary_head(m15_d['summary_variables'])
    _ = flow_m15(m15_d['inference_variables'], conditions=emb_d)
    
    logits_d, emb_d = summary_head(m16_d['summary_variables'])
    _ = flow_m16(m16_d['inference_variables'], conditions=emb_d)

    print("Assembling Joint Model...")
    joint_model = JointNeuralInference(summary_head, flow_m14, flow_m15, flow_m16)
    
    # Run the dummy data 
    _ = joint_model(dummy_data)
    
    # Clipnorm prevents exploding gradients 
    joint_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0))

    print(f"Starting training ({BATCHES_PER_EPOCH} steps/epoch)...")
    history = joint_model.fit(x=gen, steps_per_epoch=BATCHES_PER_EPOCH, epochs=55)

    # Plot and save training diagnostics
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Joint NLL Loss', color='dodgerblue')
    plt.title("Joint Training: -log p(M, θ | x)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("M14_M16_JNPE_training_loss.png")

    # Save network weights individually. 
    with open("M14_M16_JNPE_summary_head.pkl", "wb") as f:
        pickle.dump(joint_model.summary_head.get_weights(), f)
    with open("M14_M16_JNPE_flow_m14.pkl", "wb") as f:
        pickle.dump(joint_model.flow_m14.get_weights(), f)
    with open("M14_M16_JNPE_flow_m15.pkl", "wb") as f:
        pickle.dump(joint_model.flow_m15.get_weights(), f)
    with open("M14_M16_JNPE_flow_m16.pkl", "wb") as f:
        pickle.dump(joint_model.flow_m16.get_weights(), f)

    print(f"Training finished in {time.time() - start_time:.2f} seconds!")
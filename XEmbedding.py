import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContinuousTimeEmbedding(nn.Module):
    """
    Learns a continuous representation of time intervals.
    Replaces fixed sin/cos encoding with a learnable MLP approach.
    """
    def __init__(self, time_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.LayerNorm(time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DeepStatisticalTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, time_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Feature Projection (Value + Mask)
        self.input_proj = nn.Linear(input_dim * 2, hidden_dim)
        
        # 2. Time Embedding
        self.time_embed = ContinuousTimeEmbedding(time_dim)
        
        # 3. Bi-Directional GRU
        # Bi-directionality allows the model to see 'future' context to determine 
        # if a point is a local max/min or part of a slope.
        # Input size: projected_features + time_embedding
        rnn_input_dim = hidden_dim + time_dim
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # 4. Attention Mechanism
        # Learns which time steps are most critical (Learning "Count" / relevance)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 5. Output Projection
        # We concatenate: 
        # - Last Hidden State (Forward) -> Like "Last"
        # - Last Hidden State (Backward) -> Context
        # - Max Pool over time -> Like "Max"
        # - Avg Pool over time -> Like "Mean"
        # - Attention Context -> Learned relevance
        
        concat_dim = (hidden_dim * 2) * 3  # (Last + Max + Avg) * bidirectional
        
        self.fc_mean = nn.Linear(concat_dim, latent_dim)
        self.fc_logstd = nn.Linear(concat_dim, latent_dim)

    def forward(self, batch_data, deterministic=False, temperature=1.0):
        # Unpack data
        times = batch_data['times'].to(DEVICE)   # [B, L]
        values = batch_data['values'].to(DEVICE) # [B, L, D]
        masks = batch_data['masks'].to(DEVICE)   # [B, L, D]
        lengths = batch_data['lengths'].to(DEVICE) # [B]
        
        B, L, D = values.size()

        # --- Preprocessing & Embeddings ---
        
        # 1. Calculate Delta Time (Time Gaps)
        # Pad beginning with 0
        prev_times = torch.cat([torch.zeros(B, 1).to(DEVICE), times[:, :-1]], dim=1)
        delta_t = (times - prev_times).unsqueeze(-1) # [B, L, 1]
        t_embed = self.time_embed(delta_t)           # [B, L, time_dim]

        # 2. Embed Values & Masks
        # Concatenate values and masks: [B, L, D*2]
        inputs = torch.cat([values, masks], dim=-1)
        x_embed = self.input_proj(inputs)            # [B, L, hidden_dim]

        # 3. Combine Time + Features
        rnn_input = torch.cat([x_embed, t_embed], dim=-1) # [B, L, hidden + time_dim]

        # --- Recurrent Processing ---
        
        # Pack sequence for efficiency and correct handling of padding in RNN
        packed_input = nn.utils.rnn.pack_padded_sequence(
            rnn_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Run Bi-GRU
        # outputs: [B, L, hidden_dim * 2] (contains all hidden states)
        packed_output, _ = self.rnn(packed_input)
        
        # Unpack
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=L
        )
        
        # Create a mask for valid timesteps (to ignore padding in pooling)
        # mask shape: [B, L, 1]
        mask_seq = torch.arange(L, device=DEVICE).expand(B, L) < lengths.unsqueeze(1)
        mask_seq = mask_seq.unsqueeze(-1).float()

        # Apply mask to outputs (zero out padded steps)
        rnn_outputs = rnn_outputs * mask_seq

        # --- Statistical Abstractions (The "Learnable" Part) ---
        
        # 1. Last State (Equivalent to "Last Value")
        # We need to grab the last valid time step for each batch
        # Gather indices: lengths - 1
        idx = (lengths - 1).view(-1, 1).expand(B, rnn_outputs.size(2)).unsqueeze(1)
        last_state = rnn_outputs.gather(1, idx).squeeze(1) # [B, hidden*2]

        # 2. Max Pooling (Equivalent to "Max" / "Peak Detection")
        # Mask padding with -inf before max
        masked_for_max = rnn_outputs.clone()
        masked_for_max[mask_seq.expand_as(rnn_outputs) == 0] = -1e9
        max_pool = torch.max(masked_for_max, dim=1)[0] # [B, hidden*2]

        # 3. Average Pooling (Equivalent to "Mean")
        sum_pool = torch.sum(rnn_outputs, dim=1)
        avg_pool = sum_pool / (lengths.unsqueeze(1).float() + 1e-6) # [B, hidden*2]

        # 4. Attention (Optional: Focus on specific critical events)
        attn_weights = self.attention(rnn_outputs) # [B, L, 1]
        attn_weights = attn_weights.masked_fill(mask_seq == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        context_vec = torch.sum(rnn_outputs * attn_weights, dim=1) # [B, hidden*2]

        # --- Concatenation ---
        # Combine all statistical views
        # Note: I used context_vec instead of simple avg_pool here, 
        # or you can use both. Let's use Last + Max + Context(Weighted Mean)
        # The latent Z will now contain structural info about peaks, averages, and latest trends.
        combined = torch.cat([last_state, max_pool, context_vec], dim=1)

        # --- Latent Projection ---
        mean = self.fc_mean(combined)
        log_std = self.fc_logstd(combined)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std) * temperature

        policy_dist = dist.Normal(mean, std)

        if deterministic:
            z = mean
            log_prob = None
        else:
            z = policy_dist.rsample()
            log_prob = policy_dist.log_prob(z).sum(dim=-1)

        return z, log_prob, mean
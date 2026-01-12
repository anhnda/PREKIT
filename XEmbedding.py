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
        rnn_input_dim = hidden_dim + time_dim
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # 4. Multi-Head Statistical Extractors
        # Each head focuses on a specific statistic

        # Attention for weighted mean
        self.attention_mean = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Attention for variance/std (focus on deviations)
        self.attention_std = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Trend/Slope detector
        self.trend_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )

        # Count/Density estimator (based on mask patterns)
        self.density_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )

        # 5. Output Projection
        # Concatenate ALL statistical views:
        # - Last (forward & backward)
        # - Max pooling
        # - Min pooling  [NEW]
        # - Mean (attention-weighted)
        # - Std estimation  [NEW]
        # - Trend/Slope  [NEW]
        # - Density/Count  [NEW]

        concat_dim = (hidden_dim * 2) * 7  # 7 statistical views * bidirectional

        self.fc_mean = nn.Linear(concat_dim, latent_dim)
        self.fc_logstd = nn.Linear(concat_dim, latent_dim)
        self.latent_dim = latent_dim

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
        idx = (lengths - 1).view(-1, 1).expand(B, rnn_outputs.size(2)).unsqueeze(1)
        last_state = rnn_outputs.gather(1, idx).squeeze(1)  # [B, hidden*2]

        # 2. Max Pooling (Equivalent to "Max")
        masked_for_max = rnn_outputs.clone()
        masked_for_max[mask_seq.expand_as(rnn_outputs) == 0] = -1e9
        max_pool = torch.max(masked_for_max, dim=1)[0]  # [B, hidden*2]

        # 3. Min Pooling (Equivalent to "Min") [NEW]
        masked_for_min = rnn_outputs.clone()
        masked_for_min[mask_seq.expand_as(rnn_outputs) == 0] = 1e9
        min_pool = torch.min(masked_for_min, dim=1)[0]  # [B, hidden*2]

        # 4. Mean via Attention (Equivalent to "Mean")
        attn_weights_mean = self.attention_mean(rnn_outputs)  # [B, L, 1]
        attn_weights_mean = attn_weights_mean.masked_fill(mask_seq == 0, -1e9)
        attn_weights_mean = F.softmax(attn_weights_mean, dim=1)
        mean_vec = torch.sum(rnn_outputs * attn_weights_mean, dim=1)  # [B, hidden*2]

        # 5. Std via Attention (Equivalent to "Std") [NEW]
        # Compute variance by attending to deviations from mean
        attn_weights_std = self.attention_std(rnn_outputs)  # [B, L, 1]
        attn_weights_std = attn_weights_std.masked_fill(mask_seq == 0, -1e9)
        attn_weights_std = F.softmax(attn_weights_std, dim=1)

        # Weighted variance computation
        deviations = rnn_outputs - mean_vec.unsqueeze(1)
        variance_vec = torch.sum(attn_weights_std * (deviations ** 2), dim=1)
        std_vec = torch.sqrt(variance_vec + 1e-6)  # [B, hidden*2]

        # 6. Trend/Slope Detection (Equivalent to "Slope") [NEW]
        # Compare first and last states with learned transformation
        first_state = rnn_outputs[:, 0, :]  # [B, hidden*2]
        trend_vec = self.trend_detector(last_state - first_state)  # [B, hidden*2]

        # 7. Density/Count Estimation (Equivalent to "Count") [NEW]
        # Based on average pooling with density transformation
        sum_pool = torch.sum(rnn_outputs * mask_seq, dim=1)
        avg_pool = sum_pool / (lengths.unsqueeze(1).float() + 1e-6)
        density_vec = self.density_estimator(avg_pool)  # [B, hidden*2]

        # --- Concatenation ---
        # Combine ALL 7 statistical views
        # This gives the model explicit access to all handcrafted statistics
        combined = torch.cat([
            last_state,   # Last
            max_pool,     # Max
            min_pool,     # Min [NEW]
            mean_vec,     # Mean
            std_vec,      # Std [NEW]
            trend_vec,    # Slope/Trend [NEW]
            density_vec   # Count/Density [NEW]
        ], dim=1)

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

    def extract_statistics(self, batch_data):
        """
        Extract learned statistics explicitly for auxiliary supervision.
        Returns a dictionary of learned statistics that can be compared to handcrafted features.
        """
        times = batch_data['times'].to(DEVICE)
        values = batch_data['values'].to(DEVICE)
        masks = batch_data['masks'].to(DEVICE)
        lengths = batch_data['lengths'].to(DEVICE)

        B, L, D = values.size()

        # Preprocessing
        prev_times = torch.cat([torch.zeros(B, 1).to(DEVICE), times[:, :-1]], dim=1)
        delta_t = (times - prev_times).unsqueeze(-1)
        t_embed = self.time_embed(delta_t)

        inputs = torch.cat([values, masks], dim=-1)
        x_embed = self.input_proj(inputs)
        rnn_input = torch.cat([x_embed, t_embed], dim=-1)

        # RNN processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            rnn_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.rnn(packed_input)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=L
        )

        mask_seq = torch.arange(L, device=DEVICE).expand(B, L) < lengths.unsqueeze(1)
        mask_seq = mask_seq.unsqueeze(-1).float()
        rnn_outputs = rnn_outputs * mask_seq

        # Extract all statistics
        idx = (lengths - 1).view(-1, 1).expand(B, rnn_outputs.size(2)).unsqueeze(1)
        last_state = rnn_outputs.gather(1, idx).squeeze(1)

        masked_for_max = rnn_outputs.clone()
        masked_for_max[mask_seq.expand_as(rnn_outputs) == 0] = -1e9
        max_pool = torch.max(masked_for_max, dim=1)[0]

        masked_for_min = rnn_outputs.clone()
        masked_for_min[mask_seq.expand_as(rnn_outputs) == 0] = 1e9
        min_pool = torch.min(masked_for_min, dim=1)[0]

        attn_weights_mean = self.attention_mean(rnn_outputs)
        attn_weights_mean = attn_weights_mean.masked_fill(mask_seq == 0, -1e9)
        attn_weights_mean = F.softmax(attn_weights_mean, dim=1)
        mean_vec = torch.sum(rnn_outputs * attn_weights_mean, dim=1)

        attn_weights_std = self.attention_std(rnn_outputs)
        attn_weights_std = attn_weights_std.masked_fill(mask_seq == 0, -1e9)
        attn_weights_std = F.softmax(attn_weights_std, dim=1)
        deviations = rnn_outputs - mean_vec.unsqueeze(1)
        variance_vec = torch.sum(attn_weights_std * (deviations ** 2), dim=1)
        std_vec = torch.sqrt(variance_vec + 1e-6)

        first_state = rnn_outputs[:, 0, :]
        trend_vec = self.trend_detector(last_state - first_state)

        sum_pool = torch.sum(rnn_outputs * mask_seq, dim=1)
        avg_pool = sum_pool / (lengths.unsqueeze(1).float() + 1e-6)
        density_vec = self.density_estimator(avg_pool)

        return {
            'last': last_state,
            'max': max_pool,
            'min': min_pool,
            'mean': mean_vec,
            'std': std_vec,
            'trend': trend_vec,
            'density': density_vec
        }
import torch
import torch.nn as nn

class MetaBlock(nn.Module):
    """
    Metadata Processing Block (MetaBlock).
    
    Modulates image feature groups using metadata via learned scale and shift.
    
    Args:
        num_feature_groups (V): number of feature groups (channel groups) the 
                                 image features are split into
        num_meta_features  (U): number of metadata input features
    
    Input:
        features  : [B, V, D] — image features split into V groups of size D
        meta_data : [B, U]    — metadata vector per sample
    
    Output:
        modulated features: [B, V, D]
    """
    def __init__(self, num_feature_groups, num_meta_features):
        super(MetaBlock, self).__init__()
        # Scale branch (f_b): maps metadata -> per-group scale factors
        self.scale_branch = nn.Sequential(
            nn.Linear(num_meta_features, num_feature_groups),
            nn.BatchNorm1d(num_feature_groups)
        )
        # Shift branch (g_b): maps metadata -> per-group shift factors
        self.shift_branch = nn.Sequential(
            nn.Linear(num_meta_features, num_feature_groups),
            nn.BatchNorm1d(num_feature_groups)
        )

    def forward(self, features, meta_data):
        """
        Args:
            features  : [B, V, D] image feature groups
            meta_data : [B, U]    metadata features
        Returns:
            modulated features: [B, V, D]
        """
        scale = self.scale_branch(meta_data)  # [B, V]
        shift = self.shift_branch(meta_data)  # [B, V]

        # unsqueeze to [B, V, 1] for broadcasting across the D dimension
        scale = scale.unsqueeze(-1)  # [B, V, 1]
        shift = shift.unsqueeze(-1)  # [B, V, 1]

        # Apply modulation: sigmoid(tanh(features * scale) + shift)
        return torch.sigmoid(torch.tanh(features * scale) + shift)  # [B, V, D]
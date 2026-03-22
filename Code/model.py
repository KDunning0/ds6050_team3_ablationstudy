from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import torch
import torch.nn as nn
from metablock import MetaBlock

class SkinEffnetB4(nn.Module):
    def __init__(self, pretrained=True, feature_extract=False, use_metablock=False, dropout_p = 0.0, num_classes=8, meta_num=15):
        super().__init__()
           
        ## Initialization 
        if pretrained: # define the EfficientNet model with DEFAULT weights; pretrained = True, feature_extract = False --> fine-tuning
            self.model = efficientnet_b4(weights = EfficientNet_B4_Weights.DEFAULT)
        else: # train the model from scratch by initializing weights to None; pretrained = False, feature_extract = False --> train from scratch
            self.model = efficientnet_b4(weights = None)

        # warn about nonsensical combination 
        if not pretrained and feature_extract:
            import warnings
            warnings.warn("feature_extract=True with pretrained=False freezes randomly-initialized weights, "
                            "which is not meaningful. Consider setting feature_extract=False.")
        
        # Freeze the model backbone for feature extraction mode
        if feature_extract: # pretrained = True, feature_extract = True --> feature extraction
            for param in self.model.features.parameters(): 
                param.requires_grad = False

        # Capture the number of features output by the backbone (1792 for B4)
        self.n_feat_conv = self.model.classifier[1].in_features 
        
        # Remove original, built-in classifier cleanly (necessary esp. if using MetaBlock prior to classifier to avoid risk of double-classification) 
        self.model.classifier = nn.Identity()

        # MetaBlock grouping
        self.comb_feat_maps = 32 # number of groups to split 1792 channels into before MetaBlock, must divide 1792; 32 metadata-controlled feature groups
        assert self.n_feat_conv % self.comb_feat_maps == 0, \
            f"comb_feat_maps ({self.comb_feat_maps}) must divide n_feat_conv ({self.n_feat_conv})"
        
        # Selects between Metablock and not Metablock
        self.use_metablock = use_metablock
        if self.use_metablock: 
            #  V is number of feature maps (i.e., number of channels) 
            #  U is number of metadata features (12 metadata features per sample - encodings for age, sex, lesion location) 
            #  MetaBlock modulates feature maps channel-wise using metadata
            #  Instead of using all channles directly, they are reshaped into "meta-aware" feature groups
            self.metablock = MetaBlock(num_feature_groups = self.comb_feat_maps, num_meta_features = meta_num)

        # Dropout applied to the flattened feature vector before classification.
        # dropout_p=0.0 disables dropout entirely, preserving default behaviour.
        # EfficientNet-B5's built-in classifier uses p=0.4 as a reference point.
        self.dropout = nn.Dropout(p=dropout_p)

        # redefine the final classifier
        self.classifier = nn.Linear(self.n_feat_conv, num_classes)

    def forward(self, img, meta_data = None):
        x = self.model.features(img) # [B, 1792, H, W]
        
        if self.use_metablock: # if using MetaBlock
            if meta_data is None:
                raise ValueError("meta_data must be provided when use_metablock=True")
                
            B, C, H, W = x.shape # [B, 1792, H, W]
            D = (C // self.comb_feat_maps) * H * W  # inner dim per group, 64 * H * W 
            
            x = x.view(B, self.comb_feat_maps, D) # reshape into feature groups, when comb_feat_maps = 32, x shape = [B, 32, 64 * H * W]
            
            # apply MetaBlock; each feature group gets metadata-conditioned scaling and shifting
            # x = sigmoid(tanh(x * scale) + shift)            
            x = self.metablock(x, meta_data.float()) # [B, 32, 64 * H * W]
            x = x.view(B, C, H, W) # restore shape, x shape = [B, 1792, H, W]
            
        x = nn.functional.adaptive_avg_pool2d(x, 1) # x shape = [B, 1792, 1, 1]
        x = torch.flatten(x, 1) # flatten for the classifer layer, x shape = [B, 1792]
        x = self.dropout(x)
        
        return self.classifier(x)
        
import torch.nn as nn
import argparse

# --- All the necessary imports ---
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):
    
    # --- This is the new, CORRECT __init__ ---
    def __init__(self, num_class, **kwargs):
        super(Model, self).__init__()
        
        # 1. Create a "fake" opt object
        opt = argparse.Namespace()
        
        # 2. Load all settings from the .yaml
        for key, value in kwargs.items():
            setattr(opt, key, value)
        
        # 3. Manually add num_class
        opt.num_class = 38  # Example: set to 38, change as needed
        
        # 4. Store this opt object
        self.opt = opt
        
        # --- NEW, CORRECTED MODEL BUILDING LOGIC ---
        
        """ Transformation """
        if opt.Transformation == 'TPS':
            # This is the correct call that fixes the TypeError
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, 
                I_size=(opt.imgH, opt.imgW), 
                I_r_size=(opt.imgH, opt.imgW), 
                I_channel_num=opt.input_channel
            )
        else:
            self.Transformation = nn.Identity() # No transformation
        
        """ FeatureExtraction """
        if opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction Module')
        
        self.FeatureExtraction_output = self.opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # This layer is part of the original model
        
        """ SequenceModeling """
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            self.SequenceModeling = nn.Identity() # No sequence modeling
            self.SequenceModeling_output = self.FeatureExtraction_output
            
        """ Prediction """
        if opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)

    # --- This is the original, CORRECT forward method ---
    def forward(self, input, text, is_train=False):
        """ Transformation stage """
        if not self.opt.Transformation == 'None':
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        # This was the missing step in some previous versions
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if not self.opt.SequenceModeling == 'None':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature # for convenience

        """ Prediction stage """
        if not self.opt.Prediction == 'None':
            # The 'batch_max_length' needs to be in opt
            if 'batch_max_length' not in self.opt:
                # Add a default value if it's missing (it shouldn't be, but this is safe)
                self.opt.batch_max_length = 25 
            
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)
        else:
            prediction = contextual_feature

        return prediction
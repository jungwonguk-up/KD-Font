import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):

    def __init__(self, 
                 Transformation: str, 
                 FeatureExtraction: str, 
                 SequenceModeling: str, 
                 Prediction: str,
                 num_fiducial,
                 img_scale: tuple, # (H, W)
                 input_channel: int,
                 output_channel: int,
                 hidden_size,
                 num_class,
                 batch_max_length):
        super(Model, self).__init__()
        self.Transformation_value = Transformation
        self.FeatureExtraction_value = FeatureExtraction
        self.SequenceModeling_value = SequenceModeling
        self.Prediction_value = Prediction
        self.stages = {'Trans': self.Transformation_value, 
                       'Feat': self.FeatureExtraction_value,
                       'Seq': self.SequenceModeling_value, 
                       'Pred': self.Prediction_value}
        
        self.num_fiducial = num_fiducial
        self.img_scale = img_scale
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.batch_max_length = batch_max_length
        
        """ Transformation """
        if self.Transformation_value == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=self.num_fiducial, I_size=self.img_scale, I_r_size=self.img_scale, I_channel_num=self.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if self.FeatureExtraction_value == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(self.input_channel, self.output_channel)
        elif self.FeatureExtraction_value == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(self.input_channel, self.output_channel)
        elif self.FeatureExtraction_value == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(self.input_channel, self.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = self.output_channel # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1)) # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if self.SequenceModeling_value == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, self.hidden_size, self.hidden_size),
                BidirectionalLSTM(self.hidden_size, self.hidden_size, self.hidden_size))
            self.SequenceModeling_output = self.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if self.Prediction_value == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, self.num_class)
        elif self.Prediction_value == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, self.hidden_size, self.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.batch_max_length)

        return prediction
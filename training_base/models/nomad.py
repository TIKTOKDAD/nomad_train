import torch.nn as nn


class NoMaD(nn.Module):
    """NoMaD model composed from explicit navigation modules."""

    def __init__(self, vision_encoder, diffusion_model, distance_predictor) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.diffusion_model = diffusion_model
        self.distance_predictor = distance_predictor

    def encode_vision(self, obs_img, goal_img, goal_mask):
        return self.vision_encoder(obs_img, goal_img, input_goal_mask=goal_mask)

    def predict_noise(self, sample, timestep, global_cond):
        return self.diffusion_model(sample=sample, timestep=timestep, global_cond=global_cond)

    def predict_distance(self, obsgoal_cond):
        return self.distance_predictor(obsgoal_cond)

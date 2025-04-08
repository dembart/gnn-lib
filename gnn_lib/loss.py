from torch import nn

class EFSLoss(nn.Module):

    def __init__(self, energy_weight = 1.0, forces_weight=0.0, stress_weight=0.0):
        super(EFSLoss, self).__init__()
        self.energy_weight = energy_weight
        self.forces_weight = forces_weight
        self.stress_weight = stress_weight
        self.mse = nn.MSELoss() 

    def forward(self, e_pred, e_target, f_pred=None, f_target=None, s_pred=None, s_target=None):
        energy_loss = self.mse(e_pred, e_target)
        total_loss = self.energy_weight * energy_loss
        forces_loss = None
        stress_loss = None

        if f_pred is not None and f_target is not None:
            forces_loss = self.mse(f_pred, f_target)
            total_loss += self.forces_weight * forces_loss

        if s_pred is not None and s_target is not None:
            stress_loss = self.mse(s_pred, s_target)
            total_loss += self.stress_weight * stress_loss

        return energy_loss, forces_loss, stress_loss, total_loss
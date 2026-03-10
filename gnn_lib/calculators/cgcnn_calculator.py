import torch
from pathlib import Path
from ase.calculators.calculator import Calculator, all_changes

from ..graph_data import atoms2data
from ..data_utils import full_3x3_to_voigt_6_stress, get_cached_model
from ..models.cgcnn import CrystalGraphConvNet
from .utils import get_cached_model



class CrystalGraphConvNetCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model, device="cuda"):
        
        """
        Parameters
        ----------

        model: str or nn.Module
            can be:
              - "matpes_pbe_small"
              - "/path/to/checkpoint.pt"
              - already initialized PyTorch model
        device: str, 'cuda' by default
            device, can be 'cuda' or 'cpu'
        """
        super().__init__()
        self.device = device

        if model == "matpes_pbe_small":
            checkpoint_path = get_cached_model(
                model_name=model,
                url="place_holder"
            )

        elif isinstance(model, str):
            checkpoint_path = Path(model)

        elif isinstance(model, torch.nn.Module):
            self.model = model.to(device)
            self.model.eval()
            self.config = getattr(model, "config", None)
            return

        else:
            raise ValueError("model must be a string (checkpoint path) or nn.Module")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        self.config = checkpoint["config"]

        self.model = CrystalGraphConvNet(**self.config.model).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        properties = properties or self.implemented_properties
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        n_max_neighbors = (
            self.config.data.n_max_neighbors
            if self.config.data.n_max_neighbors
            else None
        )

        data = atoms2data(
            atoms,
            r_cut=self.config.data.r_cut,
            n_max_neighbors=n_max_neighbors,
        )
        data = data.to(self.device)    

        with torch.no_grad():
            out = self.model(data)
            energy = out["energy"]
            forces = out.get("forces", None)
            stress = out.get("stress", None)

        if getattr(self.model, "ref_energies", None) is not None:
            energy = self.model._energy_from_e0s_and_residuals(energy, data)

        self.results["energy"] = (energy.squeeze().detach().cpu().numpy() * len(atoms))

        if forces is not None:
            self.results["forces"] = forces.detach().cpu().numpy()

        if stress is not None:
            stress_np = full_3x3_to_voigt_6_stress(stress).detach().cpu().numpy().reshape(-1)
            self.results["stress"] = stress_np
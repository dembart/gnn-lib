import torch

# from https://github.com/ACEsuit/mace/blob/main/mace/modules/utils.py
def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: torch.Tensor | None,
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement

# from MACE github
def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    grad_outputs = [torch.ones_like(energy)]
    
    if compute_stress:
        forces, virials = torch.autograd.grad(
            outputs=[energy],
            inputs=[positions, displacement],
            grad_outputs=grad_outputs,
            retain_graph=training,
            create_graph=training,
            allow_unused=True,
        )
    else:
        forces = torch.autograd.grad(
            outputs=[energy],
            inputs=[positions],
            grad_outputs=grad_outputs,
            retain_graph=training,
            create_graph=training,
            allow_unused=True,
        )[0]
        virials = None
    
    stress = None
    if compute_stress and virials is not None:
        cell_reshaped = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell_reshaped).abs().view(-1,1,1)
        stress = -virials / volume
        stress = torch.clamp(stress, -1e10, 1e10)
    else:
        virials = None
    
    forces = torch.zeros_like(positions) if forces is None else -forces
    
    return forces, virials, stress
import json
from pathlib import Path
import wandb
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .loss import EFSLoss
from .data_utils import Normalizer



class CSVLogger:
    
    def __init__(self, log_dir, filename = "training_log.csv"):
        self.log_path = Path(log_dir) / filename
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = False
        
    def log(self, metrics, step):
        metrics["step"] = step
        if not self.header_written:
            with open(self.log_path, "w") as f:
                f.write(",".join(metrics.keys()) + "\n")
            self.header_written = True
        with open(self.log_path, "a") as f:
            f.write(",".join(map(str, metrics.values())) + "\n")



class WandbLogger:
    def __init__(self, config):
        wandb_config = config.get("logging", {})
        wandb.init(
            project=wandb_config.get("wandb_project", "my_project"),
            entity=wandb_config.get("wandb_entity", None),
            name=wandb_config.get("run_name", None),
            config=config
        )

    def log(self, metrics, step):
        wandb.log(metrics, step=step)

    def close(self):
        """Mandatory cleanup to avoid BrokenPipeError."""
        if wandb.run is not None:
            wandb.finish()



class Trainer:

    def __init__(self, model, config, verbose = True):

        self.device = config['device']
        self.config = config
        self.verbose = verbose
        self.model = model.to(self.device)
        self.current_step = 0
        self.best_metric = float("inf")
        self.normalizer = None
        self.wandb_logger = None

        if self.config['data']['normalize_energy']:
            self.normalizer = Normalizer()
            file = 'normalizer_params.json'
            file = Path(self.config['data']['processed_data_path']) / file
            with open(file, 'r') as f:
                params = json.load(f)
            self.normalizer.load_state_dict(params )
        
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logger()
        self._setup_loss_fn()
        
    def _setup_optimizer(self):
        optimizer_config = self.config["training"]["optimizer"]
        self.optimizer = getattr(torch.optim, optimizer_config["type"])(
            self.model.parameters(),
            **optimizer_config["params"]
        )
    
    def _setup_scheduler(self):
        scheduler_config = self.config["training"].get("scheduler", None)
        if scheduler_config:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                **scheduler_config["params"]
            )
        else:
            self.scheduler = None
    
    def _setup_logger(self):
        log_config = self.config["logging"]
        self.logger = CSVLogger(
            log_dir=log_config["log_dir"],
            filename=log_config.get("filename", "training_log.csv")
        )
    
        if log_config.get("use_wandb", False):
            self.wandb_logger = WandbLogger(self.config)

    def _setup_loss_fn(self):
        loss_config = self.config["loss"]
        self.loss_fn = EFSLoss(**loss_config['params'])
    
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        batch.to(self.device)
        outputs = self.model(batch)
        labels = batch.energy
        if self.normalizer is not None:
            labels = self.normalizer.norm(labels)
        energy_loss, forces_loss, stress_loss, total_loss = self.loss_fn(outputs, labels.view(-1, 1))
        total_loss.backward()
        self.optimizer.step()
        return {"train_total_loss": total_loss.item(),
                "train_energy_loss": energy_loss.item(),
                "train_forces_loss": 0 if forces_loss is None else forces_loss.item(),
                "train_stress_loss": 0 if stress_loss is None else stress_loss.item(),
                "lr": self.optimizer.param_groups[0]['lr']
                 }
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        energy_loss = 0
        forces_loss = 0
        stress_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch.to(self.device)
                outputs = self.model(batch)
                labels = batch.energy
                if self.normalizer is not None:
                    labels = self.normalizer.norm(labels)
                loss = self.loss_fn(outputs, labels.view(-1, 1))
                total_loss+= loss[-1].item()
                energy_loss+= loss[0].item()
                if loss[1] is not None:
                    forces_loss+= loss[1].item()
                if loss[2] is not None:
                    stress_loss+= loss[2].item()

        return {"val_total_loss": total_loss / len(dataloader),
                "val_energy_loss": energy_loss / len(dataloader),
                "val_forces_loss": forces_loss / len(dataloader),
                "val_stress_loss": stress_loss / len(dataloader),
                 }
    
    def save_checkpoint(self, filename):
        checkpoint = {
            "step": self.current_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config
        }
        
        save_path = Path(self.config["logging"]["checkpoint_dir"]) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
    

    def train(self, train_loader, val_loader):
        checkpoint_config = self.config["logging"]
        try:
            for epoch in range(1, self.config["training"]["num_epochs"]):
                
                epoch_train_metrics = []
                for batch in train_loader:
                    self.current_step += 1
                    step_metrics = self.train_step(batch)
                    epoch_train_metrics.append(step_metrics)
                
                avg_train_metrics = {
                    k: sum(m[k] for m in epoch_train_metrics) / len(epoch_train_metrics)
                    for k in epoch_train_metrics[0].keys()
                }
                
                val_metrics = self.evaluate(val_loader)
                combined_metrics = {
                    "epoch": epoch,
                    **{"" + k: v for k, v in avg_train_metrics.items()},
                    **{"" + k: v for k, v in val_metrics.items()}
                }
                
                self.logger.log(combined_metrics, self.current_step)
                if self.wandb_logger is not None:
                    self.wandb_logger.log(combined_metrics, self.current_step)

                print(", ".join(
                    f"{k}: {int(v):d}" if k in {'epoch', 'step'} else f"{k}: {v:.4f}"
                    for k, v in combined_metrics.items()
                ))

                self.combined_metrics = combined_metrics

                current_metric = val_metrics[checkpoint_config["metric"]]
                is_best = (current_metric < self.best_metric) 
            
                self.save_checkpoint(f"last_checkpoint.pt")

                if epoch % checkpoint_config["checkpoint_freq"] == 0:
                    self.save_checkpoint(f"epoch_{epoch}_checkpoint.pt")
                
                if is_best:
                    self.best_metric = current_metric
                    self.save_checkpoint(f"best_checkpoint.pt")
            
                if self.scheduler:
                    self.scheduler.step(current_metric)

        except KeyboardInterrupt:
            print("Interrupted!")
        finally:
            if self.wandb_logger is not None:
                self.wandb_logger.close()
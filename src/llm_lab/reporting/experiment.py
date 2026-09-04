import json
import time
from pathlib import Path
from typing import Dict, Any

class ExperimentTracker:
    def __init__(self, experiment_id: str, output_dir: str):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir) / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = {
            "experiment_id": self.experiment_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": "",
            "dataset": "",
            "hyperparameters": {},
            "metrics": {},
            "hardware": {
                "peak_vram_gb": 0.0,
                "peak_ram_gb": 0.0
            },
            "training": {
                "wall_hours": 0.0,
                "gpu_hours": 0.0,
                "tokens": 0
            }
        }
        
    def set_config(self, model: str, dataset: str, hyperparameters: Dict[str, Any]):
        self.data["model"] = model
        self.data["dataset"] = dataset
        self.data["hyperparameters"] = hyperparameters
        
    def update_metrics(self, metrics: Dict[str, float]):
        self.data["metrics"].update(metrics)
        
    def update_hardware(self, peak_vram: float, peak_ram: float):
        self.data["hardware"]["peak_vram_gb"] = max(self.data["hardware"]["peak_vram_gb"], peak_vram)
        self.data["hardware"]["peak_ram_gb"] = max(self.data["hardware"]["peak_ram_gb"], peak_ram)
        
    def save(self):
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
            
    def append_task_result(self, result: Dict[str, Any]):
        results_path = self.output_dir / "generations.jsonl"
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

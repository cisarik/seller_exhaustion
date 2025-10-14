"""
Parameter persistence system for storing and loading evolved strategy parameters.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict
from strategy.seller_exhaustion import SellerParams
from backtest.engine import BacktestParams


class ParamsStore:
    """
    Manages saving and loading of strategy and backtest parameters.
    
    Stores:
    - Strategy parameters (SellerParams)
    - Backtest parameters (BacktestParams)
    - Metadata (date range, timeframe, generation, fitness)
    - Notes (evolution history, performance metrics)
    """
    
    def __init__(self, storage_dir: str = ".strategy_params"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
    
    def save_params(
        self,
        seller_params: SellerParams,
        backtest_params: BacktestParams,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Save parameter set to JSON file.
        
        Args:
            seller_params: Strategy parameters
            backtest_params: Backtest parameters
            metadata: Additional metadata (date range, fitness, etc.)
            name: Optional custom name (default: timestamp)
        
        Returns:
            Path to saved file
        """
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filepath = self.storage_dir / f"{name}.json"
        
        data = {
            "saved_at": datetime.now().isoformat(),
            "seller_params": asdict(seller_params),
            "backtest_params": backtest_params.model_dump(),
            "metadata": metadata or {}
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def load_params(self, name: str) -> Dict[str, Any]:
        """
        Load parameter set from JSON file.
        
        Args:
            name: Filename (with or without .json extension)
        
        Returns:
            Dict with keys: seller_params, backtest_params, metadata
        """
        if not name.endswith(".json"):
            name = f"{name}.json"
        
        filepath = self.storage_dir / name
        
        if not filepath.exists():
            raise FileNotFoundError(f"Parameter file not found: {filepath}")
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Reconstruct objects
        seller_params = SellerParams(**data["seller_params"])
        backtest_params = BacktestParams(**data["backtest_params"])
        
        return {
            "seller_params": seller_params,
            "backtest_params": backtest_params,
            "metadata": data.get("metadata", {}),
            "saved_at": data.get("saved_at")
        }
    
    def list_saved_params(self) -> list[Dict[str, Any]]:
        """
        List all saved parameter sets with metadata.
        
        Returns:
            List of dicts with: name, saved_at, metadata
        """
        results = []
        
        for filepath in sorted(self.storage_dir.glob("*.json"), reverse=True):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                results.append({
                    "name": filepath.stem,
                    "saved_at": data.get("saved_at", "unknown"),
                    "metadata": data.get("metadata", {})
                })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        
        return results
    
    def save_generation(
        self,
        generation: int,
        population: list[Dict[str, Any]],
        best_fitness: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save entire generation from GA optimization.
        
        Args:
            generation: Generation number
            population: List of individuals with params and fitness
            best_fitness: Best fitness in this generation
            metadata: Additional metadata
        
        Returns:
            Path to saved file
        """
        name = f"gen_{generation:04d}"
        filepath = self.storage_dir / f"{name}.yaml"
        
        data = {
            "generation": generation,
            "saved_at": datetime.now().isoformat(),
            "best_fitness": best_fitness,
            "population_size": len(population),
            "population": population,
            "metadata": metadata or {}
        }
        
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        
        return str(filepath)
    
    def export_to_yaml(
        self,
        seller_params: SellerParams,
        backtest_params: BacktestParams,
        name: str
    ) -> str:
        """
        Export parameters to human-readable YAML format.
        
        Args:
            seller_params: Strategy parameters
            backtest_params: Backtest parameters
            name: Output filename (without extension)
        
        Returns:
            Path to saved file
        """
        filepath = self.storage_dir / f"{name}.yaml"
        
        data = {
            "saved_at": datetime.now().isoformat(),
            "strategy": {
                "name": "Seller Exhaustion",
                "params": asdict(seller_params)
            },
            "backtest": {
                "params": backtest_params.model_dump()
            }
        }
        
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        return str(filepath)


# Global instance
params_store = ParamsStore()

"""
GPU Memory Manager - Query GPU specs and optimize parameters.

Automatically detects GPU capabilities and suggests optimal batch sizes
to maximize VRAM utilization.
"""

import torch
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """GPU hardware information."""
    available: bool
    device_name: str
    total_vram_gb: float
    compute_capability: str
    multiprocessors: int
    cuda_version: str


class GPUMemoryManager:
    """
    Manages GPU memory and suggests optimal batch sizes.
    
    For RTX 3080 (10GB VRAM):
    - Can process 150+ individuals simultaneously
    - Optimal batch size depends on data size
    - Dynamic sizing based on available memory
    """
    
    def __init__(self):
        self.available = torch.cuda.is_available()
        
        if self.available:
            self.device = torch.device('cuda:0')
            props = torch.cuda.get_device_properties(0)
            
            self.device_name = torch.cuda.get_device_name(0)
            self.total_vram = props.total_memory
            self.compute_capability = f"{props.major}.{props.minor}"
            self.multiprocessors = props.multi_processor_count
            self.cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else 'Unknown'
        else:
            self.device = torch.device('cpu')
            self.device_name = 'CPU'
            self.total_vram = 0
            self.compute_capability = 'N/A'
            self.multiprocessors = 0
            self.cuda_version = 'N/A'
    
    def get_info(self) -> GPUInfo:
        """Get comprehensive GPU information."""
        return GPUInfo(
            available=self.available,
            device_name=self.device_name,
            total_vram_gb=self.total_vram / 1e9 if self.available else 0.0,
            compute_capability=self.compute_capability,
            multiprocessors=self.multiprocessors,
            cuda_version=self.cuda_version
        )
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dict with allocated, reserved, total, and free memory in GB
        """
        if not self.available:
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'total_gb': 0.0,
                'free_gb': 0.0,
                'utilization': 0.0
            }
        
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        total = self.total_vram
        free = total - allocated
        
        return {
            'allocated_gb': allocated / 1e9,
            'reserved_gb': reserved / 1e9,
            'total_gb': total / 1e9,
            'free_gb': free / 1e9,
            'utilization': allocated / total if total > 0 else 0.0
        }
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if self.available:
            torch.cuda.empty_cache()
    
    def estimate_memory_per_individual(self, data_size: int) -> int:
        """
        Estimate VRAM required per individual.
        
        Args:
            data_size: Number of bars in dataset
        
        Returns:
            Estimated bytes per individual
        """
        # OHLCV data: 5 arrays × data_size × 4 bytes (float32)
        data_mem = data_size * 5 * 4
        
        # Indicators: ~10 indicator arrays
        indicator_mem = data_size * 10 * 4
        
        # Intermediate calculations
        intermediate_mem = data_size * 5 * 4
        
        # PyTorch overhead per tensor
        overhead = 1024 * 100  # ~100KB overhead
        
        total = data_mem + indicator_mem + intermediate_mem + overhead
        
        return total
    
    def get_optimal_batch_size(
        self,
        data_size: int,
        safety_margin: float = 0.85,
        min_batch: int = 1,
        max_batch: int = 500
    ) -> int:
        """
        Calculate optimal batch size to maximize GPU utilization.
        
        Args:
            data_size: Number of bars in dataset
            safety_margin: Use only this fraction of available memory (0-1)
            min_batch: Minimum batch size
            max_batch: Maximum batch size
        
        Returns:
            Optimal batch size for GPU
        """
        if not self.available:
            return min_batch
        
        # Get current free memory
        mem_info = self.get_memory_usage()
        available_vram = mem_info['free_gb'] * 1e9  # Convert to bytes
        
        # Estimate memory per individual
        per_individual = self.estimate_memory_per_individual(data_size)
        
        # Calculate batch size with safety margin
        raw_batch = int((available_vram * safety_margin) / per_individual)
        
        # Clamp to reasonable range
        batch_size = max(min_batch, min(raw_batch, max_batch))
        
        return batch_size
    
    def suggest_population_size(self, data_size: int) -> int:
        """
        Suggest optimal population size based on available VRAM.
        
        This rounds to nice numbers for user-friendly settings.
        
        Args:
            data_size: Number of bars in dataset
        
        Returns:
            Suggested population size
        """
        if not self.available:
            return 24  # Default CPU size
        
        # Get raw optimal batch size
        batch_size = self.get_optimal_batch_size(data_size)
        
        # Round to nice numbers
        if batch_size >= 200:
            # Round to nearest 50
            return (batch_size // 50) * 50
        elif batch_size >= 100:
            # Round to nearest 20
            return (batch_size // 20) * 20
        elif batch_size >= 50:
            # Round to nearest 10
            return (batch_size // 10) * 10
        elif batch_size >= 20:
            # Round to nearest 5
            return (batch_size // 5) * 5
        else:
            return batch_size
    
    def estimate_speedup(
        self,
        population_size: int,
        data_size: int
    ) -> float:
        """
        Estimate GPU speedup factor vs CPU.
        
        Args:
            population_size: Number of individuals
            data_size: Number of bars
        
        Returns:
            Expected speedup factor (e.g., 10.0 = 10x faster)
        """
        if not self.available:
            return 1.0
        
        # Base speedup from GPU parallelism
        # RTX 3080 has 8704 CUDA cores, can handle massive parallelism
        base_speedup = 5.0  # Conservative estimate
        
        # More speedup with larger populations (better GPU utilization)
        population_factor = min(population_size / 50, 3.0)
        
        # More speedup with larger datasets (amortize overhead)
        data_factor = min(data_size / 5000, 1.5)
        
        total_speedup = base_speedup * population_factor * data_factor
        
        return min(total_speedup, 50.0)  # Cap at 50x
    
    def get_recommendations(self, data_size: int) -> Dict[str, any]:
        """
        Get comprehensive recommendations for GPU optimization.
        
        Args:
            data_size: Number of bars in dataset
        
        Returns:
            Dict with recommendations
        """
        if not self.available:
            return {
                'available': False,
                'message': 'GPU not available. Using CPU mode (slower).',
                'recommendations': [
                    'Install PyTorch with CUDA support for GPU acceleration',
                    'Reduce population size to 24 or less for CPU performance'
                ]
            }
        
        mem_info = self.get_memory_usage()
        suggested_pop = self.suggest_population_size(data_size)
        estimated_speedup = self.estimate_speedup(suggested_pop, data_size)
        
        recommendations = []
        
        # Memory utilization
        if mem_info['utilization'] > 0.9:
            recommendations.append(
                f"⚠️ High VRAM usage ({mem_info['utilization']:.1%}). "
                "Consider reducing population size or clearing cache."
            )
        elif mem_info['utilization'] < 0.5:
            recommendations.append(
                f"💡 Low VRAM usage ({mem_info['utilization']:.1%}). "
                "You can increase population size for better exploration!"
            )
        
        # Population size
        recommendations.append(
            f"📊 Suggested population size: {suggested_pop} individuals"
        )
        
        # Expected performance
        recommendations.append(
            f"🚀 Expected speedup: {estimated_speedup:.1f}x faster than CPU"
        )
        
        # Batch size info
        optimal_batch = self.get_optimal_batch_size(data_size)
        recommendations.append(
            f"⚙️ Optimal batch size: {optimal_batch} individuals per batch"
        )
        
        return {
            'available': True,
            'device': self.device_name,
            'memory': mem_info,
            'suggested_population': suggested_pop,
            'optimal_batch': optimal_batch,
            'estimated_speedup': estimated_speedup,
            'recommendations': recommendations
        }
    
    def print_info(self):
        """Print comprehensive GPU information."""
        info = self.get_info()
        
        print("\n" + "="*60)
        print("GPU Information")
        print("="*60)
        
        if info.available:
            print(f"✓ GPU Available: {info.device_name}")
            print(f"  Total VRAM: {info.total_vram_gb:.2f} GB")
            print(f"  Compute Capability: {info.compute_capability}")
            print(f"  Multiprocessors: {info.multiprocessors}")
            print(f"  CUDA Version: {info.cuda_version}")
            
            mem = self.get_memory_usage()
            print(f"\n  Current Memory Usage:")
            print(f"    Allocated: {mem['allocated_gb']:.2f} GB ({mem['utilization']:.1%})")
            print(f"    Reserved: {mem['reserved_gb']:.2f} GB")
            print(f"    Free: {mem['free_gb']:.2f} GB")
        else:
            print("❌ GPU not available")
            print("   Using CPU mode (slower)")
        
        print("="*60 + "\n")
    
    def print_recommendations(self, data_size: int):
        """Print optimization recommendations."""
        recs = self.get_recommendations(data_size)
        
        print("\n" + "="*60)
        print("GPU Optimization Recommendations")
        print("="*60)
        
        if recs['available']:
            print(f"Device: {recs['device']}")
            print(f"VRAM: {recs['memory']['free_gb']:.2f} GB free / "
                  f"{recs['memory']['total_gb']:.2f} GB total\n")
            
            for rec in recs['recommendations']:
                print(f"  {rec}")
        else:
            print(f"Status: {recs['message']}\n")
            for rec in recs['recommendations']:
                print(f"  • {rec}")
        
        print("="*60 + "\n")


# Global instance
_gpu_manager = None

def get_gpu_manager() -> GPUMemoryManager:
    """Get global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUMemoryManager()
    return _gpu_manager


# Convenience functions
def get_gpu_info() -> GPUInfo:
    """Get GPU information."""
    return get_gpu_manager().get_info()


def get_optimal_batch_size(data_size: int) -> int:
    """Get optimal batch size for given data size."""
    return get_gpu_manager().get_optimal_batch_size(data_size)


def suggest_population_size(data_size: int) -> int:
    """Suggest optimal population size for given data size."""
    return get_gpu_manager().suggest_population_size(data_size)


def print_gpu_info():
    """Print GPU information to console."""
    get_gpu_manager().print_info()


def print_recommendations(data_size: int):
    """Print optimization recommendations to console."""
    get_gpu_manager().print_recommendations(data_size)


if __name__ == '__main__':
    """Test GPU manager."""
    manager = GPUMemoryManager()
    
    # Print info
    manager.print_info()
    
    # Test with different data sizes
    for data_size in [1000, 5000, 10000, 50000]:
        print(f"\nData size: {data_size} bars")
        manager.print_recommendations(data_size)

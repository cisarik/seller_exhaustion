"""Agent Feed - Evolution data storage for agent-based optimization.

Stores important evolution information in a structured format without UI dependencies.
Data can be queried, exported, and analyzed for agent-based workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
from threading import Lock


@dataclass
class GenerationRecord:
    """Record of a single generation's evolution."""
    
    generation: int
    timestamp: datetime
    
    # Population metrics
    population_size: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float
    
    # Best individual
    best_params: Dict[str, Any]
    best_metrics: Dict[str, Any]
    
    # Optional coach analysis
    coach_triggered: bool = False
    coach_recommendations_count: int = 0
    mutations_applied: int = 0
    ga_changes_applied: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class CoachAnalysisRecord:
    """Record of a coach analysis event."""
    
    generation: int
    timestamp: datetime
    session_id: str
    
    # Analysis inputs
    population_size: int
    diversity: float
    
    # Analysis outputs
    overall_assessment: str
    recommendations_count: int
    recommendations: List[Dict[str, Any]]
    
    # Actions taken
    mutations_applied: int
    ga_params_changed: int
    fitness_gates_changed: int
    
    # Timing
    analysis_duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AgentFeed:
    """
    Structured storage for evolution data without UI dependencies.
    
    Thread-safe storage of:
    - Generation-by-generation evolution history
    - Coach analysis events and recommendations
    - Best individuals over time
    - Population statistics
    
    Data can be queried, filtered, and exported for analysis.
    """
    
    def __init__(self, max_generations: int = 1000):
        """
        Initialize agent feed.
        
        Args:
            max_generations: Maximum generations to keep in memory
        """
        self.max_generations = max_generations
        self._generations: List[GenerationRecord] = []
        self._coach_analyses: List[CoachAnalysisRecord] = []
        self._lock = Lock()
        
        # Track best ever across all generations
        self._best_ever_fitness: float = float('-inf')
        self._best_ever_generation: Optional[int] = None
        self._best_ever_params: Optional[Dict[str, Any]] = None
    
    def record_generation(
        self,
        generation: int,
        population_size: int,
        best_fitness: float,
        mean_fitness: float,
        worst_fitness: float,
        diversity: float,
        best_params: Dict[str, Any],
        best_metrics: Dict[str, Any],
        coach_triggered: bool = False,
        coach_recommendations_count: int = 0,
        mutations_applied: int = 0,
        ga_changes_applied: int = 0
    ) -> None:
        """
        Record a generation's evolution data.
        
        Args:
            generation: Generation number
            population_size: Number of individuals
            best_fitness: Best fitness in generation
            mean_fitness: Mean fitness
            worst_fitness: Worst fitness
            diversity: Population diversity metric
            best_params: Best individual's parameters
            best_metrics: Best individual's metrics
            coach_triggered: Whether coach was triggered this generation
            coach_recommendations_count: Number of coach recommendations
            mutations_applied: Number of mutations applied
            ga_changes_applied: Number of GA parameter changes
        """
        with self._lock:
            record = GenerationRecord(
                generation=generation,
                timestamp=datetime.utcnow(),
                population_size=population_size,
                best_fitness=best_fitness,
                mean_fitness=mean_fitness,
                worst_fitness=worst_fitness,
                diversity=diversity,
                best_params=best_params,
                best_metrics=best_metrics,
                coach_triggered=coach_triggered,
                coach_recommendations_count=coach_recommendations_count,
                mutations_applied=mutations_applied,
                ga_changes_applied=ga_changes_applied
            )
            
            self._generations.append(record)
            
            # Trim if needed
            if len(self._generations) > self.max_generations:
                excess = len(self._generations) - self.max_generations
                self._generations = self._generations[excess:]
            
            # Update best ever
            if best_fitness > self._best_ever_fitness:
                self._best_ever_fitness = best_fitness
                self._best_ever_generation = generation
                self._best_ever_params = best_params.copy()
    
    def record_coach_analysis(
        self,
        generation: int,
        session_id: str,
        population_size: int,
        diversity: float,
        overall_assessment: str,
        recommendations: List[Dict[str, Any]],
        mutations_applied: int = 0,
        ga_params_changed: int = 0,
        fitness_gates_changed: int = 0,
        analysis_duration_seconds: float = 0.0
    ) -> None:
        """
        Record a coach analysis event.
        
        Args:
            generation: Generation number when analysis occurred
            session_id: Coach session identifier
            population_size: Population size analyzed
            diversity: Population diversity at analysis
            overall_assessment: Coach's overall assessment text
            recommendations: List of recommendations (as dicts)
            mutations_applied: Number of mutations applied
            ga_params_changed: Number of GA parameters changed
            fitness_gates_changed: Number of fitness gates changed
            analysis_duration_seconds: Time taken for analysis
        """
        with self._lock:
            record = CoachAnalysisRecord(
                generation=generation,
                timestamp=datetime.utcnow(),
                session_id=session_id,
                population_size=population_size,
                diversity=diversity,
                overall_assessment=overall_assessment,
                recommendations_count=len(recommendations),
                recommendations=recommendations,
                mutations_applied=mutations_applied,
                ga_params_changed=ga_params_changed,
                fitness_gates_changed=fitness_gates_changed,
                analysis_duration_seconds=analysis_duration_seconds
            )
            
            self._coach_analyses.append(record)
    
    def get_generations(self, last_n: Optional[int] = None) -> List[GenerationRecord]:
        """
        Get generation records.
        
        Args:
            last_n: If specified, return only last N generations
        
        Returns:
            List of generation records
        """
        with self._lock:
            if last_n is None:
                return list(self._generations)
            return list(self._generations[-last_n:])
    
    def get_coach_analyses(self) -> List[CoachAnalysisRecord]:
        """Get all coach analysis records."""
        with self._lock:
            return list(self._coach_analyses)
    
    def get_best_ever(self) -> Optional[Dict[str, Any]]:
        """
        Get best individual ever found.
        
        Returns:
            Dict with 'generation', 'fitness', 'params', or None if no data
        """
        with self._lock:
            if self._best_ever_generation is None:
                return None
            
            return {
                'generation': self._best_ever_generation,
                'fitness': self._best_ever_fitness,
                'params': self._best_ever_params
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Summary dict with key metrics
        """
        with self._lock:
            if not self._generations:
                return {
                    'total_generations': 0,
                    'coach_analyses': 0,
                    'best_ever': None
                }
            
            recent = self._generations[-10:] if len(self._generations) >= 10 else self._generations
            
            return {
                'total_generations': len(self._generations),
                'first_generation': self._generations[0].generation,
                'last_generation': self._generations[-1].generation,
                'coach_analyses': len(self._coach_analyses),
                'best_ever': {
                    'generation': self._best_ever_generation,
                    'fitness': self._best_ever_fitness
                },
                'recent_mean_fitness': sum(g.mean_fitness for g in recent) / len(recent),
                'recent_best_fitness': max(g.best_fitness for g in recent),
                'recent_diversity': sum(g.diversity for g in recent) / len(recent)
            }
    
    def get_fitness_trend(self, last_n: Optional[int] = 10) -> Dict[str, Any]:
        """
        Analyze fitness trend over recent generations.
        
        Args:
            last_n: Number of recent generations to analyze
        
        Returns:
            Trend analysis dict with:
            - generations: List of generation numbers
            - best_fitness: List of best fitness values
            - mean_fitness: List of mean fitness values
            - improvement_rate: Average improvement per generation
            - is_stagnating: True if improvement is near zero
            - is_regressing: True if fitness is declining
        """
        with self._lock:
            if not self._generations:
                return {
                    'generations': [],
                    'best_fitness': [],
                    'mean_fitness': [],
                    'improvement_rate': 0.0,
                    'is_stagnating': False,
                    'is_regressing': False
                }
            
            recent = self._generations[-last_n:] if last_n else self._generations
            
            if len(recent) < 2:
                return {
                    'generations': [g.generation for g in recent],
                    'best_fitness': [g.best_fitness for g in recent],
                    'mean_fitness': [g.mean_fitness for g in recent],
                    'improvement_rate': 0.0,
                    'is_stagnating': False,
                    'is_regressing': False
                }
            
            # Calculate improvements
            best_improvements = [recent[i].best_fitness - recent[i-1].best_fitness 
                               for i in range(1, len(recent))]
            avg_improvement = sum(best_improvements) / len(best_improvements)
            
            # Detect stagnation/regression
            is_stagnating = abs(avg_improvement) < 0.001
            is_regressing = avg_improvement < -0.001
            
            return {
                'generations': [g.generation for g in recent],
                'best_fitness': [g.best_fitness for g in recent],
                'mean_fitness': [g.mean_fitness for g in recent],
                'improvement_rate': avg_improvement,
                'is_stagnating': is_stagnating,
                'is_regressing': is_regressing
            }
    
    def get_parameter_statistics(self, parameter_name: str, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific parameter across recent generations.
        
        Args:
            parameter_name: Name of parameter to analyze
            last_n: Number of recent generations to include
        
        Returns:
            Parameter statistics dict with:
            - mean: Average value
            - std: Standard deviation
            - min/max: Range
            - trend: 'increasing', 'decreasing', or 'stable'
        """
        with self._lock:
            if not self._generations:
                return {
                    'parameter': parameter_name,
                    'count': 0,
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'trend': 'unknown'
                }
            
            recent = self._generations[-last_n:] if last_n else self._generations
            
            # Extract parameter values from best individuals
            values = []
            for gen_record in recent:
                if parameter_name in gen_record.best_params:
                    values.append(gen_record.best_params[parameter_name])
            
            if not values:
                return {
                    'parameter': parameter_name,
                    'count': 0,
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'trend': 'unknown'
                }
            
            # Calculate statistics
            import statistics
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            
            # Detect trend (compare first half vs second half)
            if len(values) >= 4:
                mid = len(values) // 2
                first_half_mean = statistics.mean(values[:mid])
                second_half_mean = statistics.mean(values[mid:])
                diff = second_half_mean - first_half_mean
                
                if abs(diff) < std_val * 0.5:
                    trend = 'stable'
                elif diff > 0:
                    trend = 'increasing'
                else:
                    trend = 'decreasing'
            else:
                trend = 'insufficient_data'
            
            return {
                'parameter': parameter_name,
                'count': len(values),
                'mean': mean_val,
                'std': std_val,
                'min': min(values),
                'max': max(values),
                'trend': trend,
                'values': values[-10:]  # Last 10 values for inspection
            }
    
    def clear(self) -> None:
        """Clear all stored data."""
        with self._lock:
            self._generations.clear()
            self._coach_analyses.clear()
            self._best_ever_fitness = float('-inf')
            self._best_ever_generation = None
            self._best_ever_params = None
    
    def export_json(self, output_path: str | Path) -> None:
        """
        Export all data to JSON file.
        
        Args:
            output_path: Output file path
        """
        with self._lock:
            data = {
                'exported_at': datetime.utcnow().isoformat(),
                'summary': self.get_summary(),
                'generations': [g.to_dict() for g in self._generations],
                'coach_analyses': [a.to_dict() for a in self._coach_analyses],
                'best_ever': self.get_best_ever()
            }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✓ Agent feed exported to: {output_path}")
    
    def import_json(self, input_path: str | Path) -> None:
        """
        Import data from JSON file.
        
        Args:
            input_path: Input file path
        """
        input_path = Path(input_path)
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        with self._lock:
            self.clear()
            
            # Reconstruct generation records
            for gen_data in data.get('generations', []):
                gen_data['timestamp'] = datetime.fromisoformat(gen_data['timestamp'])
                record = GenerationRecord(**gen_data)
                self._generations.append(record)
            
            # Reconstruct coach analysis records
            for analysis_data in data.get('coach_analyses', []):
                analysis_data['timestamp'] = datetime.fromisoformat(analysis_data['timestamp'])
                record = CoachAnalysisRecord(**analysis_data)
                self._coach_analyses.append(record)
            
            # Restore best ever
            best_ever = data.get('best_ever')
            if best_ever:
                self._best_ever_generation = best_ever['generation']
                self._best_ever_fitness = best_ever['fitness']
                self._best_ever_params = best_ever.get('params')
        
        print(f"✓ Agent feed imported from: {input_path}")


# Global singleton instance
agent_feed = AgentFeed()

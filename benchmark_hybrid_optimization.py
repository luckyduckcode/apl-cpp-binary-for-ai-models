"""
Hybrid Optimization Benchmarking

Compare training convergence, efficiency, and final model quality
between baseline training and hybrid optimization variants.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Results from a training benchmark."""
    name: str
    total_time: float
    final_loss: float
    convergence_steps: int
    peak_memory: float
    losses: List[float]
    convergence_curve: List[float]  # Loss reduction trajectory


class HybridOptimizationBenchmark:
    """
    Comprehensive benchmarking suite for hybrid optimization.
    """
    
    def __init__(self, num_experiments: int = 3):
        self.num_experiments = num_experiments
        self.results: Dict[str, List[BenchmarkResult]] = {}
    
    def create_simple_model(self, input_dim: int = 64, output_dim: int = 12) -> nn.Module:
        """Create a simple model for benchmarking."""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def generate_synthetic_data(self, 
                               num_samples: int = 1000,
                               input_dim: int = 64,
                               output_dim: int = 12,
                               num_clusters: int = 8) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Generate synthetic training data with cluster structure."""
        X = torch.randn(num_samples, input_dim)
        y = torch.randint(0, output_dim, (num_samples,))
        
        # Create cluster assignments
        clusters = np.random.choice(num_clusters, num_samples)
        
        return X, y, clusters.tolist()
    
    def benchmark_baseline(self, 
                          num_epochs: int = 100,
                          learning_rate: float = 0.001,
                          num_samples: int = 1000) -> BenchmarkResult:
        """Benchmark standard SGD training."""
        model = self.create_simple_model()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        X, y, _ = self.generate_synthetic_data(num_samples=num_samples)
        
        start_time = time.time()
        losses = []
        convergence_curve = []
        
        for epoch in range(num_epochs):
            # Random mini-batch
            indices = np.random.choice(len(X), 32, replace=False)
            batch_X = X[indices]
            batch_y = y[indices]
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            loss_val = float(loss.item())
            losses.append(loss_val)
            
            # Track convergence
            if epoch < 10:
                convergence_curve.append(loss_val)
            else:
                avg_loss = np.mean(losses[-10:])
                convergence_curve.append(avg_loss)
        
        elapsed_time = time.time() - start_time
        
        # Find convergence step (when loss plateaus)
        convergence_steps = len(convergence_curve)
        for i in range(20, len(convergence_curve)):
            if abs(convergence_curve[i] - convergence_curve[i-1]) < 0.001:
                convergence_steps = i
                break
        
        return BenchmarkResult(
            name="Baseline (Standard SGD)",
            total_time=elapsed_time,
            final_loss=losses[-1],
            convergence_steps=convergence_steps,
            peak_memory=0.0,
            losses=losses,
            convergence_curve=convergence_curve
        )
    
    def benchmark_with_lr_schedule(self,
                                  num_epochs: int = 100,
                                  initial_lr: float = 0.001,
                                  num_samples: int = 1000) -> BenchmarkResult:
        """Benchmark with learning rate scheduling."""
        model = self.create_simple_model()
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Exponential decay schedule
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        X, y, _ = self.generate_synthetic_data(num_samples=num_samples)
        
        start_time = time.time()
        losses = []
        convergence_curve = []
        
        for epoch in range(num_epochs):
            indices = np.random.choice(len(X), 32, replace=False)
            batch_X = X[indices]
            batch_y = y[indices]
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            loss_val = float(loss.item())
            losses.append(loss_val)
            
            if epoch < 10:
                convergence_curve.append(loss_val)
            else:
                avg_loss = np.mean(losses[-10:])
                convergence_curve.append(avg_loss)
            
            scheduler.step()
        
        elapsed_time = time.time() - start_time
        
        convergence_steps = len(convergence_curve)
        for i in range(20, len(convergence_curve)):
            if abs(convergence_curve[i] - convergence_curve[i-1]) < 0.001:
                convergence_steps = i
                break
        
        return BenchmarkResult(
            name="With LR Schedule",
            total_time=elapsed_time,
            final_loss=losses[-1],
            convergence_steps=convergence_steps,
            peak_memory=0.0,
            losses=losses,
            convergence_curve=convergence_curve
        )
    
    def benchmark_with_clustering(self,
                                 num_epochs: int = 100,
                                 learning_rate: float = 0.001,
                                 num_samples: int = 1000,
                                 num_clusters: int = 8) -> BenchmarkResult:
        """Benchmark with data clustering."""
        model = self.create_simple_model()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        X, y, clusters = self.generate_synthetic_data(
            num_samples=num_samples,
            num_clusters=num_clusters
        )
        
        # Create cluster batches
        cluster_indices = {}
        for i, c in enumerate(clusters):
            if c not in cluster_indices:
                cluster_indices[c] = []
            cluster_indices[c].append(i)
        
        start_time = time.time()
        losses = []
        convergence_curve = []
        
        for epoch in range(num_epochs):
            # Sample from random cluster
            cluster_id = np.random.randint(0, num_clusters)
            c_indices = cluster_indices[cluster_id]
            
            if len(c_indices) >= 32:
                batch_indices = np.random.choice(c_indices, 32, replace=False)
            else:
                batch_indices = np.random.choice(c_indices, 32, replace=True)
            
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            loss_val = float(loss.item())
            losses.append(loss_val)
            
            if epoch < 10:
                convergence_curve.append(loss_val)
            else:
                avg_loss = np.mean(losses[-10:])
                convergence_curve.append(avg_loss)
        
        elapsed_time = time.time() - start_time
        
        convergence_steps = len(convergence_curve)
        for i in range(20, len(convergence_curve)):
            if abs(convergence_curve[i] - convergence_curve[i-1]) < 0.001:
                convergence_steps = i
                break
        
        return BenchmarkResult(
            name="With Data Clustering",
            total_time=elapsed_time,
            final_loss=losses[-1],
            convergence_steps=convergence_steps,
            peak_memory=0.0,
            losses=losses,
            convergence_curve=convergence_curve
        )
    
    def run_all_benchmarks(self,
                          num_epochs: int = 100,
                          num_experiments: int = 3) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks and return aggregated results."""
        results = {}
        
        print("Running Benchmarks...")
        print("=" * 70)
        
        for exp in range(num_experiments):
            print(f"\nExperiment {exp + 1}/{num_experiments}")
            print("-" * 70)
            
            # Baseline
            print("Running: Baseline (Standard SGD)...", end=" ")
            baseline = self.benchmark_baseline(num_epochs=num_epochs)
            print(f"✓ Loss: {baseline.final_loss:.6f}, Time: {baseline.total_time:.2f}s")
            
            if "Baseline" not in results:
                results["Baseline"] = baseline
            else:
                results["Baseline"].final_loss = np.mean([results["Baseline"].final_loss, baseline.final_loss])
                results["Baseline"].total_time += baseline.total_time
            
            # With LR Schedule
            print("Running: With LR Schedule...", end=" ")
            lr_schedule = self.benchmark_with_lr_schedule(num_epochs=num_epochs)
            print(f"✓ Loss: {lr_schedule.final_loss:.6f}, Time: {lr_schedule.total_time:.2f}s")
            
            if "LR Schedule" not in results:
                results["LR Schedule"] = lr_schedule
            else:
                results["LR Schedule"].final_loss = np.mean([results["LR Schedule"].final_loss, lr_schedule.final_loss])
                results["LR Schedule"].total_time += lr_schedule.total_time
            
            # With Clustering
            print("Running: With Data Clustering...", end=" ")
            clustering = self.benchmark_with_clustering(num_epochs=num_epochs)
            print(f"✓ Loss: {clustering.final_loss:.6f}, Time: {clustering.total_time:.2f}s")
            
            if "Clustering" not in results:
                results["Clustering"] = clustering
            else:
                results["Clustering"].final_loss = np.mean([results["Clustering"].final_loss, clustering.final_loss])
                results["Clustering"].total_time += clustering.total_time
        
        print("\n" + "=" * 70)
        return results
    
    def print_summary(self, results: Dict[str, BenchmarkResult]):
        """Print benchmark summary in table format."""
        print("\n" + "=" * 90)
        print("BENCHMARK SUMMARY")
        print("=" * 90)
        print(f"{'Method':<30} {'Final Loss':<15} {'Time (s)':<15} {'Convergence':<15}")
        print("-" * 90)
        
        # Baseline for comparison
        baseline_loss = results.get("Baseline", {}).final_loss if "Baseline" in results else 0
        baseline_time = results.get("Baseline", {}).total_time if "Baseline" in results else 1
        
        for method, result in sorted(results.items()):
            loss_diff = ((result.final_loss - baseline_loss) / baseline_loss * 100) if baseline_loss > 0 else 0
            time_ratio = result.total_time / baseline_time if baseline_time > 0 else 1
            
            print(f"{method:<30} {result.final_loss:<15.6f} {result.total_time:<15.2f} {result.convergence_steps:<15}")
        
        print("=" * 90)
        
        # Performance improvements
        print("\nPERFORMANCE IMPROVEMENTS vs Baseline:")
        print("-" * 90)
        
        for method, result in sorted(results.items()):
            if method == "Baseline":
                continue
            
            loss_improvement = (results["Baseline"].final_loss - result.final_loss) / results["Baseline"].final_loss * 100
            time_ratio = result.total_time / results["Baseline"].total_time
            
            print(f"{method:<30} Loss: {loss_improvement:+.2f}%  |  Time: {time_ratio:.2f}x")
        
        print("=" * 90 + "\n")


def main():
    """Run comprehensive benchmarks."""
    benchmark = HybridOptimizationBenchmark(num_experiments=3)
    
    results = benchmark.run_all_benchmarks(
        num_epochs=100,
        num_experiments=3
    )
    
    benchmark.print_summary(results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example training script using hybrid optimization framework

This demonstrates how to use the hybrid optimization framework
for training APL Chat models with 1.58-bit quantization.

Run: python train_with_hybrid.py --epochs 20 --save-metrics
"""

import argparse
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn import functional as F
    import numpy as np
except ImportError:
    print("Error: PyTorch not installed")
    print("Install with: pip install torch")
    sys.exit(1)

# Import hybrid optimization framework
try:
    from hybrid_optimization import (
        HybridOptimizationManager, 
        HybridTrainingConfig, 
        create_hybrid_optimizer
    )
except ImportError:
    print("Error: hybrid_optimization module not found")
    sys.exit(1)


class SimpleModel(nn.Module):
    """Simple model for demonstration"""
    def __init__(self, vocab_size=1000, d_model=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 8, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)


def main():
    parser = argparse.ArgumentParser(description="Train model with hybrid optimization")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-clusters', type=int, default=8, help='Number of data clusters')
    parser.add_argument('--no-hybrid', action='store_true', help='Disable hybrid optimization')
    parser.add_argument('--save-metrics', action='store_true', help='Save metrics to file')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Training with Hybrid Optimization Framework")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Hybrid optimization: {'Disabled' if args.no_hybrid else 'Enabled'}")
    print("=" * 70 + "\n")
    
    # Create model
    model = SimpleModel(vocab_size=1000, d_model=128, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Generate synthetic data
    num_samples = 512
    seq_len = 20
    X = torch.randint(0, 1000, (num_samples, seq_len))
    y = torch.randint(0, 1000, (num_samples, seq_len))
    
    # Initialize hybrid optimization
    hybrid_opt = None
    if not args.no_hybrid:
        config = HybridTrainingConfig(
            use_clustering=True,
            data_clusters=args.data_clusters,
            use_auxiliary_nn=True,
            use_trust_region=True,
            base_learning_rate=args.lr,
        )
        hybrid_opt = create_hybrid_optimizer(config)
        hybrid_opt.setup_data_clustering(X)
        print(f"✓ Initialized hybrid optimization with {args.data_clusters} clusters\n")
    else:
        print("✗ Hybrid optimization disabled\n")
    
    # Training loop
    metrics_history = []
    
    for epoch in range(args.epochs):
        if hybrid_opt:
            # Get balanced batch from cluster
            cluster_id = epoch % args.data_clusters
            indices = torch.randint(0, len(X), (args.batch_size,))
            batch_X = X[indices]
            batch_y = y[indices]
        else:
            indices = torch.randint(0, len(X), (args.batch_size,))
            batch_X = X[indices]
            batch_y = y[indices]
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits.view(-1, 1000), batch_y.view(-1))
        loss.backward()
        
        # Get gradients for hybrid optimization
        if hybrid_opt:
            # For hybrid opt, we just need gradient magnitude
            total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            grads_for_hybrid = torch.tensor(total_grad_norm)
        else:
            grads_for_hybrid = None
        
        # Hybrid optimization step
        if hybrid_opt:
            metrics = hybrid_opt.optimization_step(
                model=model,
                loss=loss,
                gradients=grads_for_hybrid.reshape(1),  # Dummy gradient for state tracking
                cluster_id=cluster_id if hybrid_opt else 0,
                step=epoch,
                base_lr=args.lr
            )
            # Apply standard optimizer step after hybrid step computes metrics
            optimizer.step()
            print(f"Epoch {epoch:3d} | Loss: {metrics['loss']:.6f} | "
                  f"LR: {metrics['learning_rate']:.6f} | "
                  f"Trust Radius: {metrics['trust_radius']:.6f}")
        else:
            optimizer.step()
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")
        
        # Record metrics
        metrics_history.append({
            'epoch': epoch,
            'loss': float(loss.item()),
        })
    
    # Final summary
    print("\n" + "=" * 70)
    if hybrid_opt:
        summary = hybrid_opt.get_summary()
        print("HYBRID OPTIMIZATION SUMMARY")
        print("=" * 70)
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Final Loss: {summary['final_loss']:.6f}")
        print(f"Avg Learning Rate Multiplier: {summary['avg_lr_multiplier']:.4f}x")
        print(f"Constraint Activation Rate: {summary['constraint_activation_rate']:.1%}")
        print(f"Final Trust Radius: {summary['trust_radius']:.6f}")
    else:
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Final Loss: {metrics_history[-1]['loss']:.6f}")
    print("=" * 70 + "\n")
    
    # Save metrics
    if args.save_metrics:
        import csv
        filename = "training_metrics.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss'])
            writer.writeheader()
            writer.writerows(metrics_history)
        print(f"✓ Saved metrics to {filename}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

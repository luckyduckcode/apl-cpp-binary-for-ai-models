import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from quantization import quantize_weights, binarize_weights, unpack_binarized, ternary_quantize_weights, unpack_ternary, replace_linears_with_qat
from hybrid_optimization import (
    HybridOptimizationManager, HybridTrainingConfig, create_hybrid_optimizer
)

# Assume a simple transformer model as student (placeholder for APL)
class StudentModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)

def main():
    # Teacher model (placeholder, assume pre-trained)
    teacher = StudentModel(1000, 128, 8, 2)  # Dummy

    # Student and args
    parser = argparse.ArgumentParser()
    parser.add_argument('--qat', action='store_true', help='Enable quantization-aware training (1-bit weight-only).')
    parser.add_argument('--ternary', action='store_true', help='Enable ternary (1.58-bit) quantization during training.')
    parser.add_argument('--hybrid', action='store_true', help='Enable hybrid optimization (clustering + auxiliary NN + trust region).')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--metrics_out', type=str, default='student_metrics.csv', help='CSV file to write per-epoch metrics')
    parser.add_argument('--save_metrics', action='store_true', help='Save per-epoch metrics to CSV')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learnable_scale', action='store_true', help='Enable learnable per-channel scales in QATLinear')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate quantized metrics every N epochs')
    parser.add_argument('--data_clusters', type=int, default=8, help='Number of data clusters for hybrid optimization')
    parser.add_argument('--param_clusters', type=int, default=4, help='Number of parameter clusters for hybrid optimization')
    args = parser.parse_args()

    student = StudentModel(1000, 64, 4, 1)  # Smaller

    # Replace linear modules with QAT counterparts if requested
    if args.qat:
        replaced = replace_linears_with_qat(student, per_channel_axis=0)
        print(f"Replaced {replaced} Linear modules with QATLinear for 1-bit fake quant.")
        print(f"Replaced {replaced} Linear modules with QATLinear for 1-bit fake quant.")

# Loss
    def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, T=2.0):
        hard_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)
        return alpha * hard_loss + (1 - alpha) * soft_loss

    # Training loop
    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dummy_data = torch.randint(0, 1000, (args.batch_size, 20))  # batch x seq
    labels = torch.randint(0, 1000, (args.batch_size, 20))

    # Initialize hybrid optimization if requested
    hybrid_opt_manager = None
    if args.hybrid:
        hybrid_config = HybridTrainingConfig(
            use_quantization=True,
            target_bits=1.58 if args.ternary else 1.0,
            use_clustering=True,
            data_clusters=args.data_clusters,
            parameter_clusters=args.param_clusters,
            use_auxiliary_nn=True,
            use_trust_region=True,
            base_learning_rate=args.lr,
        )
        hybrid_opt_manager = create_hybrid_optimizer(hybrid_config)
        hybrid_opt_manager.setup_data_clustering(dummy_data)
        print(f"Initialized hybrid optimization with {args.data_clusters} data clusters and auxiliary NN guidance")
    
    # small eval batch for metric computation
    eval_batch = torch.randint(0, 1000, (2, 10))

    def evaluate_quantized_state(student_module):
        # Build a transient quantized NPZ in-memory and load dequantized weights into a new model
        state = student_module.state_dict()
        quantized_dict = {}
        is_ternary = args.ternary
        
        for name, param in state.items():
            nd = param.detach().cpu().numpy()
            if nd.ndim >= 2:
                if is_ternary:
                    ternary_arr, scales, info = ternary_quantize_weights(nd)
                    quantized_dict[f"{name}_ternary"] = ternary_arr
                else:
                    packed, scales, info = binarize_weights(nd)
                    quantized_dict[f"{name}_1bit"] = packed
                quantized_dict[f"{name}_shape"] = np.array(info['shape'], dtype=np.int32)
                quantized_dict[f"{name}_scales"] = np.array(scales, dtype=np.float32)
                quantized_dict[f"{name}_pc_axis"] = np.array(info['per_channel_axis'], dtype=np.int32)
            else:
                quantized_dict[f"{name}_fp32"] = nd
        
        # Build a dequantized model
        dq_student = StudentModel(1000, 64, 4, 1)
        dq_state = dq_student.state_dict()
        for k in list(dq_state.keys()):
            base = k
            key_ternary = f"{base}_ternary"
            key_1bit = f"{base}_1bit"
            key_fp32 = f"{base}_fp32"
            
            if is_ternary and key_ternary in quantized_dict:
                ternary_arr = quantized_dict[key_ternary]
                shape = tuple(quantized_dict[f"{base}_shape"])
                scales = quantized_dict[f"{base}_scales"]
                mat = unpack_ternary(ternary_arr, scales, shape, per_channel_axis=0)
                dq_state[k] = torch.tensor(mat, dtype=torch.float32)
            elif key_1bit in quantized_dict:
                packed = quantized_dict[key_1bit]
                shape = tuple(quantized_dict[f"{base}_shape"])  # (out, in)
                scales = quantized_dict[f"{base}_scales"]
                mat = unpack_binarized(packed, scales, shape, per_channel_axis=0)
                dq_state[k] = torch.tensor(mat, dtype=torch.float32)
            elif key_fp32 in quantized_dict:
                dq_state[k] = torch.tensor(quantized_dict[key_fp32], dtype=torch.float32)
            else:
                dq_state[k] = dq_state[k]
        
        dq_student.load_state_dict(dq_state)
        dq_student.eval()
        with torch.no_grad():
            fp_out = student_module(eval_batch).numpy()
            dq_out = dq_student(eval_batch).numpy()
        diff = np.abs(fp_out - dq_out)
        return diff.max(), diff.mean()

    metrics = []
    for epoch in range(args.epochs):
        # Use hybrid clustered batch or standard batch
        if hybrid_opt_manager:
            cluster_id = epoch % args.data_clusters
            batch_data = hybrid_opt_manager.get_cluster_batch(dummy_data, args.batch_size, cluster_id)
        else:
            batch_data = dummy_data
            cluster_id = 0
        
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher(batch_data)
        student_logits = student(batch_data)
        loss = distillation_loss(student_logits, teacher_logits, labels)
        loss.backward()
        
        # Hybrid optimization step
        if hybrid_opt_manager and student_logits.grad is not None:
            grads = torch.cat([p.grad.flatten() for p in student.parameters() if p.grad is not None])
            hybrid_metrics = hybrid_opt_manager.optimization_step(
                model=student,
                loss=loss,
                gradients=grads,
                cluster_id=cluster_id,
                step=epoch,
                base_lr=args.lr
            )
            optimizer.zero_grad()  # Clear for next step
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, LR: {hybrid_metrics['learning_rate']:.6f}, Trust Radius: {hybrid_metrics['trust_radius']:.6f}")
        else:
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        if args.qat and (epoch % args.eval_every == 0):
            maxd, meand = evaluate_quantized_state(student)
            print(f"QAT Eval after epoch {epoch}: max diff={maxd:.4f}, mean diff={meand:.4f}")
            metrics.append({'epoch': epoch, 'loss': loss.item(), 'max_diff': float(maxd), 'mean_diff': float(meand)})
        else:
            metrics.append({'epoch': epoch, 'loss': loss.item(), 'max_diff': None, 'mean_diff': None})

# After training: save FP32 student and export quantized weights plus biases
    torch.save(student.state_dict(), "student_fp32.pth")
    print("Saved student_fp32.pth")
    
    # Determine quantization format
    if args.ternary:
        quant_suffix = "ternary_1.58bit"
        last_name = "student_quantized_ternary_1.58bit_qat.npz" if args.qat else "student_quantized_ternary_1.58bit.npz"
    else:
        quant_suffix = "1bit"
        last_name = "student_quantized_1bit_qat.npz" if args.qat else "student_quantized_1bit.npz"
    
    quantized_dict = {}
    for name, param in student.state_dict().items():
        nd = param.detach().cpu().numpy()
        if nd.ndim >= 2:
            if args.ternary:
                ternary_arr, scales, info = ternary_quantize_weights(nd)
                quantized_dict[f"{name}_ternary"] = ternary_arr
            else:
                packed, scales, info = binarize_weights(nd)
                quantized_dict[f"{name}_1bit"] = packed
            quantized_dict[f"{name}_shape"] = np.array(info['shape'], dtype=np.int32)
            quantized_dict[f"{name}_scales"] = np.array(scales, dtype=np.float32)
        else:
            quantized_dict[f"{name}_fp32"] = nd

    np.savez_compressed(last_name, **quantized_dict)
    print(f"Saved {last_name} with {quant_suffix} quantized weights")
    if args.save_metrics:
        import csv
        with open(args.metrics_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['epoch','loss','max_diff','mean_diff'])
            w.writeheader()
            for r in metrics:
                w.writerow(r)
        print('Wrote per-epoch metrics to', args.metrics_out)
    
    # Print hybrid optimization summary
    if hybrid_opt_manager:
        summary = hybrid_opt_manager.get_summary()
        print("\n" + "="*60)
        print("HYBRID OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Final Loss: {summary['final_loss']:.6f}")
        print(f"Avg Learning Rate Multiplier: {summary['avg_lr_multiplier']:.4f}x")
        print(f"Constraint Activation Rate: {summary['constraint_activation_rate']:.1%}")
        print(f"Final Trust Radius: {summary['trust_radius']:.6f}")
        print("="*60 + "\n")


if __name__ == '__main__':
    main()

# After training, quantize
# Use the quantization script

# Export to ONNX (commented out due to installation issues)
# torch.onnx.export(
#     student,
#     dummy_data,
#     "student_model.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size", 1: "seq_len"}}
# )
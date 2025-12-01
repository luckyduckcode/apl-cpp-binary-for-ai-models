#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

// Optimized matrix multiplication using SIMD (simplified)
std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int m = A.size();
    int n = B[0].size();
    int p = A[0].size();
    std::vector<std::vector<double>> C(m, std::vector<double>(n, 0.0));
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < p; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Softmax
std::vector<std::vector<double>> softmax(const std::vector<std::vector<double>>& X) {
    std::vector<std::vector<double>> Y = X;
    for (auto& row : Y) {
        double max_val = *std::max_element(row.begin(), row.end());
        double sum = 0.0;
        for (auto& val : row) {
            val = std::exp(val - max_val);
            sum += val;
        }
        for (auto& val : row) {
            val /= sum;
        }
    }
    return Y;
}

// ReLU
std::vector<std::vector<double>> relu(const std::vector<std::vector<double>>& X) {
    std::vector<std::vector<double>> Y = X;
    for (auto& row : Y) {
        for (auto& val : row) {
            val = std::max(0.0, val);
        }
    }
    return Y;
}
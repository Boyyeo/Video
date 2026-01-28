import torch
import torch.nn as nn

# 1. 定義原始的 Conv3d 層
in_channels = 3
out_channels = 1152
kernel_size = (2, 16, 16)
stride = kernel_size

conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride)

# 2. 模擬輸入數據 (N, C, D, H, W)
# 假設 11520 是 Batch Size
input_tensor = torch.randn(11520, 3, 2, 16, 16) * 10 + 2

# --- 方法 A: 使用原本的 PyTorch Conv3d ---
with torch.no_grad():
    output_conv = conv(input_tensor) 
    # 輸出形狀會是 (11520, 1152, 1, 1, 1)

# --- 方法 B: 使用矩陣乘法 (MatMul) ---
with torch.no_grad():
    # 提取參數
    weight = conv.weight  # Shape: (1152, 3, 2, 16, 16)
    bias = conv.bias      # Shape: (1152)

    # 1. 展平輸入: (N, C*D*H*W)
    # 這裡的 1536 來自 3 * 2 * 16 * 16
    x_flat = input_tensor.view(input_tensor.size(0), -1)
    print('x_flat shape:', x_flat.shape)

    # 2. 展平權重並轉置: (C_out, 1536) -> (1536, C_out)
    w_flat = weight.view(out_channels, -1).t()

    print('w_flat shape:', w_flat.shape)
    # 3. 執行矩陣乘法 Y = XW + b
   # output_matmul = torch.matmul(x_flat, w_flat) + bias
    output_matmul = x_flat @ w_flat + bias
    # 4. 如果需要跟 Conv3d 的輸出形狀完全一致，補回維度
    output_matmul_reshaped = output_matmul.view(input_tensor.size(0), out_channels, 1, 1, 1)

# --- 驗證結果 ---
diff = torch.abs(output_conv - output_matmul_reshaped).max()
print(f"最大誤差: {diff.item()}")
print(f"結果是否一致: {torch.allclose(output_conv, output_matmul_reshaped, atol=1e-6)}")
print('output_conv :', output_conv)
print('output_matmul_reshaped :', output_matmul_reshaped)
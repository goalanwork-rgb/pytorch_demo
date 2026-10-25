"""第一堂課：PyTorch 張量（Tensor）基礎

執行前請先啟動環境：
    .\\activate.bat

或直接：
    python 01_tensor_basics.py
"""

import torch
import numpy as np

print("=" * 60)
print("第一堂課：PyTorch 張量基礎")
print("=" * 60)

# 1. 確認環境與 PyTorch 版本
print("\n--- 1. 環境資訊 ---")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
else:
    print("目前使用 CPU（學習階段完全沒問題）")

# 2. 建立張量
print("\n--- 2. 建立張量 ---")

# 從列表建立
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"從列表建立: {x_data.flatten().tolist()}")

# 從 NumPy 建立
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"從 NumPy 建立: {x_np.flatten().tolist()}")

# 內建函數
x_ones = torch.ones(2, 3)
print(f"\n全 1 張量 (2,3):\n{x_ones}")

x_zeros = torch.zeros(2, 3)
print(f"\n全 0 張量 (2,3):\n{x_zeros}")

x_rand = torch.rand(2, 3)
print(f"\n隨機張量 (2,3):\n{x_rand}")

# 3. 張量屬性
print("\n--- 3. 張量屬性 ---")
tensor = torch.rand(3, 4)
print(f"內容:\n{tensor}")
print(f"形狀 (shape): {tensor.shape}")
print(f"維度 (ndim): {tensor.ndim}")
print(f"資料型態 (dtype): {tensor.dtype}")
print(f"所在裝置 (device): {tensor.device}")

# 改變 dtype
x_int = torch.tensor([1, 2, 3])
x_float = x_int.float()
print(f"\n整數張量 dtype: {x_int.dtype}")
print(f"轉換為 float: {x_float.dtype}")

# 4. 張量運算
print("\n--- 4. 張量運算 ---")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"a = {a.tolist()}")
print(f"b = {b.tolist()}")
print(f"a + b = {(a + b).tolist()}")
print(f"a * b = {(a * b).tolist()}")
print(f"a ** 2 = {(a ** 2).tolist()}")

# 矩陣乘法
x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2,3)
y = torch.tensor([[7, 8], [9, 10], [11, 12]])  # (3,2)
z = x @ y
print(f"\n矩陣乘法 (2,3) @ (3,2) = (2,2):\n{z}")

# 聚合運算
m = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print(f"\n總和: {m.sum().item()}")
print(f"平均值: {m.mean().item()}")
print(f"最大值: {m.max().item()}")
print(f"最小值: {m.min().item()}")
print(f"每列總和 (dim=0): {m.sum(dim=0).tolist()}")
print(f"每行總和 (dim=1): {m.sum(dim=1).tolist()}")

# 5. 索引與切片
print("\n--- 5. 索引與切片 ---")
mat = torch.tensor([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]])
print(f"原始張量:\n{mat}")
print(f"第一列: {mat[0].tolist()}")
print(f"元素 (0,2): {mat[0, 2].item()}")
print(f"第二欄所有元素: {mat[:, 1].tolist()}")
print(f"前兩列前兩欄:\n{mat[:2, :2]}")

# 形狀操作
x = torch.arange(12)
print(f"\n原始 x: {x.tolist()}, shape={x.shape}")
y = x.reshape(3, 4)
print(f"reshape(3,4):\n{y}")
f = y.flatten()
print(f"flatten(): {f.tolist()}, shape={f.shape}")
w = x.reshape(2, -1)
print(f"reshape(2,-1) 自動推算: shape={w.shape}")

# 拼接
t1 = torch.tensor([[1, 2], [3, 4]])
t2 = torch.tensor([[5, 6], [7, 8]])
print(f"\n沿 dim=0 拼接:\n{torch.cat([t1, t2], dim=0)}")
print(f"沿 dim=1 拼接:\n{torch.cat([t1, t2], dim=1)}")

# 6. 設備（Device）
print("\n--- 6. 設備 ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置: {device}")
t = torch.tensor([1., 2., 3.]).to(device)
print(f"張量在裝置上: {t.device}")

# 7. 綜合練習（第一堂課最後的挑戰）
print("\n--- 7. 綜合練習 ---")
num = torch.arange(1, 31).float().reshape(5, 6)
print(f"1-30 的 5x6 張量:\n{num}")
num2 = num.reshape(6, 5)
print(f"\nreshape 成 (6,5):\n{num2}")
print(f"第三列 (index 2): {num2[2].tolist()}")
print(f"最大值: {num2.max().item()}, 最小值: {num2.min().item()}, 平均值: {num2.mean().item():.2f}")

print("\n" + "=" * 60)
print("恭喜！你完成了第一堂課的程式碼！")
print("=" * 60)
print("\n下一步：")
print("  1. 嘗試修改上面每段程式碼，觀察輸出變化")
print("  2. 把上面的綜合練習改成自己的版本")
print("  3. 準備第二堂課：自動微分（Autograd）")

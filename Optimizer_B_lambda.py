import torch

# V_N 和 B 的更新: 迭代直接更

# 乘子Z的更新，PGD params: Z, B_new, B_old
# 非凸共轭邻近算子
# 非凸正则项
def nonconvex_regularization(y, lambda2):
    return lambda2 * torch.sum(torch.norm(torch.abs(y)-1, p=1))

# h 的 共轭函数，
def hConj(y, lambda2, large_numb=1e8):
    # 创建一个与 y 形状相同，但所有元素都是 large_number 的张量
    result = torch.full_like(y, large_numb)
    # 找到满足条件的元素的索引
    mask = (y >= -lambda2) & (y <= lambda2)
    # 对于满足条件的元素，计算其绝对值
    result[mask] = torch.abs(y[mask])
    return result

# 共轭邻近算子
def hConjProx(v, lambda2, alpha3):
    prox_values = torch.where(v < -lambda2 - alpha3, torch.full_like(v, -lambda2),
                    torch.where((v >= -lambda2 - alpha3) & (v < -alpha3), v + alpha3,
                    torch.where((v >= -alpha3) & (v <= alpha3), torch.zeros_like(v),
                    torch.where((v > alpha3) & (v <= alpha3 + lambda2), v - alpha3,
                    torch.full_like(v, lambda2)))))
    return prox_values

# 乘子Z的值一直在[-lambda2, lambda2]之间，要想Z一直更新， 并按照我自己的初始化方式，会在后面调参数时一直考虑：lambda2 > alpha3
def updateZ(Z, B_new, B_old, lambda2, alpha3):
    B_extra = 2 * B_new - B_old
    Z_extra = Z + alpha3 * B_extra
    Z_new = hConjProx(Z_extra, lambda2, alpha3)

    return Z_new

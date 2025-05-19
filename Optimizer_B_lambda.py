import torch
# dual hash-PGD with  extrapolation : Z, B_new, B_old

def nonconvex_regularization(y, lambda2):
    return lambda2 * torch.sum(torch.norm(torch.abs(y)-1, p=1))


def hConj(y, lambda2, large_numb=1e8):
    result = torch.full_like(y, large_numb)
    mask = (y >= -lambda2) & (y <= lambda2)
    result[mask] = torch.abs(y[mask])
    return result

def hConjProx(v, lambda2, alpha3):
    prox_values = torch.where(v < -lambda2 - alpha3, torch.full_like(v, -lambda2),
                    torch.where((v >= -lambda2 - alpha3) & (v < -alpha3), v + alpha3,
                    torch.where((v >= -alpha3) & (v <= alpha3), torch.zeros_like(v),
                    torch.where((v > alpha3) & (v <= alpha3 + lambda2), v - alpha3,
                    torch.full_like(v, lambda2)))))
    return prox_values

# The value of multiplier Z is always constrained within the range [-lambda2, lambda2].
# To ensure that Z updates continuously according to my initialization approach,
# when tuning parameters later I will always maintain the condition: lambda2 > alpha3
def updateZ(Z, B_new, B_old, lambda2, alpha3):
    B_extra = 2 * B_new - B_old
    Z_extra = Z + alpha3 * B_extra
    Z_new = hConjProx(Z_extra, lambda2, alpha3)

    return Z_new

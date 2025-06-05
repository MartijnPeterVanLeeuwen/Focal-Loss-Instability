import torch
import torch.nn.functional as F

from torchvision.utils import _log_api_usage_once   #Modification 


def sigmoid_focal_loss_modified(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
    epsilon=1e-3 #Modification 

    
) -> torch.Tensor:
    """
    Modified version of the Focal Loss. The epsilon scalar that is 
    added to the output stabilizes the model training. Whenever 
    epsilon is set to 0, it simplifies to the original Focal loss.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
                Stores the binary classification label for each element 
                in inputs (0 for the negative class and 
                1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for 
                ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        epsilon(float): Smoothing constant preventing the 
                instabilities when gamma values between 0 and 1
                are used. Default: ``1e-3``
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. 
                Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    #Modification of the Original implementation from 
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(sigmoid_focal_loss_modified)  #Modification 
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, 
                reduction="none")
    p_t = (p) * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t+epsilon) ** gamma) #Modification 

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n 
            Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

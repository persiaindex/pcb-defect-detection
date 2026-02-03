def normalize_tensor(x):
    """
    Normalize tensor to range [0, 1]
    """
    return (x - x.min()) / (x.max() - x.min())

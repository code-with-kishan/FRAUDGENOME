try:
    from imblearn.combine import SMOTETomek
except Exception:
    SMOTETomek = None

import numpy as np

def resample_smote_tomek(X, y, random_state=42):
    """Resample using SMOTE-Tomek if available, otherwise return inputs unchanged."""
    if SMOTETomek is None:
        return X, y
    smt = SMOTETomek(random_state=random_state)
    Xr, yr = smt.fit_resample(X, y)
    return Xr, yr

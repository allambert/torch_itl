from .datasets import (import_toy_synthesis, import_data_otoliths, # noqa
                       import_affectnet_va_embedding, import_data_toy_quantile)
from .outliers import (add_local_outliers, add_global_outliers_worse, # noqa
                       add_global_outliers_linear, add_type1_outliers,
                       add_type2_outliers, add_type3_outliers)
from .synthetic_func_or import synthetic_gaussian, SyntheticGPmixture # noqa
from .synthetic_style import SyntheticTriangles # noqa

"""Tools registry for Quantara"""
try:
    from .kalshi_tool import *
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .polymarket_tool import *
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .analysis_tool import *
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .enrichment_tool import *
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .niche_tool import *
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .planner_tool import *
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .rag_tool import *
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .weather_prediction_tool import *
except (ImportError, ModuleNotFoundError):
    pass

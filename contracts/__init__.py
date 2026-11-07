"""natHacks 2026 (Pandas) - shared contracts for the night initial-attack drone swarm.

Single source of truth for message shapes, entities, topics, interfaces and scenario config.
Depends only on pydantic. See context.md for conventions and how to change a contract.
"""
from .base import CONTRACT_VERSION, Contract, FrozenContract, Seconds
from .config import *  # noqa: F401,F403
from .entities import *  # noqa: F401,F403
from .enums import *  # noqa: F401,F403
from .geometry import *  # noqa: F401,F403
from .interfaces import *  # noqa: F401,F403
from .messages import *  # noqa: F401,F403
from .sensing import *  # noqa: F401,F403
from .topics import *  # noqa: F401,F403

__version__ = CONTRACT_VERSION

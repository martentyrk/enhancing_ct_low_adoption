"""Utility functions for WandB."""


class WandbDummy:
  """Dummy class for WandB."""

  def __init__(self, *args, **kwargs):
    del args, kwargs
    self.name = "dummy"

  def log(self, *args, **kwargs):
    """Dummy logging function."""
    del args, kwargs

  def finish(self, *args, **kwargs):
    """Dummy finish function."""
    del args, kwargs

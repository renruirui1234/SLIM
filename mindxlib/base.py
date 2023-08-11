import abc
import sys

"""
Ensure compatibility with Python2/3
"""
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class PostHocWhiteBoxBase(ABC):
    
    """
    PostHocBase is the base class for post-hoc white-box explainers(e.g. shap, LIME).
    """
    
    def __init__(self, *argv, **kwargs):
        """
        Initialize a PostHocBase object
        """

    @abc.abstractmethod
    def set_parameters(self, *argv, **kwargs):
        """
        Set parameters for post-hoc explainer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def explain_instance(self, *argv, **kwargs):
        """
        Explain an input instance x.
        """
        raise NotImplementedError


class PostHocBlackBoxBase(ABC):
    
    """
    PostHocBase is the base class for post-hoc black-box explainers(e.g. shap, LIME).
    """
    
    def __init__(self, *argv, **kwargs):
        """
        Initialize a PostHocBase object
        """

    @abc.abstractmethod
    def set_parameters(self, *argv, **kwargs):
        """
        Set parameters for post-hoc explainer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def explain_instance(self, *argv, **kwargs):
        """
        Explain an input instance x.
        """
        raise NotImplementedError
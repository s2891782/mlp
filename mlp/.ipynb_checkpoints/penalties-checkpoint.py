import numpy as np

seed = 22102017 
rng = np.random.RandomState(seed)


class L1Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L1 norm.
    """

    def __init__(self, coefficient):
        """Create a new L1 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L1 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return self.coefficient*np.sum(np.abs(parameter))

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.coefficient*np.sign(parameter)

    def __repr__(self):
        return 'L1Penalty({0})'.format(self.coefficient)


class L2Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L2 norm.
    """

    def __init__(self, coefficient):
        """Create a new L2 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L2 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        return (0.5)*self.coefficient*np.sum(np.square(parameter))

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        return self.coefficient*(parameter)

    def __repr__(self):
        return 'L2Penalty({0})'.format(self.coefficient)

class L1L2MixPenalty(object):
    """Combined L1 + L2 parameter penalty.
    """

    def __init__(self, l1_coefficient, l2_coefficient):
        """Create a new L1+L2 mix penalty object.

        Args:
            l1_coefficient: Positive constant for the L1 part.
            l2_coefficient: Positive constant for the L2 part.
        """
        assert l1_coefficient > 0., "L1 coefficient must be positive."
        assert l2_coefficient > 0., "L2 coefficient must be positive."

        self.l1_coefficient = l1_coefficient
        self.l2_coefficient = l2_coefficient

    def __call__(self, parameter):
        """Calculate L1+L2 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of combined penalty term.
        """
        l1_term = self.l1_coefficient * np.sum(np.abs(parameter))
        l2_term = 0.5 * self.l2_coefficient * np.sum(np.square(parameter))
        return l1_term + l2_term

    def grad(self, parameter):
        """Calculate gradient of L1+L2 penalty w.r.t. the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Gradient of the combined penalty (same shape as parameter).
        """
        l1_grad = self.l1_coefficient * np.sign(parameter)
        l2_grad = self.l2_coefficient * parameter
        return l1_grad + l2_grad

    def __repr__(self):
        return 'L1L2MixPenalty(L1={0}, L2={1})'.format(
            self.l1_coefficient, self.l2_coefficient)
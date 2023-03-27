class DecisionMaker:
    """Represents a human or artificial decision maker."""
    def __init__(self, dm_type):
        self.dm_type = dm_type

    def interact(self, **kwargs):
        """
        Performs a single interaction with the decision maker where they provide their own inputs.
        :return:
        :rtype:
        """

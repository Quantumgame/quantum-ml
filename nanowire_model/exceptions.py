# Exceptions in the program. Neater way to handle edge cases then to work with if-else statements or hardcode in physical values.


class InvalidChargeState(Exception):
    '''
    This exception is generally raised while solving the ThomasFermi problem. It occurs when the fixed N state the solver tries to solve is not physically possible.
    '''
    pass





import fxcmpy as fxcmpy


def establish_conn():
    """
    Establish connection
    :return: connection object
    """
    con = fxcmpy.fxcmpy(config_file='Config.cfg', server='demo')
    return con


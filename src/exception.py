import sys
from src.log import logging

def error_msg(e,E:sys):
    _,_, exc = E.exc_info()
    filename = exc.tb_frame.f_code.co_filename
    msg = f'Error occured in Python script name [{filename}] line number [{exc.tb_lineno}] error message [{str(e)}]'
    return msg

class CustomException (Exception):
    def __init__(self, msg, E:sys):
        super().__init__(msg)
        self.msg = error_msg(msg, E=E)

    def __str__(self):
        return self.msg
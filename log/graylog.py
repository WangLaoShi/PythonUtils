import logging
import graypy
def log2gray(content,serverIP,ports =12201,inputName='GrayLogMadeByPython'):
    my_logger = logging.getLogger('test_logger')
    my_logger.setLevel(logging.DEBUG)

    handler = graypy.GELFUDPHandler(serverIP, ports, localname=inputName)
    my_logger.addHandler(handler)

    my_logger.debug(content)
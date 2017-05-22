import logging
import datetime
import platform
import os

LOGGER = logging.getLogger(__name__)

def set_up_logger():
    """This function initialises the logger.

    We set up a logger that prints both to the console at the information level
    and to file at the debug level. It will store in the /temp directory on
    *NIX machines and in the local directory on windows.
    """
    timestamp = datetime.datetime.now()

    logfile_name = 'detect_lines-{0:04}-{1:02}-{2:02}-{3:02}{4:02}{5:02}.log'\
                   .format(timestamp.year,
                           timestamp.month,
                           timestamp.day,
                           timestamp.hour,
                           timestamp.minute,
                           timestamp.second)

    if platform.system() == 'Windows':
        os.mkdir('./tmp')
        logfile_name = './tmp/' + logfile_name
    else:
        logfile_name = '/tmp/' + logfile_name

    logging.basicConfig(filename=logfile_name,
                        level=logging.INFO)

    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_logger.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_logger)

    LOGGER.info('All logging will be written to %s', logfile_name)

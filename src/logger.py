import os
from pathlib import Path




LOG_TYPES = ["ERROR", "WARNING", "INFO", "DEBUG"]


class Log:
    def __init__(self, event:int, type:int):
        self.event = event
        self.type = type
    def __str__(self):
        return f'{LOG_TYPES[self.type]}: {self.event}'

class Logger:
    def __init__(self, verbosity: int=0, save_file_path:str=None):
        """Initializes the logger with a verbosity level.
        
        Args:
            verbosity (int): The verbosity level of the logger.
                - 0: No logs.
                - 1: Logs only errors.
                - 2: Logs errors and warnings.
                - 3: Logs errors, warnings and info.
                - 4: Logs errors, warnings, info and debug.
            save_file_path (str): The path to save the log file.
                - if None, the log is not saved, only printed.
        """
        self.log = []
        if(self._validate_verbosity(verbosity)):
            self.verbosity = verbosity
        else:
            raise ValueError(f"Invalid verbosity level for logging: {verbosity}")
        
        if save_file_path:
            if(self._validate_path(save_file_path)):
                self.save_file_path = save_file_path
            else:
                raise ValueError(f"Invalid path for saving log file: {save_file_path}")
        else:
            self.save_file_path = None
    
    def _validate_path(self, path:str):
        """Validates the path to save the log file."""
        
        # try to create the file
        try:
            with open(path, 'w') as file:
                pass
        except:
            return False
        
        return True

    def _validate_verbosity(self, verbosity:int):
        """Validates the verbosity level."""
        return verbosity in range(len(LOG_TYPES))

    

    
    def get_log_verbose_definition():
        return LOG_TYPES
    
    def handle_log_event(self, event:str, type:int):
        """Handles an event by logging it.
        
        Args:
            event (str): The event to log.
            type (int): The type of the event.
        """
        
        # check if the type is valid
        if type not in range(len(LOG_TYPES)):
            self.handle_log_event(f'Type {type} is not valid for event: {event}', 0)
        
        # check if the type is within the verbosity level
        if type > self.verbosity:
            return
        
        # create the log
        log = Log(event, type)
        self.log.append(log)

        # print the log if path is None
        if self.save_file_path is None:
            # print with color red if it is an error, yellow if it is a warning
            if type == 0:
                # if it is an error, raise an exception and print in red
                # it will halt the execution
                raise Exception('\033[91m' + str(log) + '\033[0m')
            elif type == 1:
                print('\033[93m' + str(log) + '\033[0m')
            else:
                print(log)
        else:
            # add the log to the file
            with open(self.save_file_path, 'a') as file:
                file.write(str(log) + '\n')
        
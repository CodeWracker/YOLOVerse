
LOG_TYPES = ["ERROR", "WARNING", "INFO", "DEBUG"]


class Log:
    def __init__(self, event, type):
        self.event = event
        self.type = type
    def __str__(self):
        return f'{LOG_TYPES[self.type]}: {self.event}'

class Logger:
    def __init__(self, verbosity=0, save_path=None):
        """Initializes the logger with a verbosity level.
        
        Args:
            verbosity (int): The verbosity level of the logger.
                - 0: No logs.
                - 1: Logs only errors.
                - 2: Logs errors and warnings.
                - 3: Logs errors, warnings and info.
                - 4: Logs errors, warnings, info and debug.
            save_path (str): The path to save the log file.
                - if None, the log is not saved, only printed.
        """
        self.log = []
    
    def get_verbose_definition():
        return LOG_TYPES
    
    def handle_event(self, event, type):
        """Handles an event by logging it.
        
        Args:
            event (str): The event to log.
            type (int): The type of the event.
        """
        
        # check if the type is valid
        if type not in range(len(LOG_TYPES)):
            self.handle_event(f'Type {type} is not valid.', 0)
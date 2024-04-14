### !! LOGGING CLASS !! ###

try:
    print("LOGGING MODULE IMPORTS...")
    import sys
    from termcolor import colored
    from datetime import datetime
    import os
    import inspect
    import time

    print("Finished Logging Imports")
except Exception as e:
    print(f"There was an error with logging imports: {e}")
    sys.exit()


class Console:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            # Creates an instance of the __init__ method for the class.
        return cls._instance

    def __init__(self):
        self.ready = False
        self.ready = True

    def log(self, content=None, ctype=1):
        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime("%Y-%m-%d")
        formatted_time = current_datetime.strftime("%H:%M:%S")

        frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(frame)[1]
        caller_filename = caller_frame.filename

        if not os.path.isdir("./logs/"):
            os.mkdir("./logs/")

        with open(f"./logs/{formatted_date}.txt", "a", encoding="utf-8") as file:
            # Write content to the file
            file.write(
                f"[{str(formatted_date)}] - [{str(formatted_time)}] ->>> {str(caller_filename)} ->> {str(ctype)} -> {str(content)}\n"
            )
        try:
            if self.ready:
                if content != None:
                    statement = f"""[{time.strftime("%H:%M:%S")}]:{content}"""
                    if ctype == 1:
                        value = colored(statement, "green")
                    elif ctype == 2:
                        value = colored(statement, "yellow")
                    elif ctype == 3:
                        value = colored(statement, "red")  # orange
                    else:
                        value = colored(statement, "magenta")

                    print(value)
                    return True
                else:
                    return False
            else:
                return False
        except Exception as error:
            print(f"FATAL ERROR: {error}")
            return False

    # Print iterations progress
    def printProgressBar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()

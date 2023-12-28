import time

class Timer:
    def __init__(self):
        pass
    
    def start(self):
        self.start_time = time.time()
    
    def end(self):
        self.end_time = time.time()
    
    def get_time_usage_ms(self):
        return (self.end_time - self.start_time)*1000
    
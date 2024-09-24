# Module for multiplexer CD74HC4067

import RPi.GPIO as GPIO
import numpy as np

class CD74HC4067:
    def __init__(self):
        self.binary_address = np.array([[0,  0,  0,  0],
        [1,  0,  0,  0],
        [0,  1,  0,  0],
        [1,  1,  0,  0],
        [0,  0,  1,  0],
        [1,  0,  1,  0],
        [0,  1,  1,  0],
        [1,  1,  1,  0],
        [0,  0,  0,  1],
        [1,  0,  0,  1],
        [0,  1,  0,  1],
        [1,  1,  0,  1],
        [0,  0,  1,  1],
        [1,  0,  1,  1],
        [0,  1,  1,  1],
        [1,  1,  1,  1]])
	    
        self.pins = [16, 12, 19, 26]
	    
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
		
    def set_channel(self, channel_number):
        for i, pin_state in enumerate(self.binary_address[channel_number]):
            if pin_state == 1:
                GPIO.output(self.pins[i], GPIO.HIGH)
            elif pin_state == 0:
                GPIO.output(self.pins[i], GPIO.LOW)

# /*****************************************************************************
# * | File        :	  EPD_1in54.py
# * | Author      :   Waveshare team
# * | Function    :   Hardware underlying interface
# * | Info        :
# *----------------
# * |	This version:   V1.0
# * | Date        :   2019-01-24
# * | Info        :   
# ******************************************************************************/
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documnetation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to  whom the Software is
# furished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS OR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

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

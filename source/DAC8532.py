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

import source.config as config
import RPi.GPIO as GPIO


channel_A   = 0x30
channel_B   = 0x34

DAC_Value_MAX = 65535

DAC_VREF = 5

class DAC8532:
    
    def __init__(self):
        self.cs_dac_pin = config.CS_DAC_PIN
        #config.module_init()
        
        self.channel_A   = 0x30
        self.channel_B   = 0x34

        self.DAC_Value_MAX = 65535

        self.DAC_VREF = 5
    
    def DAC8532_Write_Data(self, Channel, Data):
        config.digital_write(self.cs_dac_pin, GPIO.LOW)#cs  0
        config.spi_writebyte([Channel, Data >> 8, Data & 0xff])
        config.digital_write(self.cs_dac_pin, GPIO.HIGH)#cs  0
        
    def DAC8532_Out_Voltage(self, Channel, Voltage):
        if((Voltage <= self.DAC_VREF) and (Voltage >= 0)):
            temp = int(Voltage * self.DAC_Value_MAX / self.DAC_VREF)
            self.DAC8532_Write_Data(Channel, temp)
  
### END OF FILE ###


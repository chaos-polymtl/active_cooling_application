import config
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


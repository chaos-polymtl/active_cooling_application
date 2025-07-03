import numpy as np

class TLA2528:
    """
    Firmware interface for TLA2528 using auto-sequence mode.
    Reads all 8 analog input channels using I2C and SMBus2.
    """

    # Register addresses (from TLA2528 datasheet Table 8)
    SYSTEM_STATUS = 0x00
    GENERAL_CFG = 0x01
    DATA_CFG = 0x02
    OSR_CFG = 0x03
    OPMODE_CFG = 0x04
    PIN_CFG = 0x05
    SEQUENCE_CFG = 0x10
    CHANNEL_SEL = 0x11
    AUTO_SEQ_CH_SEL = 0x12

    # Opcodes (from Section 7.5.2)
    OP_SINGLE_WRITE = 0x08
    OP_SINGLE_READ = 0x10

    def __init__(self, address):
        from smbus2 import SMBus
        import time

        self.bus = SMBus(1)
        self.address = address
        self.AVDD = 5.55
        self.data = np.zeros(8)

        # Configure all pins as analog inputs
        self.write_register(self.PIN_CFG, 0x00)

        # Disable auto-sequence mode (SEQ_MODE = 00)
        self.write_register(self.SEQUENCE_CFG, 0b00)

        # Disable APPEND_STATUS to receive 2 bytes per channel only
        self.write_register(self.DATA_CFG, 0x00)

        # Start sequencer (SEQ_START=1, CNVST=0 initially)
        self.write_register(self.GENERAL_CFG, 0x10)
        time.sleep(0.1)

    def write_register(self, register, value):
        self.bus.write_i2c_block_data(self.address, self.OP_SINGLE_WRITE, [register, value])

    def read_register(self, register):
        self.bus.write_i2c_block_data(self.address, self.OP_SINGLE_READ, [register])
        return self.bus.read_byte(self.address)

    def wait_osr_done(self, timeout=1):
        import time
        start = time.time()
        while True:
            status = self.read_register(self.SYSTEM_STATUS)
            if status & 0x08:  # OSR_DONE = bit 3
                break
            if time.time() - start > timeout:
                raise TimeoutError("OSR not done within timeout period")

    def burst_read_2_bytes(self):
        from smbus2 import i2c_msg
        read = i2c_msg.read(self.address, 2)
        self.bus.i2c_rdwr(read)
        return list(read)

    def measure_voltage(self):
        # Trigger conversion: clear and then set CNVST and SEQ_START

        for i in range(8):
            self.write_register(self.CHANNEL_SEL, i)
            raw_data = self.burst_read_2_bytes()
            msb = raw_data[0]
            lsb = raw_data[1]
            adc_value = (msb << 4) | (lsb >> 4)
            voltage = adc_value * self.AVDD / 4096
            self.data[i] = voltage
        return self.data

if __name__ == "__main__":
    from time import sleep

    adc = TLA2528(address=0x12)
    while True:
        voltages = adc.measure_voltage()
        print("Voltages:", [f"{v:.3f} V" for v in voltages])
        sleep(1)
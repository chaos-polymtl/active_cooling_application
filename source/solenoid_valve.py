'''
Copyright 2024-2025, the Active Cooling Experimental Application Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np

class Solenoid:

    def __init__(self, n_region, test_UI = False):

        self.test_UI = test_UI
        self.state = [0b00000000, 0b00000000]

        if test_UI:
            return
        from source.DRV8806 import DRV8806

        self.DRV = DRV8806()

        # Create solenoid mask to follow harware
        self.solenoid_mask = {0 : 9, 1 : 8, 2 : 5, 3 : 4, 4 : 7, 5 : 6, 6 : 1, 7 : 0, 8 : 3, 9 : 2}
    
    def set_solenoid_state(self, solenoid_id, new_state: bool):

        solenoid_id = self.solenoid_mask[solenoid_id]
        
        byte_index = solenoid_id // 8
        bit_in_byte = solenoid_id % 8

        while byte_index >= len(self.state):
            self.state.append(0b00000000)

        if new_state == True:
            self.state[byte_index] |= (1 << bit_in_byte)

        elif new_state == False:
            self.state[byte_index] &= ~(1 << bit_in_byte)

        self.update_solenoids()

    def update_solenoids(self):
        if self.test_UI:
            return
        self.DRV.spi.xfer2(self.state[::-1])


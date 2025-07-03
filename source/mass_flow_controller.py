'''
Copyright 2024-2025, the Active Cooling Experimental Application Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np

class MFC():

	def __init__(self, n_region, test_UI = False):

		self.test_UI = test_UI
		self.flow_rate = np.zeros(n_region)
		self.flow_rate_setpoint = np.zeros(n_region)

		if test_UI:
			return

		from source.TLA2825IRTER import TLA2528
		from adafruit_dacx578 import DACx578
		
		import board
		import busio

		i2c = busio.I2C(board.SCL, board.SDA)
			
		self.ADC = [TLA2528(address=0x12), TLA2528(address=0x13)]
		self.ADC_analog = np.zeros(n_region)
	
		self.DAC = [DACx578(i2c, address=0x48), DACx578(i2c, address=0x47)]

		self.n_region = n_region
		self.flow_rate = np.zeros(n_region)
	
	def get_analog_read(self):
		if self.test_UI:
			return
		
		n_points_ADC0 = min(8, self.n_region)
		self.ADC_analog[:n_points_ADC0] = self.ADC[0].measure_voltage()
		if self.n_region > 8:
			self.ADC_analog[n_points_ADC0:10] = self.ADC[1].measure_voltage()[:2]

	def get_flow_rate(self):
		if self.test_UI:
			return
		self.get_analog_read()
		
		for i in range(self.flow_rate.shape[0]):
			flow_rate = (self.ADC_analog[i] - 1) * 75 * 0.98
			
			if flow_rate <= 0:
				self.flow_rate[i] = 0
			else:
				self.flow_rate[i] = np.round(flow_rate, decimals=2)
				
	def set_flow_rate(self, region, flow_rate):
		if self.test_UI:
			return
		flow_rate = max(0., flow_rate)
		analog_input = flow_rate/75. + 1.
		if((analog_input <= 5.) and (analog_input > 1.)):
			self.flow_rate_setpoint[region] = (analog_input - 1.) * 75.
		elif analog_input > 5.:
			self.flow_rate_setpoint[region] = 5.
			analog_input = 5.
		else:
			self.flow_rate_setpoint[region] = 0.
			analog_input = 0.

		if region < 8:
			self.DAC[0].channels[region].normalized_value = analog_input / 5.
		else:
			self.DAC[1].channels[region - 8].normalized_value = analog_input / 5.

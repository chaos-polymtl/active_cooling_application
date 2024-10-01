#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np

class MFC():

	def __init__(self, n_region, test_UI = False):

		self.test_UI = test_UI
		self.flow_rate = np.zeros(n_region)
		self.flow_rate_setpoint = np.zeros(n_region)

		if test_UI:
			return

		from source.ADS1256 import ADS1256
		from source.DAC8532 import DAC8532
		from source.CD74HC4067 import CD74HC4067
		#import adafruit_pca9685
		#import busio
		import RPi.GPIO as GPIO
		
		self.ADC = ADS1256()
		
		self.ADC.ADS1256_init()
		
		self.DAC = DAC8532()
		self.multiplexer = CD74HC4067()

		self.DAC.DAC8532_Out_Voltage(0x30,0)
		self.DAC.DAC8532_Out_Voltage(0x34,0)
	
	def get_analog_read(self):
		if self.test_UI:
			return
		self.ADC_Value = self.ADC.ADS1256_GetAll()
		self.ADC_analog = []
		for i in range(len(self.ADC_Value)):
			self.ADC_analog.append((self.ADC_Value[i]*5.0/0x7fffff))

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
		#self.multiplexer.set_channel(region)
		flow_rate = max(0, flow_rate)
		analog_input = flow_rate/75 + 1
		if((analog_input <= 5) and (analog_input > 1)):
			self.flow_rate_setpoint[region] = (analog_input - 1) * 75
		else:
			self.flow_rate_setpoint[region] = 0
			analog_input = 0

		# TODO: adapt for multiplexer
		if region == 0:
		
			self.DAC.DAC8532_Out_Voltage(self.DAC.channel_A, analog_input)

		elif region == 1:
			self.DAC.DAC8532_Out_Voltage(self.DAC.channel_B, analog_input)

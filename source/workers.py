'''
Copyright 2024-2025, the Active Cooling Application Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from PySide6.QtCore import QObject, QTimer, QElapsedTimer, Signal, Slot
import numpy as np

# Define a worker class for measure and control logic
class MeasureAndControlWorker(QObject):

    update_ui_signal = Signal()
    stop_signal = Signal()
    flow_command_signal = Signal(object)

    def __init__(self, application):
        super().__init__()
        self.application = application
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.perform_measure_and_control)
        self.flow_command_signal.connect(self.set_flow_and_solenoid_states)

        self.elapsed_timer = QElapsedTimer()
        self.elapsed_timer.start()
        self.application.time = self.elapsed_timer.elapsed() / 1000
        self.application.previous_time = self.elapsed_timer.elapsed() / 1000

        # Triggers for time restart
        self.application.UI.scheduler_checkbox.checkStateChanged.connect(self.elapsed_timer.restart)
        self.application.UI.save_checkbox.checkStateChanged.connect(self.elapsed_timer.restart)

        # Define signal to communicate with main thread
        self.timer.start(500)

    def perform_measure_and_control(self):
        self.get_time()
        self.application.temperature.get_temperature()
        self.application.temperature.get_temperature_average(self.application.n_region, self.application.UI.region_boundaries)
        if not self.application.test_UI:
            self.application.MFC.get_flow_rate()
        self.apply_control()
        self.apply_scheduler()
        self.save_data()
        # Emit the signal to update the UI
        self.update_ui_signal.emit()

    def apply_control(self):
        '''Apply PID or MPC control to MFC flow rate'''

        # Apply PID control if enabled
        if self.application.UI.pid_temperature_checkbox.isChecked():
            current_flow_rate = self.application.MFC.flow_rate
            temperature_average = self.application.temperature.temperature_average
            temperature_setpoint = self.application.UI.temperature_setpoint
            time_step = self.application.time_step
            self.application.UI.time_step = self.application.time_step

            # Initialize a list to store pid_output for each region
            pid_outputs = np.zeros(self.application.n_region)

            # Apply flow rate increment to MFCs
            for j in range(self.application.n_region):
                # Calculate flow rate increment from PID controller
                pid_output = self.application.PID[j].compute_output(temperature_average[j], temperature_setpoint[j], time_step, current_flow_rate[j])
                pid_outputs[j] = pid_output
                if not self.application.UI.decoupler_checkbox.isChecked():
                    self.application.MFC.set_flow_rate(j, pid_output)
            
            # Apply decoupling terms if decoupler is enabled
            if self.application.UI.decoupler_checkbox.isChecked():
                for j in range(self.application.n_region):
                    decoupled_output = self.application.decouplers.compute_decoupled_output(pid_outputs)
                    self.application.MFC.set_flow_rate(j, decoupled_output[j])

        elif self.application.UI.mpc_temperature_checkbox.isChecked():
            # Only one region is active for MPC control
            n_region = 1

            # Compute average temperature for the active region
            avg_temperature = self.application.temperature.get_temperature_average(n_region, self.application.UI.region_boundaries)

            # Get the temperature setpoint and MPC parameters for the active region
            temperature_setpoint = self.application.UI.temperature_setpoint
            prediction_horizon = self.application.MPC.prediction_horizon
            control_horizon = self.application.MPC.control_horizon
            control_weight = self.application.MPC.control_weight
            dt = self.application.time_step

            # Compute the optimal flow rate using MPC
            ######## TO BE IMPLEMENTED ########
            # Placeholder for MPC computation
            control_output = np.ones(self.application.MFC.flow_rate.shape[0]) * 100

            # Apply control output to MFCs
            for j in range(self.application.MFC.flow_rate.shape[0]):
                self.application.MFC.set_flow_rate(j, control_output[j])


        # elif self.application.UI.mpc_temperature_checkbox.isChecked():
        #     avgT = self.application.temperature.get_temperature_average()[0]
        #     setpoint = self.application.UI.mpc_setpoint
        #     H = self.application.UI.mpc_prediction_horizon
        #     lam = self.application.UI.mpc_control_weight
        #     dt = self.application.UI.time_step

        #     # 1. Compute normalized arrangement in [-1, 1]^n
        #     arrangement = self.application.MPC.compute_arrangement(
        #         avg_temperature=avgT,
        #         setpoint=setpoint,
        #         prediction_horizon=H,
        #         control_weight=lam,
        #         dt=dt
        #     )

        #     # 2. Apply to actuators
        #     self.apply_mpc_arrangement(arrangement)

        #     # 3. Optional diagnostic
        #     print(f"[MPC] Arrangement: {np.round(arrangement, 2)}")


    def apply_mpc_arrangement(self, arrangement: np.ndarray):
        """
        Apply MPC arrangement to actuators:
        - negative value (port is an outlet) → solenoid open, MFC flow = 0
        - positive value (port is closed or an inlet) → solenoid closed, MFC flow = value * 300 L/min
        """
        n_actuators = len(arrangement)
        mfc = self.application.MFC
        solenoid = self.application.solenoid

        for j in range(n_actuators):
            val = float(arrangement[j])

            if val <= 0:
                # Negative or zero → exhaust (solenoid open), no inflow
                mfc.set_flow_rate(j, 0.0)
                solenoid.set_solenoid_state(j, True)
            else:
                # Positive → inflow (solenoid closed), scaled MFC flow
                flow_rate = val * 300.0
                mfc.set_flow_rate(j, flow_rate)
                solenoid.set_solenoid_state(j, False)

    def apply_scheduler(self):
        '''Apply scheduler to MFC flow rates and temperature setpoints'''    

        if self.application.UI.scheduler_checkbox.isChecked() and len(self.application.UI.scheduler_filename) > 1:
            change_time = self.application.UI.scheduler_change_time
            scheduled_flow_rates = self.application.UI.scheduler_data[0][1:]
            scheduled_temperature_setpoints = self.application.UI.scheduler_data[0][1:]
            
            if change_time > 0 and self.application.time >= change_time:
                self.application.UI.scheduler_data = np.delete(self.application.UI.scheduler_data, axis = 0, obj = 0)
                
                if self.application.UI.scheduler_data.shape[0] > 1:
                    self.application.UI.scheduler_change_time = self.application.UI.scheduler_data[1][0]
                    self.application.UI.scheduler_current_time.setText(str(self.application.UI.scheduler_data[0][0]) + " --- " + str(self.application.UI.scheduler_data[1][0]))

                else:
                    self.application.UI.scheduler_change_time = -1
                    self.application.UI.scheduler_current_time.setText(str(self.application.UI.scheduler_data[0][0]) + " --- end")

                # Print new current state: OUT for outlets, integer for inlets
                current = self.application.UI.scheduler_data[0][1:]
                outlet_print = [("OUT" if v == -1 else f"{int(v)}") for v in current]
                self.application.UI.scheduler_current_state.setText("[" + ", ".join(outlet_print) + "]")
      
            for j in range(self.application.n_region):                                      
                if self.application.UI.pid_temperature_checkbox.isChecked():
                    self.application.UI.temperature_setpoint[j] = scheduled_temperature_setpoints[j]
            else:
                # Delegate flow rate commands to set_flow_and_solenoid_states method
                self.set_flow_and_solenoid_states(scheduled_flow_rates)

    def set_flow_and_solenoid_states(self, flow_command):
        """
        Apply flow command for all regions.
        -1 => outlet (solenoid open, MFC 0, region_modes='outlet')
        >=0 => inlet  (solenoid closed, MFC=value clamped 0–300, region_modes='inlet')
        """
        for j, val in enumerate(flow_command):
            try:
                v = float(val)
            except Exception:
                v = 0.0

            if v < 0.0:
                # Outlet for any negative value
                self.application.region_modes[j] = "outlet"
                self.application.solenoid.set_solenoid_state(j, True)
                self.application.MFC.set_flow_rate(j, 0.0)
            else:
                # Inlet for zero or positive value
                self.application.region_modes[j] = "inlet"
                self.application.solenoid.set_solenoid_state(j, False)
                v = max(0.0, min(300.0, v))
                self.application.MFC.set_flow_rate(j, v)
                            
    def start_threads(self):
        # Create and start the thread for measure and control
        self.moveToThread(self.application.measure_and_control_thread)

        # Connect the worker's signal to the update_plot method
        self.update_ui_signal.connect(lambda: self.application.UI.update_plot(self.application.time, self.application.temperature, self.application.MFC))

        # Start the worker and the thread
        self.application.measure_and_control_thread.start()

    @Slot()
    def stop(self):
        """Stop the worker’s QTimer from within the worker thread."""
        # Guard against double calls
        if getattr(self, "_stopped", False):
            return
        self._stopped = True

        try:
            if self.timer.isActive():
                self.timer.stop()
        except Exception as e:
            print(e)
            
    def get_time(self):
        self.application.time = self.elapsed_timer.elapsed() / 1000
        self.application.time_step = self.elapsed_timer.elapsed() / 1000 - self.application.previous_time
        self.application.previous_time = self.application.time

    def save_data(self):
        if self.application.UI.save_mode:
            if self.application.UI.pid_temperature_checkbox.isChecked():
                self.save_temperature_array = np.zeros(1 + len(self.application.temperature.temperature))              
                self.save_data_array = np.zeros(1 + 10 * self.application.n_region)
                self.save_data_array[2*self.application.n_region + 1 : 3*self.application.n_region + 1] = self.application.UI.temperature_setpoint
                for i in range(self.application.n_region):
                    for j in range(3):
                        self.save_data_array[(3+j)*self.application.n_region + 1 + i] = self.application.UI.PID[i].gains[j]

                    data_indexing = 1+ 6*self.application.n_region + (i*4)
                    self.save_data_array[data_indexing : data_indexing +4] = self.application.UI.region_boundaries[i]

            else:
                self.save_data_array = np.zeros(1 + 6* self.application.n_region)
                self.save_temperature_array = np.zeros(1 + len(self.application.temperature.temperature))

                for i in range(self.application.n_region):
                    data_indexing = 1+ 2*self.application.n_region + (i*4)
                    self.save_data_array[data_indexing : data_indexing +4] = self.application.UI.region_boundaries[i]

            self.save_data_array[0] = self.application.time
            self.save_temperature_array[0] = self.application.time
            
            if not self.application.test_UI:
                self.save_data_array[1:self.application.n_region + 1] = self.application.MFC.flow_rate

            self.save_data_array[self.application.n_region + 1 : self.application.n_region * 2 + 1] = self.application.temperature.temperature_average
            self.save_temperature_array[1:] = self.application.temperature.temperature

            self.save_data_array = self.save_data_array.reshape(1, -1)
            self.save_temperature_array = self.save_temperature_array.reshape(1, -1)

            with open(self.application.UI.filename, 'a') as file:
                np.savetxt(file, self.save_data_array, delimiter=',', fmt='%10.5f')

            with open(self.application.UI.filename.replace('.csv', '_temp.csv'), 'a') as file:
                np.savetxt(file, self.save_temperature_array, delimiter = ',', fmt = '%10.5f')
        

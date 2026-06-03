'''
Copyright 2024-2025, the Active Cooling Application Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from PySide6.QtCore import QObject, QTimer, QElapsedTimer, Signal, Slot, QThread
import numpy as np
import os

from source import experimental_mpc

 # Define a worker class for the MPC optimization logic
class MPCWorker(QObject):
    """Runs one MPC solve on a dedicated thread; emits result when done."""
    result_ready = Signal(object)  # emit flow_command np.ndarray when MPC solve is done

    def __init__(self, mpc_controller):
        super().__init__()
        self._mpc = mpc_controller
        self._busy = False

    @property
    def busy(self):
        return self._busy
    
    @Slot(object, object, object)
    def solve(self, temp_vec, temperature_shape, flow_rates):
        self._busy = True
        try:
            self._mpc.skip_heat_load_reconstruction = False

            Q0, cost, predicted_temps = self._mpc.compute_mpc_control_action(
                current_temperatures=temp_vec,
                temperature_shape=temperature_shape,
                current_flow_rates=flow_rates
            )
            # Convert Q0 [-1, 1] into hardware-ready values
            Q0_clean = Q0.copy()
            # Treat anything within noise threshold of zero as exactly zero
            Q0_clean[np.abs(Q0_clean) < 1e-6] = 0.0
            flow_command = np.where(Q0_clean < 0, -1.0, Q0_clean * 300.0)

            # Build full Q sequence and predicted avg temperatures over horizon
            N = self._mpc.mpc_prediction_horizon
            Q_opt_sequence = self._mpc.previous_Q_sequence  # shape (N, D), stored by compute_mpc_control_action

            sorted_steps = sorted(predicted_temps.keys())
            T_pred_avgs = np.array(
                [float(np.mean(predicted_temps[k])) for k in sorted_steps],
                dtype=float
            )  # shape (N,) — spatial mean of top-face temperature at each prediction step

            T_pred_fields = np.array(
                [predicted_temps[k] for k in sorted_steps],
                dtype=float
            )  # shape (N, M) — full top-face temperature field at each prediction step

            # Reconstructed h (mean over face, scalar) — stored by adjoint in MPC controller
            h_reconstructed_mean = float(getattr(self._mpc, '_last_h_reconstructed_mean', np.nan))

            self.result_ready.emit({
                'flow_command':        flow_command,
                'Q_opt_sequence':      Q_opt_sequence,
                'T_pred_avgs':         T_pred_avgs,
                'T_pred_fields':       T_pred_fields,
                'h_reconstructed_mean': h_reconstructed_mean,
                'cost':                cost,
                'h_reconstructed':     self._mpc._last_h_reconstructed,
                'adjoint_error':       getattr(self._mpc, '_last_adjoint_error', np.nan),
                'adjoint_iterations':  getattr(self._mpc, '_last_adjoint_iterations', np.nan),
            })

            print(f"[MPC] solve done, applying flow command: {flow_command}, cost: {cost:.2f}")
        except Exception as e:
            print(f"[MPC] solve error: {e}")
        finally:
            self._busy = False

# Define a worker class for measure and control logic
class MeasureAndControlWorker(QObject):

    update_ui_signal = Signal()
    stop_signal = Signal()
    flow_command_signal = Signal(object)
    mpc_solve_requested = Signal(object, object, object)  # temp_vec, temperature_shape, flow_rates

    def __init__(self, application):
        super().__init__()
        self.application = application
        self.timer = None # QTimer will be created in start_timer() to ensure it lives in the correct thread
        self.flow_command_signal.connect(self.set_flow_and_solenoid_states)

        self.elapsed_timer = QElapsedTimer()
        self.elapsed_timer.start()
        self.application.time = self.elapsed_timer.elapsed() / 1000
        self.application.previous_time = self.elapsed_timer.elapsed() / 1000

        # Triggers for time restart
        self.application.UI.scheduler_checkbox.checkStateChanged.connect(self.elapsed_timer.restart)
        self.application.UI.save_checkbox.checkStateChanged.connect(self.elapsed_timer.restart)

        # Connect the worker's signal to the update_plot method
        self.update_ui_signal.connect(lambda: self.application.UI.update_plot(self.application.time, self.application.temperature, self.application.MFC, self.application.region_modes, self.application.last_flow_command))

        # MPC thread setup
        self._mpc_thread = QThread()
        self._mpc_worker = MPCWorker(self.application.MPC)
        self._mpc_worker.moveToThread(self._mpc_thread)

        # Wire solve request signal from main thread to MPC worker (cross-thread, queued automatically)
        self.mpc_solve_requested.connect(self._mpc_worker.solve)
        # Wire MPC result signal back to main thread handler to apply flow command (queued into this thread)
        self._mpc_worker.result_ready.connect(self._on_mpc_result)

        self._mpc_thread.start()

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

        # Apply MPC control if enabled
        elif self.application.UI.mpc_temperature_checkbox.isChecked():
            
            # ####################################
            # 1) MPC updates at a selected time interval
            # ####################################
            dt = float(self.application.MPC.time_step)
            t = self.application.time

            if not hasattr(self, "next_mpc_time"):
                self.next_mpc_time = t

            if t < self.next_mpc_time:
                return
            
            self.next_mpc_time += dt

            # Skip if MPC worker is still busy from previous solve
            if self._mpc_worker.busy:
                print("[MPC] solve still in progress at t={:.2f}s, skipping trigger this cycle".format(t))
                return
            
            # Snapshot current state for MPC inputs
            grid = self.application.temperature.temperature_grid # full camera grid

            # region 0 boundaries (from GUI)
            x_min, x_max, y_min, y_max = self.application.UI.region_boundaries[0]

            # clamping
            H_full, W_full = grid.shape
            x_min = max(0, min(x_min, W_full - 1))
            x_max = max(0, min(x_max, W_full - 1))
            y_min = max(0, min(y_min, H_full - 1))
            y_max = max(0, min(y_max, H_full - 1))

            # ensure ordering
            if x_max < x_min: x_max = x_min
            if y_max < y_min: y_max = y_min

            # subregion
            subgrid = grid[y_min:y_max+1, x_min:x_max+1]
            H_sub, W_sub = subgrid.shape

            temp_vec = subgrid.flatten(order="C").copy()
            flow_snapshot = self.application.MFC.flow_rate.copy()

            print(f"[MPC] Triggering solve at t={t:.2f}s")
            self.mpc_solve_requested.emit(temp_vec, (H_sub, W_sub), flow_snapshot)

    @Slot(object)
    def _on_mpc_result(self, result:dict):
        """Apply the MPC result at the moment it is received from the MPC worker thread."""
        flow_command = result['flow_command']
        self.set_flow_and_solenoid_states(flow_command)
        self.flow_command_signal.emit(flow_command)  # also emit to main thread if needed for UI display
        # Cache MPC extras for save_data to log them
        self._last_mpc_result = result
        self._new_mpc_result_available = True  # flag so Tfield file is written exactly once per solve

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
        self.application.last_flow_command =np.array(flow_command, copy=True)

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
        self.application.measure_and_control_thread.started.connect(self._start_timer)
        # Start the worker and the thread
        self.application.measure_and_control_thread.start()

    @Slot()
    def _start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.perform_measure_and_control)
        self.flow_command_signal.connect(self.set_flow_and_solenoid_states)
        self.timer.start(500)

    @Slot()
    def stop(self):
        """Stop the worker’s QTimer from within the worker thread."""
        # Guard against double calls
        if getattr(self, "_stopped", False):
            return
        self._stopped = True

        try:
            if self.timer and self.timer.isActive():
                self.timer.stop()
        except Exception as e:
            print(e)

        self._mpc_thread.quit()
        self._mpc_thread.wait(3000)
            
    def get_time(self):
        self.application.time = self.elapsed_timer.elapsed() / 1000
        self.application.time_step = self.elapsed_timer.elapsed() / 1000 - self.application.previous_time
        self.application.previous_time = self.application.time

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_row(filename, headers, values, fmt='.6f'):
        """Write *headers* once (when file is new) then append *values* as a CSV row."""
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(','.join(headers) + '\n')
        row = ','.join(format(v, fmt) for v in values) + '\n'
        with open(filename, 'a') as f:
            f.write(row)

    def _sensor_headers(self):
        """Column names for the main sensor CSV, matching the array layout in save_data."""
        nr = self.application.n_region
        headers = ['time']
        headers += [f'mfc_{i}' for i in range(nr)]
        headers += [f'T_avg_region{i}' for i in range(nr)]
        if self.application.UI.pid_temperature_checkbox.isChecked():
            headers += [f'T_set_region{i}' for i in range(nr)]
            for label in ['P', 'I', 'D']:
                headers += [f'{label}_region{i}' for i in range(nr)]
        for i in range(nr):
            headers += [f'roi{i}_x_min', f'roi{i}_x_max', f'roi{i}_y_min', f'roi{i}_y_max']
        if not self.application.UI.pid_temperature_checkbox.isChecked():
            headers += [f'sol_{i}' for i in range(9)]
        return headers

    def save_data(self):
        if not self.application.UI.save_mode:
            return

        nr  = self.application.n_region
        pid = self.application.UI.pid_temperature_checkbox.isChecked()

        # ---- build sensor data array (same logic as before) ----
        if pid:
            self.save_data_array = np.zeros(1 + 10 * nr)
            self.save_data_array[2*nr + 1 : 3*nr + 1] = self.application.UI.temperature_setpoint
            for i in range(nr):
                for j in range(3):
                    self.save_data_array[(3 + j)*nr + 1 + i] = self.application.UI.PID[i].gains[j]
                data_indexing = 1 + 6*nr + i*4
                self.save_data_array[data_indexing : data_indexing + 4] = self.application.UI.region_boundaries[i]
        else:
            n_sol = 9
            self.save_data_array = np.zeros(1 + 6*nr + n_sol)
            for i in range(nr):
                data_indexing = 1 + 2*nr + i*4
                self.save_data_array[data_indexing : data_indexing + 4] = self.application.UI.region_boundaries[i]

        self.save_data_array[0] = self.application.time
        if not self.application.test_UI:
            self.save_data_array[1 : nr + 1] = self.application.MFC.flow_rate
        self.save_data_array[nr + 1 : nr*2 + 1] = self.application.temperature.temperature_average

        if not pid:
            solenoid_states = self.application.solenoid.get_solenoid_states()
            self.save_data_array[-9:] = [int(s) for s in solenoid_states[:9]]

        # ---- build camera temperature array ----
        n_cam = len(self.application.temperature.temperature)
        self.save_temperature_array = np.zeros(1 + n_cam)
        self.save_temperature_array[0]  = self.application.time
        self.save_temperature_array[1:] = self.application.temperature.temperature

        # ----------------------------------------------------------------
        # 1) sensors.csv  — one row per control loop tick (500 ms)
        # ----------------------------------------------------------------
        sensor_file = self.application.UI.filename
        sensor_headers = self._sensor_headers()
        if not os.path.exists(sensor_file):
            with open(sensor_file, 'w') as f:
                f.write(','.join(sensor_headers) + '\n')
        with open(sensor_file, 'a') as f:
            np.savetxt(f, self.save_data_array.reshape(1, -1), delimiter=',', fmt='%10.5f')

        # ----------------------------------------------------------------
        # 2) sensors_Tcam.csv  — full camera grid, one row per tick
        # ----------------------------------------------------------------
        tcam_file = sensor_file.replace('.csv', '_Tcam.csv')
        if not os.path.exists(tcam_file):
            tcam_headers = ['time'] + [f'T_cam_{i}' for i in range(n_cam)]
            with open(tcam_file, 'w') as f:
                f.write(','.join(tcam_headers) + '\n')
        with open(tcam_file, 'a') as f:
            np.savetxt(f, self.save_temperature_array.reshape(1, -1), delimiter=',', fmt='%10.5f')

        # ----------------------------------------------------------------
        # MPC-specific files (only when MPC checkbox is active)
        # ----------------------------------------------------------------
        if not self.application.UI.mpc_temperature_checkbox.isChecked():
            return

        mpc_result = getattr(self, '_last_mpc_result', None)
        N = self.application.UI.mpc_prediction_horizon
        D = self.application.n_region

        if mpc_result is not None:
            Q_opt   = mpc_result['Q_opt_sequence']       # (N, D)
            T_avgs  = mpc_result['T_pred_avgs']          # (N,)
            h_mean  = mpc_result['h_reconstructed_mean'] # scalar
            adj_err = mpc_result['adjoint_error']
            adj_it  = mpc_result['adjoint_iterations']
            h_full  = mpc_result['h_reconstructed']      # (M,)
            T_fields = mpc_result.get('T_pred_fields')   # (N, M) or None
        else:
            Q_opt    = np.full((N, D), np.nan)
            T_avgs   = np.full(N, np.nan)
            h_mean   = np.nan
            adj_err  = np.nan
            adj_it   = np.nan
            h_full   = np.full(1, np.nan)
            T_fields = None

        t = self.application.time

        # ----------------------------------------------------------------
        # 3) mpc_control.csv  — Q sequence + predicted avg T + adjoint info
        #    one row per control loop tick (carries last known MPC solve)
        # ----------------------------------------------------------------
        mpc_ctrl_file = sensor_file.replace('.csv', '_mpc_control.csv')
        mpc_ctrl_headers = (
            ['time']
            + [f'Q_step{n}_jet{d}' for n in range(N) for d in range(D)]
            + [f'T_pred_avg_step{n}' for n in range(N)]
            + ['h_recon_mean', 'adjoint_error', 'adjoint_iters']
        )
        mpc_ctrl_row = (
            [t]
            + [float(Q_opt[n, d]) if n < Q_opt.shape[0] else np.nan for n in range(N) for d in range(D)]
            + [float(T_avgs[n]) if n < len(T_avgs) else np.nan for n in range(N)]
            + [h_mean, adj_err, adj_it]
        )
        self._write_row(mpc_ctrl_file, mpc_ctrl_headers, mpc_ctrl_row)

        # ----------------------------------------------------------------
        # 4) mpc_h.csv  — full reconstructed h map
        #    one row per control loop tick (carries last known MPC solve)
        # ----------------------------------------------------------------
        h_flat = np.asarray(h_full).flatten()
        mpc_h_file = sensor_file.replace('.csv', '_mpc_h.csv')
        mpc_h_headers = ['time'] + [f'h_node{i}' for i in range(len(h_flat))]
        self._write_row(mpc_h_file, mpc_h_headers, [t] + h_flat.tolist())

        # ----------------------------------------------------------------
        # 5) mpc_Tfield.csv  — full predicted temperature field per step
        #    written ONCE per MPC solve (not repeated every 500 ms tick)
        # ----------------------------------------------------------------
        if getattr(self, '_new_mpc_result_available', False) and T_fields is not None:
            N_actual, M = T_fields.shape
            mpc_tfield_file = sensor_file.replace('.csv', '_mpc_Tfield.csv')
            mpc_tfield_headers = (
                ['time']
                + [f'T_step{n}_node{m}' for n in range(N_actual) for m in range(M)]
            )
            tfield_row = [t] + T_fields.flatten().tolist()
            self._write_row(mpc_tfield_file, mpc_tfield_headers, tfield_row)
            self._new_mpc_result_available = False

    def shutdown(self):
        """
        Stop all flows and open one solenoid to release pressure when the application is closed.
        """
        print("Shutting down: stopping flows and opening solenoids.")

        zero_flow_command = np.zeros(9) # All zeros, inlet mode, solenoid closed TODO: make function of actuators (so either 5 or 9 for our systems)
        zero_flow_command[-1] = -1.0 # Last actuator as outlet, solenoid open

        self.set_flow_and_solenoid_states(zero_flow_command)
        

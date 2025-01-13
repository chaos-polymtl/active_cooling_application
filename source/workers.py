from PySide6.QtCore import QObject, QTimer, QElapsedTimer, Signal, Slot
import numpy as np

# Define a worker class for measure and control logic
class MeasureAndControlWorker(QObject):

    update_ui_signal = Signal()
    stop_signal = Signal()

    def __init__(self, application):
        super().__init__()
        self.application = application
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.perform_measure_and_control)

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
        '''Apply PID control to MFCs flow rate'''
        if self.application.UI.mfc_temperature_checkbox.isChecked():

            current_flow_rate = self.application.MFC.flow_rate
            temperature_average = self.application.temperature.temperature_average
            temperature_setpoint = self.application.UI.temperature_setpoint
            time_step = self.application.time_step
            
            # Apply flow rate increment to MFCs
            # TODO: Add other MFCS
            for j in range(self.application.n_region):
                # Calculate flow rate increment from PID controller
                pid_output = self.application.PID[j].compute_output(temperature_average[j], temperature_setpoint[j], time_step, current_flow_rate[j])

                self.application.MFC.set_flow_rate(j, pid_output)

    def apply_scheduler(self):
        '''Apply scheduler to MFCs flow rate'''    

        if self.application.UI.scheduler_checkbox.isChecked() and len(self.application.UI.scheduler_filename) > 1:
            change_time = self.application.UI.scheduler_change_time
            scheduled_flow_rates = self.application.UI.scheduler_data[0][1:]
            
            if change_time > 0 and self.application.time >= change_time:
                self.application.UI.scheduler_data = np.delete(self.application.UI.scheduler_data, axis = 0, obj = 0)
                
                if self.application.UI.scheduler_data.shape[0] > 1:
                    self.application.UI.scheduler_change_time = self.application.UI.scheduler_data[1][0]
                    self.application.UI.scheduler_current_time.setText(str(self.application.UI.scheduler_data[0][0]) + " --- " + str(self.application.UI.scheduler_data[1][0]))

                else:
                    self.application.UI.scheduler_change_time = -1
                    self.application.UI.scheduler_current_time.setText(str(self.application.UI.scheduler_data[0][0]) + " --- end")

                self.application.UI.scheduler_current_flow_rate.setText(str(self.application.UI.scheduler_data[0][1:]))                                        

            for j in range(self.application.n_region):
                self.application.MFC.set_flow_rate(j, scheduled_flow_rates[j])

                            
    def start_threads(self):
        # Create and start the thread for measure and control
        self.moveToThread(self.application.measure_and_control_thread)

        # Connect the worker's signal to the update_plot method
        self.update_ui_signal.connect(lambda: self.application.UI.update_plot(self.application.time, self.application.temperature, self.application.MFC))

        # Start the worker and the thread
        self.application.measure_and_control_thread.start()

    def get_time(self):
        self.application.time = self.elapsed_timer.elapsed() / 1000
        self.application.time_step = self.elapsed_timer.elapsed() / 1000 - self.application.previous_time
        self.application.previous_time = self.application.time

    def save_data(self):
        if self.application.UI.save_mode:
            self.save_data_array = np.zeros(1 + 2 * self.application.n_region)
            self.save_temperature_array = np.zeros(1 + len(self.application.temperature.temperature))
            self.save_data_array[0] = self.application.time
            self.save_temperature_array[0] = self.application.time
            if not self.application.test_UI:
                self.save_data_array[1:self.application.n_region + 1] = self.application.MFC.flow_rate
            self.save_data_array[self.application.n_region + 1:self.application.n_region * 2 + 1] = self.application.temperature.temperature_average
            self.save_temperature_array[1:] = self.application.temperature.temperature
            self.save_data_array = self.save_data_array.reshape(1, -1)
            self.save_temperature_array = self.save_temperature_array.reshape(1, -1)
            with open(self.application.UI.filename, 'a') as file:
                np.savetxt(file, self.save_data_array, delimiter=',', fmt='%10.5f')

            with open(self.application.UI.filename.replace('.csv', '_temp.csv'), 'a') as file:
                np.savetxt(file, self.save_temperature_array, delimiter = ',', fmt = '%10.5f')
        

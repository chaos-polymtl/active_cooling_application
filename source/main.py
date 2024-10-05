from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QElapsedTimer, QThread, Signal, Slot, QObject
from PySide6.QtGui import QIcon
import sys
import os
import numpy as np
from source.UI import UI

# Define a worker class for measure and control logic
class MeasureAndControlWorker(QObject):
    # Define signals to communicate with the main thread
    update_ui_signal = Signal()

    def __init__(self, application):
        super().__init__()
        self.application = application
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.perform_measure_and_control)

    def start(self):
        # Start the QTimer to run every second (1000 ms)
        self.timer.start(1000)

    def perform_measure_and_control(self):
        self.application.get_time()
        self.application.temperature.get_temperature()
        if not self.application.test_UI:
            self.application.MFC.get_flow_rate()
        self.application.apply_control()
        self.application.save_data()
        # Emit the signal to update the UI
        self.update_ui_signal.emit()

class Application(UI):
    def __init__(self, n_region=2, test_UI=False):
        super().__init__()
        application_dir = os.path.dirname(os.path.abspath(__file__))

        # Set application style
        with open(f"{application_dir}/style.qss", "r") as f:
            _style = f.read()
            self.setStyleSheet(_style)

        # Set window icon
        window_icon = QIcon(f"{application_dir}/nrc.png")
        self.setWindowIcon(window_icon)

        self.test_UI = test_UI
        self.init_UI(n_region, test_UI)
        
        self.start_threads()

    def start_threads(self):
        self.elapsed_timer = QElapsedTimer()
        self.elapsed_timer.start()
        self.initial_time = self.elapsed_timer.elapsed() / 1000
        self.previous_time = self.elapsed_timer.elapsed() / 1000

        # Create and start the thread for measure and control
        self.measure_and_control_thread = QThread()
        self.worker = MeasureAndControlWorker(self)
        self.worker.moveToThread(self.measure_and_control_thread)

        # Connect the worker's signal to the update_plot method
        self.worker.update_ui_signal.connect(self.update_plot)

        # Start the worker and the thread
        self.measure_and_control_thread.started.connect(self.worker.start)
        self.measure_and_control_thread.start()

    def get_time(self):
        self.time = self.elapsed_timer.elapsed() / 1000
        self.time_step = self.elapsed_timer.elapsed() / 1000 - self.previous_time
        self.previous_time = self.time

    def save_data(self):
        if self.save_mode:
            self.save_data_array = np.zeros(1 + 2 * self.n_region + self.temperature.resolution[0] * self.temperature.resolution[1])
            self.save_data_array[0] = self.time
            if not self.test_UI:
                self.save_data_array[1:self.n_region + 1] = self.MFC.flow_rate
            self.save_data_array[self.n_region + 1:self.n_region * 2 + 1] = self.temperature_average
            self.save_data_array[self.n_region * 2 + 1:] = self.temperature.temperature
            self.save_data_array = self.save_data_array.reshape(1, -1)
            with open(self.filename, 'w') as file:
                np.savetxt(file, self.save_data_array, delimiter=',', fmt='%10.5f')

    @staticmethod
    def run():
        app = QApplication(sys.argv)        
        n_region = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        window = Application(n_region=n_region)
        window.show()
        sys.exit(app.exec())

    @staticmethod
    def run_test():
        app = QApplication(sys.argv)
        n_region = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        window = Application(n_region=n_region, test_UI=True)
        window.show()
        sys.exit(app.exec())

if __name__ == '__main__':
    Application.run()

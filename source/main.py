# Import pip modules

import sys
import os
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer, QElapsedTimer
from PySide6.QtGui import QIcon
from source.UI import UI

class Application(UI):
    def __init__(self, n_region=5, test_UI=False):
        super().__init__()

        # Set application style
        with open("style.qss", "r") as f:
            _style = f.read()
            self.setStyleSheet(_style)
        
        self.test_UI = test_UI
        self.init_UI(n_region, test_UI)
        
        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), 'nrc.png')
        self.setWindowIcon(QIcon(icon_path))
        
        self.start_timer()

    def start_timer(self):
        self.timer = QTimer()
        self.elapsed_timer = QElapsedTimer()
        self.elapsed_timer.start()
        self.initial_time = self.elapsed_timer.elapsed() / 1000
        self.previous_time = self.elapsed_timer.elapsed() / 1000
        self.timer.setInterval(1000)  # Interval between acquisition in milliseconds
        self.timer.timeout.connect(self.get_time)
        self.timer.timeout.connect(self.get_time_step)
        self.timer.timeout.connect(self.thermal_cam.get_temperature)
        if not self.test_UI:
            self.timer.timeout.connect(self.MFC.get_flow_rate)
        self.timer.timeout.connect(self.apply_control)
        self.timer.timeout.connect(self.update_plot)
        self.timer.timeout.connect(self.save_data)
        self.timer.start()

    def get_time(self):
        self.time = self.elapsed_timer.elapsed() / 1000
        
    def get_time_step(self):
        self.time_step = self.elapsed_timer.elapsed() / 1000 - self.previous_time
        self.previous_time = self.time
        
    def save_data(self):
        if self.save_mode:
            self.save_data_array = np.zeros(1 + 2 * self.n_region + self.thermal_cam.resolution[0] * self.thermal_cam.resolution[1])
            self.save_data_array[0] = self.time
            if not self.test_UI:
                self.save_data_array[1:self.n_region + 1] = self.MFC.flow_rate
            self.save_data_array[self.n_region + 1:self.n_region * 2 + 1] = self.temperature_average
            self.save_data_array[self.n_region * 2 + 1:] = self.thermal_cam.temperature
            self.save_data_array = self.save_data_array.reshape(1, -1)
            with open(self.filename, 'a') as file:
                np.savetxt(file, self.save_data_array, delimiter=',', fmt='%10.5f')

    @staticmethod
    def run():
        # TODO: Add a parser to get the number of regions from the command line
        # TODO: Fix n_region
        app = QApplication(sys.argv)
        if len(sys.argv) > 1:
            n_region = int(sys.argv[1])
        else:
            n_region = 5

        window = Application(n_region=n_region, test_UI=True)
        window.show()
        app.exec()

if __name__ == '__main__':
    Application.run()

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer, QElapsedTimer, QThread
from PySide6.QtGui import QIcon
import sys
import os
import numpy as np

from source.UI import UI
from source.temperature import Temperature
from source.mass_flow_controller import MFC
from source.pid_controller import PIDControl
from source.decouplers import decouplers
from source.workers import MeasureAndControlWorker

class Application(QMainWindow):
    def __init__(self, n_region=2, test_UI=False):
        super().__init__()

        self.n_region = n_region
        self.test_UI = test_UI
        
        application_dir = os.path.dirname(os.path.abspath(__file__))

        # Set application style
        with open(f"{application_dir}/style.qss", "r") as f:
            _style = f.read()
            self.setStyleSheet(_style)

        # Set window icon
        window_icon = QIcon(f"{application_dir}/nrc.png")
        self.setWindowIcon(window_icon)

        # Create temperature instance
        self.temperature = Temperature(n_region, self.test_UI)

        # Create MFC instance
        self.MFC = MFC(n_region, test_UI)

        # Create control objects
        self.PID = []
        for j in range(n_region):
            self.PID.append(PIDControl())
        
        self.decouplers = decouplers()

        # Create UI instance
        self.UI = UI()
        self.UI.init_UI(temperature = self.temperature, MFC = self.MFC, PID = self.PID, n_region = n_region, test_UI = test_UI)
        self.setCentralWidget(self.UI)
        
        self.measure_and_control_thread = QThread()
        self.measure_and_control_worker = MeasureAndControlWorker(self)
        self.measure_and_control_worker.start_threads()

        self.measure_and_control_worker.stop_signal.connect(self.measure_and_control_thread.quit)
        self.measure_and_control_thread.finished.connect(self.measure_and_control_worker.deleteLater)

    def closeEvent(self, event):
        
        # Stop the worker's timer
        self.measure_and_control_worker.stop_signal.emit()

        # Disconnect any signals to prevent access to deleted objects
        self.measure_and_control_worker.update_ui_signal.disconnect()

        # Zero MFCs flow rate
        for j in range(self.n_region):
            self.MFC.set_flow_rate(j, 0)

        # Allow the application to close
        event.accept()

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

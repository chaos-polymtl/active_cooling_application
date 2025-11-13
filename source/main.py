'''
Copyright 2024-2025, the Active Cooling Application Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer, QElapsedTimer, QThread
from PySide6.QtGui import QIcon
import sys
import os
import argparse
import numpy as np

from source.UI import UI
from source.solenoid_valve import Solenoid
from source.temperature import Temperature
from source.mass_flow_controller import MFC
from source.pid_controller import PIDControl
from source.decouplers import decouplers
from source.workers import MeasureAndControlWorker

class Application(QMainWindow):
    def __init__(self, n_region=1, test_UI=False):
        super().__init__()

        self.n_region = n_region
        self.test_UI = test_UI
        
        application_dir = os.path.dirname(os.path.abspath(__file__))

        # Set application style
        with open(f"{application_dir}/style.qss", "r") as f:
            _style = f.read()
            self.setStyleSheet(_style)

        # Create solenoid valve instance
        self.solenoid = Solenoid(n_region, self.test_UI)

        # Create temperature instance
        self.temperature = Temperature(n_region, self.test_UI)

        # Create MFC instance
        self.MFC = MFC(n_region, test_UI)

        # Create control objects
        self.PID = []
        for j in range(n_region):
            self.PID.append(PIDControl())
        
        self.decouplers = decouplers()

        # Each region can be an "inlet" or an "outlet"
        self.region_modes = np.array(["inlet"] * n_region)

        # Create UI instance
        self.UI = UI()
        self.UI.init_UI(solenoid = self.solenoid, temperature = self.temperature, MFC = self.MFC, PID = self.PID, n_region = n_region, test_UI = test_UI)
        self.UI.application = self
        self.setCentralWidget(self.UI)
        
        self.measure_and_control_thread = QThread()
        self.measure_and_control_worker = MeasureAndControlWorker(self)
        self.measure_and_control_worker.start_threads()

        self.measure_and_control_worker.stop_signal.connect(self.measure_and_control_worker.stop)
        self.measure_and_control_worker.stop_signal.connect(self.measure_and_control_thread.quit)
        self.measure_and_control_thread.finished.connect(self.measure_and_control_worker.deleteLater)

    def closeEvent(self, event):
        """Stop worker thread and close the application."""

        # --- Stop the worker safely in its own thread ---
        try:
            if hasattr(self, "measure_and_control_worker"):
                # Emit stop signal (worker.stop() runs inside worker thread)
                self.measure_and_control_worker.stop_signal.emit()
        except Exception as e:
            print("Warning: failed to emit stop signal", e)

        # --- Tell the thread to quit and wait ---
        try:
            if hasattr(self, "measure_and_control_thread"):
                self.measure_and_control_thread.quit()
                self.measure_and_control_thread.wait(2000)
        except Exception as e:
            print("Warning: failed to quit thread", e)

        # --- Zero MFCs and solenoids for safety ---
        try:
            for j in range(self.n_region):
                self.MFC.set_flow_rate(j, 0)
                self.solenoid.set_solenoid_state(j, False)
        except Exception:
            pass

        # --- Accept event ---
        event.accept()

    @staticmethod
    def run():
        app = QApplication(sys.argv)      
        parser = argparse.ArgumentParser(description="Run the Active Cooling Application")
        parser.add_argument("n_region", help="Number of regions", type=int, default=9, nargs='?')
        args = parser.parse_args()
        window = Application(n_region=args.n_region, test_UI=False)
        window.show()
        sys.exit(app.exec())

    @staticmethod
    def run_test():
        app = QApplication(sys.argv)
        parser = argparse.ArgumentParser(description="Run the Active Cooling Application on test mode")
        parser.add_argument("n_region", help="Number of regions", type=int, default=9, nargs='?')
        args = parser.parse_args()
        window = Application(n_region=args.n_region, test_UI=True)
        window.show()
        sys.exit(app.exec())

if __name__ == '__main__':
    Application.run()

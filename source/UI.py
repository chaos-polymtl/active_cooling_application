'''
Copyright 2024-2025, the Active Cooling Experimental Application Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import os
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QComboBox, QCheckBox, QPushButton, QFileDialog, QPushButton, QGridLayout, QFrame, QTextEdit
from PySide6.QtGui import QFont
from matplotlib import patches, use
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.style as mplstyle
mplstyle.use('fast')
use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


# ================== Set matplotlib style ==================
from cycler import cycler

colors=[
    "#006e00",  # green
    "#00bbad",  # caribbean
    "#d163e6",  # lavender
    "#b24502",  # brown
    "#ff9287",  # coral
    "#5954d6",  # indigo
    "#00c6f8",  # turquoise
    "#878500",  # olive
    "#00a76c",  # jade
    "#000000"   # black
]

plt.rcParams['axes.prop_cycle'] = cycler(color = colors)

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 80
plt.rcParams['figure.autolayout'] = True

plt.rcParams['lines.markersize'] = '11'
plt.rcParams['lines.markeredgewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 4

plt.rcParams['legend.frameon'] = True

plt.rcParams['legend.fancybox'] = False
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['font.size'] = '12'

# ================== User interface ==================
class UI(QWidget):
    # Create user interface
    def init_UI(self, temperature, MFC, PID, n_region, test_UI = False):       

        # Set lower limit to temperature setpoint
        # TODO: make widget for this
        self.temperature_setpoint_lower_limit = 50

        self.time_step = 1
        
        # Set number of regions
        self.n_region = n_region

        # Set number of controller parameters
        self.n_controller_parameters = 3

        self.temperature = temperature
        self.MFC = MFC
        self.PID = PID
        
        # Colors for plots
        self.colors_qualitative = colors

        # Name of the window                
        self.setWindowTitle("Active Cooling")

        # Set main self.layout
        self.layout = QVBoxLayout()
        self.layout.setSpacing(3)

        # ================== Save mode section ==================
        # Set title        
        title = QLabel("Save controller:")
        self.layout.addWidget(title)

        # Create a save state layout
        save_state_layout = QHBoxLayout()
        self.layout.addLayout(save_state_layout)

        # Set default state filename
        self.state_filename = ''

        # Create a push button to save state
        save_state_button = QPushButton('Save state')
        save_state_button.clicked.connect(self.save_state)

        # Create a push button to load state
        load_state_button = QPushButton('Load state')
        load_state_button.clicked.connect(self.load_state)

        # Create a QLineEdit widget to display the state filename
        self.save_state_widget = QLineEdit(self.state_filename)
        self.save_state_widget = QLineEdit('State not saved/loaded.')
        self.save_state_widget.setReadOnly(True)
        self.save_state_widget.setEnabled(False)

        # Add state widgets to layout
        save_state_layout.addWidget(self.save_state_widget)
        save_state_layout.addWidget(save_state_button)
        save_state_layout.addWidget(load_state_button)

        # Set default filename
        self.filename = ''
        
        # Create a save file layout
        save_file_layout = QHBoxLayout()
        
        # Create a push button to choose file
        save_file_button = QPushButton('Save file')

        # Connect button to function
        save_file_button.clicked.connect(self.set_filename)

        # Create a QLineEdit widget to display the file
        self.save_file_widget = QLineEdit(self.filename)

        # Set read only and disabled
        self.save_file_widget.setReadOnly(True)
        self.save_file_widget.setEnabled(False)

        # Add file widgets to layout        
        save_file_layout.addWidget(self.save_file_widget)
        save_file_layout.addWidget(save_file_button)

        # Add file layout to main layout
        self.layout.addLayout(save_file_layout)

        # Add state widgets to layout
        # Create a save file layout
        save_file_layout = QHBoxLayout()

        # Create checkbox for save mode
        self.save_checkbox = QCheckBox('Save mode', self)

        # Set save_mode boolean to False
        self.save_mode = False

        # Connect checkbox to function
        self.save_checkbox.checkStateChanged.connect(self.toggle_save_mode)

        # Add save file widgets to layout
        save_file_layout.addWidget(self.save_checkbox)

        # Add save file layout to main layout
        self.layout.addLayout(save_file_layout)

        # ================== Set region boundaries section ==================

        # Create region_boundaries selection
        title = QLabel(f"Region boundaries: ")
        self.layout.addWidget(title)

        # Create a region selector
        self.region_selector = QComboBox()
        self.current_region = 0
        
        # Add regions to the region selector
        for i in range(n_region):
            self.region_selector.insertItem(i, f"Region {i}")

        # Connect region selector to function
        self.region_selector.currentIndexChanged.connect(self.update_current_region)

        # Add region selector to main layout
        self.layout.addWidget(self.region_selector)
        
        # Create a layout for region boundaries
        region_boundaries_layout = QHBoxLayout()
        self.layout.addLayout(region_boundaries_layout)
        
        # Number of corners of the region
        self.n_region_corners = 4

        # Create region boundaries array
        # Each region has 4 boundaries
        self.region_boundaries = np.zeros((n_region, self.n_region_corners), dtype = int)
        
        # Set default region boundaries
        for i in range(n_region):
            self.region_boundaries[i] = [0, temperature.resolution[1], 0, temperature.resolution[0]]

        # Create region boundaries input and display widgets        
        self.region_boundaries_input = np.zeros(self.n_region_corners, dtype = QLineEdit)
        self.region_boundaries_display = np.zeros(self.n_region_corners, dtype = QLineEdit)
        
        # Create a QLineEdit widget for region boundaries
        boundaries_titles = [QLabel("X_min: "), QLabel("X_max: "), QLabel("Y_min"), QLabel("Y_max")]
        
        for i in range(self.n_region_corners):

            # Add entry line for the current region boundary
            line_edit = QLineEdit()

            # Add region boundary entry to the array
            self.region_boundaries_input[i] = line_edit

            # Connect region boundary entry to region boundary setter
            self.region_boundaries_input[i].returnPressed.connect(self.set_region_boundaries)

            # Add region boundary widget to control layout
            region_boundaries_layout.addWidget(boundaries_titles[i])
            region_boundaries_layout.addWidget(self.region_boundaries_input[i])

            # Create a line widget to display entered text
            boundaries_lineEdit = QLineEdit()
            boundaries_lineEdit.setReadOnly(True)
            boundaries_lineEdit.setEnabled(False)
            boundaries_lineEdit.setText(str(self.region_boundaries[0][i]))
            self.region_boundaries_display[i] = boundaries_lineEdit
            region_boundaries_layout.addWidget(self.region_boundaries_display[i])

        # ================== Set MFC and temperature section ==================
        # Set MFC control section
        # Add MFC control section to main layout
        title = QLabel("Temperature/MFC setpoint: ")
        self.layout.addWidget(title)

        # Create a layout for MFC temperature selector
        mfc_temperature_selector = QHBoxLayout()

        # Create a checkbox for MFC/temperature control mode
        # When toggled, it will enable/disable the temperature input
        # Untoggled, it controls the mfc flow rate straight
        # Temperature mode applies the PID control to control MFCs flow rate according to temperature setpoint
        self.mfc_temperature_checkbox = QCheckBox('Temperature control mode - MFCs adjust according to temperature setpoint', self)

        # Connect checkbox to function
        self.mfc_temperature_checkbox.checkStateChanged.connect(self.toggle_mfc_temperature_edit)
        
        # Add checkbox to layout
        mfc_temperature_selector.addWidget(self.mfc_temperature_checkbox)
        
        self.scheduler_filename = ''

        self.scheduler_checkbox = QCheckBox('Scheduler', self)

        self.scheduler_checkbox.checkStateChanged.connect(self.toggle_scheduler)

        mfc_temperature_selector.addWidget(self.scheduler_checkbox)
     

        self.decoupler_checkbox = QCheckBox('Decoupler', self)
        self.decoupler_checkbox.setVisible(False)  # Initially hidden
        
        mfc_temperature_selector.addWidget(self.decoupler_checkbox)
        
        # Show decoupler checkbox only if temperature control mode is selected
        self.mfc_temperature_checkbox.toggled.connect(lambda: self.decoupler_checkbox.setVisible(self.mfc_temperature_checkbox.isChecked()))
        
        # Add checkbox to main layout
        self.layout.addLayout(mfc_temperature_selector)
        
        # Create a layout for MFC and temperature
        self.temperature_mfc_edit_layout = QVBoxLayout()
        self.layout.addLayout(self.temperature_mfc_edit_layout)

        # Maximum number of columns in the grid
        self.n_columns_mfc_temperature_grid = 15
        
        # By default, temperature control mode is disabled
        self.create_mfc_section()

        # ================== Set figures section ==================
        # Add figures title to main layout
        title = QLabel("Figures: ")
        self.layout.addWidget(title)

        # Create a Matplotlib plot
        # Start from create checkbox to update plot information
        # By default, update plot is enabled
        # When disabled, the plot will not be updated, improving performance
        self.update_plot_checkbox = QCheckBox('Update plot', self)
        self.update_plot_checkbox.setChecked(True)
        
        # Add checkbox to layout
        self.layout.addWidget(self.update_plot_checkbox)
        
        # Create a layout for min and max temperature
        min_max_temperature_layout = QHBoxLayout()
        
        # Add min and max temperature to layout
        self.layout.addLayout(min_max_temperature_layout)

        # Create a QLineEdit widget for min temperature
        min_temperature_label = QLabel("Min temperature: ")
        self.min_temperature_input = QLineEdit()

        # Create a QLineEdit widget for displaying min temperature
        self.min_temperature_display = QLineEdit()
        self.min_temperature_display.setReadOnly(True)
        self.min_temperature_display.setEnabled(False)

        # Set default min temperature
        self.min_temperature_display.setText(str(temperature.min))

        # Connect min temperature entry to min temperature setter
        self.min_temperature_input.returnPressed.connect(self.set_min_max_temperature_limits)

        # Add min temperature widgets to layout
        min_max_temperature_layout.addWidget(min_temperature_label)
        min_max_temperature_layout.addWidget(self.min_temperature_input)
        min_max_temperature_layout.addWidget(self.min_temperature_display)

        # Create a QLineEdit widget for max temperature
        max_temperature_label = QLabel("Max temperature: ")
        self.max_temperature_input = QLineEdit()

        # Create a QLineEdit widget for displaying max temperature
        self.max_temperature_display = QLineEdit()
        self.max_temperature_display.setReadOnly(True)
        self.max_temperature_display.setEnabled(False)

        # Set default max temperature
        self.max_temperature_display.setText(str(temperature.max))

        # Connect max temperature entry to max temperature setter
        self.max_temperature_input.returnPressed.connect(self.set_min_max_temperature_limits)

        # Add max temperature widgets to layout
        min_max_temperature_layout.addWidget(max_temperature_label)
        min_max_temperature_layout.addWidget(self.max_temperature_input)
        min_max_temperature_layout.addWidget(self.max_temperature_display)

        # Create a layout for number of points in the plot
        n_plot_points_layout = QHBoxLayout()

        # Add number of points to layout
        self.layout.addLayout(n_plot_points_layout)

        # Set default number of points in the plot
        self.n_plot_points = 15

        # Title of the number of points
        n_plot_points_label = QLabel("Number of points in the plot: ")
        n_plot_points_layout.addWidget(n_plot_points_label)

        # Create a widget for number of points in the plot
        self.n_plot_points_input = QLineEdit()
        self.n_plot_points_input.returnPressed.connect(self.change_n_plot_points)

        # Create a widget for displaying number of points in the plot
        self.n_plot_points_display = QLineEdit()
        self.n_plot_points_display.setReadOnly(True)
        self.n_plot_points_display.setEnabled(False)
        self.n_plot_points_display.setText(str(self.n_plot_points))

        # Add number of points widgets to layout
        n_plot_points_layout.addWidget(self.n_plot_points_input)
        n_plot_points_layout.addWidget(self.n_plot_points_display)

        # Create a matplotlib figure with 3 subplots: MFC outputs, Infrared camera heatmap, Average temperature per region
        self.figure, self.ax = plt.subplots(1, 4, width_ratios=[1, 1, 1, 0.3])
        
        max_ticks = 6
        self.ax[0].xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
        self.ax[2].xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
        
        self.ax[0].xaxis.set_major_formatter(FormatStrFormatter('% 1.0f'))
        self.ax[0].yaxis.set_major_formatter(FormatStrFormatter('% 1.1f'))
        
        self.ax[2].xaxis.set_major_formatter(FormatStrFormatter('% 1.0f'))
        self.ax[2].yaxis.set_major_formatter(FormatStrFormatter('% 1.1f'))
        
        self.ax[0].set_title("MFC output")
        self.ax[1].set_title("Infrared camera")
        self.ax[2].set_title("Average temperature")
        
        self.ax[0].set_xlabel("Time [s]")
        self.ax[1].set_xlabel("X pixels")
        self.ax[2].set_xlabel("Time [s]")

        
        self.ax[0].set_ylabel("Air flow rate [L/min]")
        self.ax[1].set_ylabel("Y Pixels")
        self.ax[2].set_ylabel(r"Temperature [$^o$C]")

        # Create a heatmap of the temperature
        self.temperature_image = self.ax[1].imshow(temperature.temperature_grid, cmap="turbo", interpolation = None, vmin = temperature.min, vmax = temperature.max)

        # Create a colorbar for the temperature image
        self.figure.heatmap_colorbar = self.figure.colorbar(self.temperature_image)

        # Create a list of patches to draw the regions
        self.patches = []
        
        # Create a array of time, flow rate, temperature and temperature setpoints for plotting
        self.time_plot = np.zeros(self.n_plot_points)
        self.flow_rate_plot = np.zeros((n_region, self.n_plot_points))
        self.temperature_plot = np.zeros((n_region, self.n_plot_points))

        # Create a list of patches to draw the regions from selector in heatmap
        for i in range(n_region):

            # Create a rectangle for each region
            self.patches.append(patches.Rectangle((self.region_boundaries[i][0] - .5,self.region_boundaries[i][2] - 0.5), width = self.region_boundaries[i][1] - self.region_boundaries[i][0] + 1, height = self.region_boundaries[i][3] - self.region_boundaries[i][2] + 1, linewidth = 2, edgecolor = self.colors_qualitative[i], facecolor = 'none'))
            
            # Add rectangles to figure
            self.temperature_image.axes.add_patch(self.patches[i])

            # Create a canvas for the figure
            self.ax[0].plot(self.time_plot, self.flow_rate_plot[i, :], '.', color = self.colors_qualitative[i])  # Plot new data
            self.ax[2].plot(self.time_plot, self.temperature_plot[i, :], '.', color = self.colors_qualitative[i], label = f'Region {i}')  # Plot new data

        # Place legend outside of the plot
        lines = self.ax[2].get_lines()
        labels = [i.get_label() for i in lines]
        self.ax[3].axis('off')
        self.ax[3].legend(lines, labels, loc = 'center')

        # Create a canvas for the figure
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)

        # Add canvas to layout
        self.layout.addWidget(self.canvas)
        
        # Set layout and present window
        self.setLayout(self.layout)
        self.show()

    # ================== Functions ==================

    def create_mfc_section(self):
        '''Create MFC section'''

        # Create layout for MFCs
        self.mfc_edit_layout = QGridLayout()

        # Add MFC section to temperature_mfc layout
        self.temperature_mfc_edit_layout.addLayout(self.mfc_edit_layout)

        # Create MFC input and display widgets
        self.mfc_input = np.empty(self.n_region, dtype=QLineEdit)
        self.mfc_display = np.empty(self.n_region, dtype=QLineEdit)

        # Start grid row, if passed the number of columns, start a new row
        grid_row = 0
        grid_column = 0

        for i in range(self.n_region):
            # Layout for the current MFC
            current_mfc_layout = QVBoxLayout()

            # Title of the MFC
            title = QLabel(f"MFC {i}: ")
            current_mfc_layout.addWidget(title)

            # Add horizontal layout to current MFC layout
            current_mfc_horizontal_layout = QHBoxLayout()
            current_mfc_layout.addLayout(current_mfc_horizontal_layout)

            # Add entry line for the current MFC
            self.mfc_input[i] = QLineEdit()
            self.mfc_input[i].returnPressed.connect(self.set_mfc)
            current_mfc_horizontal_layout.addWidget(self.mfc_input[i])

            # Create a line widget to display entered text
            self.mfc_display[i] = QLineEdit()
            self.mfc_display[i].setReadOnly(True)
            self.mfc_display[i].setEnabled(False)
            self.mfc_display[i].setText('0')
            current_mfc_horizontal_layout.addWidget(self.mfc_display[i])

            # Make a frame to wrap around the current MFC layout
            frame = QFrame()
            frame.setFrameShape(QFrame.Box)       # Box-shaped frame
            frame.setFrameShadow(QFrame.Raised)   # Raised effect
            frame.setLayout(current_mfc_layout)

            # Add the frame (with the layout inside) to the grid layout
            self.mfc_edit_layout.addWidget(frame, grid_row, grid_column)

            # Increment grid column
            grid_column += 3
            if grid_column >= self.n_columns_mfc_temperature_grid:
                grid_row += 1
                grid_column = 0

                
    def create_temperature_section(self):
        '''Create temperature section'''
        # Create layout for Temperatures
        temperature_edit_layout = QGridLayout()

        # Add temperature section to temperature_mfc layout
        self.temperature_mfc_edit_layout.addLayout(temperature_edit_layout)
        
        # Create array for temperature setpoints
        dummy_large_temperature_setpoint = 100000000
        self.temperature_setpoint = np.repeat(dummy_large_temperature_setpoint, self.n_region)

        # Create temperature input and display widgets
        self.temperature_input = np.empty(self.n_region, dtype=QLineEdit)
        self.temperature_display = np.empty(self.n_region, dtype=QLineEdit)

        # Create PID containers
        # Parameters of the PID controllers
        self.n_controller_parameters = 3
        self.pid_input = [np.empty(self.n_controller_parameters, dtype = QLineEdit) for _ in range(self.n_region)]
        self.pid_display = [np.empty(self.n_controller_parameters, dtype = QLineEdit) for _ in range(self.n_region)]

        # Restart grid row and column
        grid_row = 0
        grid_column = 0

        # Create a QLineEdit widget for temperature setpoint
        for i in range(self.n_region):

            # Layout for the current temperature section
            current_temperature_layout = QVBoxLayout()

            # Title of the temperature
            title = QLabel(f"Temperature {i}: ")
            current_temperature_layout.addWidget(title)

            # Horizontal layout for temperature input and display
            current_temperature_horizontal_layout = QHBoxLayout()
            current_temperature_layout.addLayout(current_temperature_horizontal_layout)

            # Add entry line for the current temperature
            self.temperature_input[i] = QLineEdit()
        
            # Connect temperature entry to temperature setter
            self.temperature_input[i].returnPressed.connect(self.set_temperature)
            
            # Add temperature widget to control layout
            current_temperature_horizontal_layout.addWidget(self.temperature_input[i])

            # Create a QLineEdit widget to display entered text
            self.temperature_display[i] = QLineEdit()
            self.temperature_display[i].setReadOnly(True)
            self.temperature_display[i].setEnabled(False)
            self.temperature_display[i].setText('---')

            # Add temperature display widget to the layout
            current_temperature_horizontal_layout.addWidget(self.temperature_display[i])

            # Add PID to current temperature layout
            self.create_pid_section(region = i, current_temperature_layout = current_temperature_layout)

            # Make a frame to wrap around the current temperature layout
            frame = QFrame()
            frame.setFrameShape(QFrame.Box)       # Box-shaped frame
            frame.setFrameShadow(QFrame.Raised)   # Raised effect

            # Add the current temperature layout to the frame
            frame.setLayout(current_temperature_layout)

            # Add the frame (with the layout inside) to the grid layout
            temperature_edit_layout.addWidget(frame, grid_row, grid_column)
            
            # Increment grid column
            grid_column += 3

            # Adjust grid row and column for next widget
            if grid_column >= self.n_columns_mfc_temperature_grid:
                grid_row += 1
                grid_column = 0


    def create_pid_section(self, region, current_temperature_layout):
        # Add PID controller section to main layout
        title = QLabel("PID parameters: ")
        current_temperature_layout.addWidget(title)

        # Add PID controller gains setter area
        control_parameter_names = ["Proportional: ", "Integral: ", "Derivative: "]
        controller_parameter_edit_layout = QVBoxLayout()
        current_temperature_layout.addLayout(controller_parameter_edit_layout)

        # Create controller parameter setter
        controller_parameter = np.zeros(self.n_controller_parameters)
        for i in range(self.n_controller_parameters):

            # Horizontal layout per parameters
            controller_gain_edit_layout = QHBoxLayout()
            controller_parameter_edit_layout.addLayout(controller_gain_edit_layout)

            # Title of the gains
            controller_parameter_title = QLabel(control_parameter_names[i])

            # Add field for each controller's gain
            control_edit_line = QLineEdit()
            self.pid_input[region][i] = control_edit_line
            
            # Set parameter upon changing value
            # When value is changed, set the parameter using the function set_pid_gains
            self.pid_input[region][i].returnPressed.connect(self.set_pid_gains)

            # Add PID widget to control layout
            controller_gain_edit_layout.addWidget(controller_parameter_title)
            controller_gain_edit_layout.addWidget(self.pid_input[region][i])

            # Add line with PID gains value
            controller_gain_lineEdit = QLineEdit()
            controller_gain_lineEdit.setReadOnly(True)
            controller_gain_lineEdit.setEnabled(False)
            controller_gain_lineEdit.setText(str(controller_parameter[i]))
            self.pid_display[region][i] = controller_gain_lineEdit
            controller_gain_edit_layout.addWidget(self.pid_display[region][i])


    def toggle_scheduler(self):
        '''Scheduler. Called when toggled'''

        # If scheduler is enabled
        # Restart setpoints for MFCs and temperature
        self.flow_rate_setpoint = np.zeros(self.n_region)
        self.temperature_setpoint = np.repeat(None, self.n_region)

        if not self.mfc_temperature_checkbox.isChecked():
            self.clear_layout(self.mfc_edit_layout)

        if self.scheduler_checkbox.isChecked():
            self.mfc_temperature_checkbox.setEnabled(False)
            self.create_scheduler_section()
            if self.mfc_temperature_checkbox.isChecked():
                for i in range(self.n_region):
                    self.temperature_input[i].setReadOnly(True)
                    self.temperature_input[i].setEnabled(False)
                    self.temperature_input[i].setText('---')
                    self.temperature_display[i].setText('---')
        else:
            self.clear_layout(self.scheduler_layout)
            self.mfc_temperature_checkbox.setEnabled(True)
            if self.mfc_temperature_checkbox.isChecked():
                for i in range(self.n_region):
                    self.temperature_input[i].setReadOnly(False)
                    self.temperature_input[i].setEnabled(True)
                    self.temperature_input[i].setText('')
                    self.temperature_display[i].setText('---')
            else:
                self.create_mfc_section()

    def create_scheduler_section(self):
        '''Create scheduler section'''

        self.scheduler_data = np.zeros((1, self.n_region + 1))

        # Create new layout (scheduler layout) for the scheduler section 
        self.scheduler_layout = QVBoxLayout()
        # Add scheduler section to temperature_mfc layout
        self.temperature_mfc_edit_layout.addLayout(self.scheduler_layout)

        # Create a QLineEdit widget for scheduler (read-only text box to display chosen scheduler file)
        scheduler_file_line = QLineEdit()
        scheduler_file_line.setReadOnly(True)
        scheduler_file_line.setEnabled(False)
        self.scheduler_layout.addWidget(scheduler_file_line)

        # Create a textbox for displaying scheduler data
        self.scheduler_text_layout = QGridLayout()
        self.scheduler_layout.addLayout(self.scheduler_text_layout)
        
        scheduler_current_time_label = QLabel('Current time interval: ')
        self.scheduler_current_time = QLineEdit()
        self.scheduler_current_time.setReadOnly(True)
        self.scheduler_current_time.setEnabled(False)
        if self.scheduler_data.shape[0] > 1:
            self.scheduler_current_time.setText(str(self.scheduler_data[0][0]) + " --- " + str(self.scheduler_data[1][0]))
        else:
            self.scheduler_current_time.setText(str(self.scheduler_data[0][0]) + " --- end")

        scheduler_current_state_label = QLabel('Current state per region: ')
        self.scheduler_current_state = QLineEdit()
        self.scheduler_current_state.setReadOnly(True)
        self.scheduler_current_state.setEnabled(False)
        self.scheduler_current_state.setText(str(self.scheduler_data[0][1:]))      

        self.scheduler_text_layout.addWidget(scheduler_current_time_label, 0, 0)
        self.scheduler_text_layout.addWidget(self.scheduler_current_time, 0, 1)
        self.scheduler_text_layout.addWidget(scheduler_current_state_label, 1, 0)
        self.scheduler_text_layout.addWidget(self.scheduler_current_state, 1, 1)

        self.scheduler_filename = QFileDialog.getOpenFileName(self, 'Choose scheduler file', os.path.expanduser('~'), 'Text files (*.csv)')[0]

        if len(self.scheduler_filename) < 1:
            scheduler_file_line.setText('File not chosen. Please, choose a file with scheduling information.')
            return

        else:

            # Read csv with time, and mfc stepoints
            self.scheduler_data = np.genfromtxt(self.scheduler_filename, delimiter = ',')

            # Update file display
            scheduler_file_line.setText(self.scheduler_filename)

            if self.scheduler_data.shape[0] > 1:
                self.scheduler_current_time.setText(str(self.scheduler_data[0][0]) + " --- " + str(self.scheduler_data[1][0]))
                self.scheduler_change_time = self.scheduler_data[1][0]
            else:
                self.scheduler_current_time.setText(str(self.scheduler_data[0][0]) + " --- end")
                self.scheduler_change_time = -1

            self.scheduler_current_state.setText(str(self.scheduler_data[0][1:]))

    def set_min_max_temperature_limits(self):
        '''Set minimum and maximum temperature limits'''

        # Get entry to min and max temperature
        min_temperature = self.min_temperature_input.text()
        max_temperature = self.max_temperature_input.text()

        # If min temperature is not empty
        if len(min_temperature) > 0:
            self.temperature.min = float(min_temperature)
            self.min_temperature_input.clear()
            self.min_temperature_display.setText(min_temperature)

        # If max temperature is not empty
        if len(max_temperature) > 0:
            self.temperature.max = float(max_temperature)
            self.max_temperature_input.clear()
            self.max_temperature_display.setText(max_temperature)

        self.temperature_image = self.ax[1].imshow(self.temperature.temperature_grid, cmap="turbo", interpolation = None, vmin = self.temperature.min, vmax = self.temperature.max)

        self.figure.heatmap_colorbar.remove()

        self.figure.heatmap_colorbar = self.figure.colorbar(self.temperature_image)
    
    def toggle_mfc_temperature_edit(self):
        '''Temperature/MFC control mode. Called when toggled'''

        # If temperature control mode is enabled
        # Restart setpoints for MFCs and temperature
        self.flow_rate_setpoint = np.zeros(self.n_region)
        self.temperature_setpoint = np.repeat(None, self.n_region)
        
        for i in range(self.n_region):
            
            # Reset MFCs flow rate
            self.MFC.set_flow_rate(i, 0)

        self.clear_layout(self.temperature_mfc_edit_layout)

        if self.mfc_temperature_checkbox.isChecked():
            self.create_temperature_section()
            self.setpoint_plot = np.full((self.n_region, self.n_plot_points), np.nan)  # Initialize with NaN
            self.valid_temperature_setpoint = np.full(self.n_region, False)  # Track validity of setpoints
        else:
            self.create_mfc_section()


    def clear_layout(self, layout):
        '''Function to delete all layouts from a parent layout'''

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            
            # If the item is a widget, delete it
            if widget is not None:
                widget.deleteLater()
            
            # If the item is a layout, recursively clear it
            elif child_layout is not None:
                self.clear_layout(child_layout)
        
        # Delete the layout item itself
        # layout.removeItem(item)
    
    def toggle_save_mode(self):
        '''Save mode. Called when toggled'''

        # Set save mode if file exists and is valid
        if len(self.filename) > 0:
            self.save_mode = self.save_checkbox.isChecked()

        # If file does not exist or is invalid, disable save mode and uncheck checkbox
        else:
            self.save_file_widget.setText('File not chosen. Please, choose a file to save data.')
            self.save_checkbox.setChecked(False)
            self.save_mode = False

    def set_filename(self):
        '''Choose file to save data'''

        self.filename = QFileDialog.getSaveFileName(self, 'Choose save file name', os.path.expanduser('~'), 'Text files (*.csv)')[0]

        if len(self.filename) < 1:
            self.save_file_widget.setText('File not chosen. Please, choose a file to save data.')
            return

        if '.csv' not in self.filename:
            self.filename += '.csv'

        # Update save state file display
        self.save_file_widget.setText(self.filename)

        # Set filename to the new filename
        self.save_file_widget.setText(self.filename)

        # Create a new file
        with open(self.filename, 'w') as file:
            
            # Write header
            header = 'time'

            # Add MFC headers
            for i in range(self.n_region):
                header += f', mfc_{i}'

            # Add Temperature headers
                for i in range(self.n_region):
                    header += f', temperature_{i}'

            # Executes if temperature control mode is enabled (be sure to create file and save data after clicking the checkbox)
            if self.mfc_temperature_checkbox.isChecked():
                # Add Temperature Setpoint headers
                for i in range(self.n_region):
                    header += f', temperature_setpoint_{i}'
                    # Add PID headers for each region
                for i in range(self.n_region):
                        header += f', P_{i}'
                for i in range(self.n_region):
                        header += f', I_{i}'
                for i in range(self.n_region):
                        header += f', D_{i}'

            for i in range(self.n_region):
                header += f', region_{i}_x_min, region_{i}_x_max, region_{i}_y_min, region_{i}_y_max'

            header += '\n'
            
            file.write(header)

        # Create file for temperature
        with open(self.filename.replace('.csv', '_temp.csv'), 'w') as file:
            header = 'time, temperature\n'
            file.write(header)
            

    def set_pid_gains(self):
        '''Set PID controller gains'''

        # Set controller parameters
        for i in range(self.n_region):
            for j in range(self.n_controller_parameters):
                if len(self.pid_input[i][j].text()) > 0:
                    self.PID[i].gains[j] = float(self.pid_input[i][j].text())
                    self.pid_display[i][j].setText(self.pid_input[i][j].text())
                    self.pid_input[i][j].clear()

        # Update display with gains of the current region
        for i in range(self.n_region):
            for j in range(self.n_controller_parameters):
                self.pid_display[i][j].setText(str(self.PID[i].gains[j]))
        

    def set_mfc(self):
        '''Set MFC flow rate upon changing value'''
        for i in range(self.n_region):

            # Only positive inputs are valid
            # TODO: Limit maximum flow rate
            if len(self.mfc_input[i].text()) > 0:
                
                # Set flow rate
                self.MFC.set_flow_rate(i, float(self.mfc_input[i].text()))
                
                #Update display
                self.mfc_display[i].setText(self.mfc_input[i].text())
                
                # Clear input line
                self.mfc_input[i].clear()
        
    # Set temperature setpoint
    def set_temperature(self):
        '''Set temperature setpoint upon changing value
        Only works if temperature control mode is enabled
        '''

        # Set temperature setpoint for each region
        for i in range(self.n_region):

            # Only positive inputs are valid
            if len(self.temperature_input[i].text()) > 0:

                # Set temperature setpoint
                self.temperature_setpoint[i] = float(self.temperature_input[i].text())

                # Update display
                self.temperature_display[i].setText(self.temperature_input[i].text())

                # Clear input line
                self.temperature_input[i].clear()
            
    
    def set_region_boundaries(self, from_state_file = False):
        '''Set region boundaries upon changing value'''

        # For each corner of the region
        for i in range(self.n_region_corners):

            # Get information from file
            if from_state_file:
                text = self.load_region_boundaries[self.current_region][i]
            
            # Get text from input line
            else:
                text = self.region_boundaries_input[i].text()
            
                # If text is empty
                if len(text) < 1:
                    continue

            # Only accept values within the resolution of the camera
            if i in [0,2] and int(text) < 0: text = 0
            if i == 1 and int(text) > self.temperature.resolution[1] - 1: text = self.temperature.resolution[1] - 1
            if i == 3 and int(text) > self.temperature.resolution[0] - 1: text = self.temperature.resolution[0] - 1

            # Set region boundaries
            self.region_boundaries[self.current_region][i] = int(text)
            
            # Update patches in figure
            self.update_patches()

            # Update display
            self.region_boundaries_display[i].setText(str(self.region_boundaries[self.current_region][i]))

            # Clear input line
            self.region_boundaries_input[i].clear()

    def update_patches(self):
        '''Update patches in figure'''

        self.patches[self.current_region].set_bounds(self.region_boundaries[self.current_region][0] - 0.5, self.region_boundaries[self.current_region][2] - .5, self.region_boundaries[self.current_region][1] - self.region_boundaries[self.current_region][0] + 1, self.region_boundaries[self.current_region][3] - self.region_boundaries[self.current_region][2] + 1)

    def update_current_region(self):
        '''Update displays for current region upon changing in region selector'''
        self.current_region = self.region_selector.currentIndex()

        # Updage patches according to the current region
        for i in range(self.n_region_corners):
            self.region_boundaries_display[i].setText(str(self.region_boundaries[self.current_region][i]))        

    def update_plot(self, time, temperature, MFC):
        '''Update all plots in the figure'''

        # If update plot is enabled
        if self.update_plot_checkbox.isChecked():

            # Update temperature heatmap according to new temperature information
            self.temperature_image.set_data(temperature.temperature_grid)
        
            # Update time array
            self.time_plot = self.time_plot[1:]
            self.time_plot = np.append(self.time_plot, time)

            # Update flow rate array only if not in test_UI mode
            self.flow_rate_plot = np.delete(self.flow_rate_plot, (0), axis = 1)
            self.flow_rate_plot = np.append(self.flow_rate_plot, np.transpose([MFC.flow_rate]), axis = 1)

            # Update temperature on plot        
            self.temperature_plot = np.delete(self.temperature_plot, (0), axis = 1)
            self.temperature_plot = np.append(self.temperature_plot, np.transpose([temperature.temperature_average]), axis = 1)

            if self.mfc_temperature_checkbox.isChecked():
                self.setpoint_plot = np.delete(self.setpoint_plot, (0), axis=1)

                new_setpoint = np.full(self.n_region, np.nan)  # Default to NaN
                for i in range(self.n_region):
                    if self.temperature_setpoint[i] is not None and 0 <= self.temperature_setpoint[i] <= 1000:
                        new_setpoint[i] = self.temperature_setpoint[i]
                        self.valid_temperature_setpoint[i] = True  # Mark as valid
                    else:
                        self.valid_temperature_setpoint[i] = False  # Mark as invalid

                self.setpoint_plot = np.append(self.setpoint_plot, np.transpose([new_setpoint]), axis=1)


            # Update plot information per region 
            [self.update_single_plot(i) for i in range(self.n_region)]

            # Update setpoint on plot
            

            # Update plot limits
            self.ax[0].relim()
            self.ax[2].relim()

            # Autoscale plot
            self.ax[0].autoscale_view()
            self.ax[2].autoscale_view()
        
            # Draw the plot
            self.canvas.draw_idle()

    
    def update_single_plot(self, i):
        '''Update plot information per region'''

        self.ax[0].get_children()[i].set_data(self.time_plot,self.flow_rate_plot[i, :])

        # Update the temperature plot
        self.ax[2].get_children()[i].set_data(self.time_plot,self.temperature_plot[i, :])

        # Ensure the setpoint is plotted as a separate curve
        if self.mfc_temperature_checkbox.isChecked() and self.valid_temperature_setpoint[i]:
            if hasattr(self, 'setpoint_lines') and len(self.setpoint_lines) > i:
                self.setpoint_lines[i].set_data(self.time_plot, self.setpoint_plot[i, :])
            else:
                if not hasattr(self, 'setpoint_lines'):
                    self.setpoint_lines = []
                line, = self.ax[2].plot(self.time_plot, self.setpoint_plot[i, :], '--', color=self.colors_qualitative[i], label=f'Setpoint {i}')
                self.setpoint_lines.append(line)
        

    def change_n_plot_points(self):
        '''Change number of points in the plot'''

        # Get new number of points
        new_n_plot_points = int(self.n_plot_points_input.text())

        # If new number of points is valid
        if new_n_plot_points > 0:

            # If new number of points is greater than the current number of points
            if new_n_plot_points > self.n_plot_points:
                    
                    # Create new time points
                    first_time_point = self.time_plot[0] - new_n_plot_points * self.time_step
                    added_time_points = np.linspace(first_time_point, self.time_plot[0], new_n_plot_points - self.n_plot_points)

                    # Flip the array to have the first time point first
                    added_time_points = np.flip(added_time_points)

                    # Guarantee that time points are not negative
                    added_time_points[added_time_points < 0] = 0
                    
                    # Update time array
                    self.time_plot = np.append(added_time_points, self.time_plot)
    
                    # Update flow rate array
                    self.flow_rate_plot = np.append(np.zeros((self.n_region, new_n_plot_points - self.n_plot_points)), self.flow_rate_plot, axis = 1)
    
                    # Update temperature array
                    self.temperature_plot = np.append(np.zeros((self.n_region, new_n_plot_points - self.n_plot_points)), self.temperature_plot, axis = 1)

                    # Update setpoint array
                    if self.mfc_temperature_checkbox.isChecked() and self.temperature_setpoint[i] >= 0 and self.temperature_setpoint[i] < 1000:
                        self.setpoint_plot = np.append(np.zeros((self.n_region, new_n_plot_points - self.n_plot_points)), self.setpoint_plot, axis = 1)

            elif new_n_plot_points < self.n_plot_points:

                # If new number of points is smaller than the current number of points
                self.time_plot = self.time_plot[-new_n_plot_points:]
                self.flow_rate_plot = self.flow_rate_plot[:, -new_n_plot_points:]
                self.temperature_plot = self.temperature_plot[:, -new_n_plot_points:]
                if self.mfc_temperature_checkbox.isChecked() and self.temperature_setpoint[i] >= 0 and self.temperature_setpoint[i] < 1000:
                        self.setpoint_plot = self.setpoint_plot[:, -new_n_plot_points:]

            # Update number of points
            self.n_plot_points = new_n_plot_points

            # Update display
            self.n_plot_points_display.setText(str(self.n_plot_points))

            # Clear input line
            self.n_plot_points_input.clear()


    def save_state(self):
        '''Save current parameters to restart experiments'''

        self.state_filename = QFileDialog.getSaveFileName(self, 'Choose save state file name', os.path.expanduser('~'), 'Text files (*.txt)')[0]

        if '.txt' not in self.state_filename:
            self.state_filename += '.txt'

        # Update save state file display
        self.save_state_widget.setText(self.state_filename)

        # Create dictionary to save parameters
        experimental_parameters = {}

        experimental_parameters["Temperature mode"] = self.mfc_temperature_checkbox.isChecked()

        if self.mfc_temperature_checkbox.isChecked():
            for i in range(self.n_region):
                pid_gains = [self.pid_display[i][j].text() for j in range(self.n_controller_parameters)]
                experimental_parameters[f"Region {i}"] = {'Region boundaries': self.region_boundaries[i].tolist(), 'PID gains': pid_gains}

        else:
            for i in range(self.n_region):
                experimental_parameters[f"Region {i}"] = {'Region boundaries': self.region_boundaries[i].tolist()}
            
        # Save parameters to file
        with open(self.state_filename, 'w') as file:
            for key, value in experimental_parameters.items():
                file.write(f"{key}: {value}\n")


    def load_state(self):
        '''Load parameters from previous experiments'''

        self.state_filename = QFileDialog.getOpenFileName(self, 'Choose load state file name', os.path.expanduser('~'), 'Text files (*.txt)')[0]

        # Update state file display
        self.save_state_widget.setText(self.state_filename)

        experimental_parameters = {}

        with open(self.state_filename, "r") as file:
            for line in file:
                # Strip newline and split by the separator
                key, value = line.strip().split(": ", 1)
                
                # Store parameter into dictionary
                experimental_parameters[key] = value

        self.mfc_temperature_checkbox.setChecked(eval(experimental_parameters['Temperature mode']))
        if self.mfc_temperature_checkbox.isChecked():
            self.mfc_temperature_checkbox.toggled.emit(True)

        self.load_region_boundaries = []

        # Assign parameters to the current experiment
        for i in range(self.n_region):
            region_parameters = eval(experimental_parameters[f"Region {i}"])
            self.load_region_boundaries.append(np.array(region_parameters['Region boundaries']).astype(int))

            if self.mfc_temperature_checkbox.isChecked():
                self.PID[i].gains = np.array(region_parameters['PID gains']).astype(float)
            
        # Update boundaries of patches
        for i in range(self.n_region):
            self.current_region = i
            self.set_region_boundaries(from_state_file = True)

        self.current_region = 0

        # Update current region
        self.update_current_region()
        
        # Update PID gains
        self.set_pid_gains()

    def set_font_QLabels(self, font_size):    
        '''Apply the font to all QLabel instances in the app'''

        # Create a font
        title_font = QFont('Arial', font_size)
        title_font.setBold(True)

        for i in range(self.layout.count()):
            widget = self.layout.itemAt(i).widget()
            if isinstance(widget, QLabel):
                widget.setFont(title_font)

    
